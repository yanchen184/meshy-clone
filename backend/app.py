"""
meshy-clone 後端 — 上傳一張圖，用 TripoSR 生成 3D 模型 (.glb)。

引擎是開源的 TripoSR (stabilityai/TripoSR, MIT)，這裡只做「殼」：
- 模型在啟動時載入一次、常駐 GPU（Meshy 體感快的關鍵，不是每請求重載）
- /api/generate 收圖 → 去背 → 推論 → marching cubes 出 mesh → 匯出 .glb
- 產物落在 outputs/<job_id>/，前端用 three.js 讀回來檢視
"""

import io
import logging
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# 把 TripoSR repo 掛進 import path（引擎與殼分開放，不 fork 進來）
ENGINE_DIR = Path(os.environ.get("TRIPOSR_DIR", r"D:\projects\meshy-clone-engine"))
sys.path.insert(0, str(ENGINE_DIR))

import rembg  # noqa: E402
from tsr.system import TSR  # noqa: E402
from tsr.utils import remove_background, resize_foreground  # noqa: E402

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
log = logging.getLogger("meshy-clone")

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MC_RESOLUTION = int(os.environ.get("MC_RESOLUTION", "256"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "8192"))
FOREGROUND_RATIO = 0.85
# 文生圖模型（打字生 3D 的第一段）。SDXL-Turbo：1-4 步出圖、免 CFG、fp16 塞得下。
TXT2IMG_MODEL = os.environ.get("TXT2IMG_MODEL", "stabilityai/sdxl-turbo")

# ---- 全域單例：模型與去背 session 只載一次 ----
model: TSR | None = None
rembg_session = None
txt2img_pipe = None  # lazy：第一次打字生 3D 才載


def load_engine() -> None:
    """啟動時載入 TripoSR，常駐記憶體。"""
    global model, rembg_session
    t0 = time.time()
    log.info(f"Loading TripoSR on {DEVICE} ...")
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(CHUNK_SIZE)
    model.to(DEVICE)
    rembg_session = rembg.new_session()
    log.info(f"Engine ready in {time.time() - t0:.1f}s (device={DEVICE})")


def _load_txt2img():
    """lazy 載入 SDXL-Turbo。第一次呼叫才下載/載入（約 7GB 權重）。"""
    global txt2img_pipe
    if txt2img_pipe is not None:
        return txt2img_pipe
    from diffusers import AutoPipelineForText2Image

    t0 = time.time()
    log.info(f"Loading {TXT2IMG_MODEL} (fp16) ...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        TXT2IMG_MODEL, torch_dtype=torch.float16, variant="fp16"
    )
    txt2img_pipe = pipe
    log.info(f"txt2img ready in {time.time() - t0:.1f}s")
    return txt2img_pipe


def text_to_image(prompt: str) -> Image.Image:
    """文字 → 一張圖。SDXL-Turbo 用完把權重移回 CPU、清 GPU，
    把 VRAM 讓回給常駐的 TripoSR（12GB 卡塞不下兩個模型同時在 GPU）。"""
    pipe = _load_txt2img()
    pipe.to(DEVICE)
    try:
        # SDXL-Turbo 設計：num_inference_steps=1、guidance_scale=0.0（免 CFG）
        img = pipe(
            prompt=prompt, num_inference_steps=1, guidance_scale=0.0
        ).images[0]
    finally:
        pipe.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return img.convert("RGB")


app = FastAPI(title="meshy-clone", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    load_engine()


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok" if model is not None else "loading",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mc_resolution": MC_RESOLUTION,
    }


def image_to_glb(pil: Image.Image, job_dir: Path, remove_bg: bool = True) -> dict:
    """圖 → glb 的共用核心：去背 → 推論 → marching cubes → 匯出。
    圖片端點與文字端點都走這裡。回傳各段耗時。"""
    timings: dict[str, float] = {}

    # 1) 去背 + 置中（Meshy 也是這步先把主體摳乾淨）
    t = time.time()
    if remove_bg:
        img = remove_background(pil, rembg_session)
        img = resize_foreground(img, FOREGROUND_RATIO)
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
        proc = Image.fromarray((arr * 255.0).astype(np.uint8))
    else:
        proc = pil.convert("RGB")
    proc.save(job_dir / "input.png")
    timings["preprocess"] = time.time() - t

    # 2) 推論：圖 → scene codes
    t = time.time()
    with torch.no_grad():
        scene_codes = model([proc], device=DEVICE)
    timings["inference"] = time.time() - t

    # 3) marching cubes 出 mesh（vertex color，不烘貼圖 atlas，快且省 VRAM）
    t = time.time()
    meshes = model.extract_mesh(scene_codes, True, resolution=MC_RESOLUTION)
    timings["mesh"] = time.time() - t

    # 4) 匯出 .glb
    t = time.time()
    meshes[0].export(str(job_dir / "model.glb"))
    timings["export"] = time.time() - t

    return timings


def _glb_response(job_id: str, timings: dict) -> JSONResponse:
    total = sum(timings.values())
    log.info(f"[{job_id}] done in {total:.2f}s {timings}")
    return JSONResponse(
        {
            "job_id": job_id,
            "model_url": f"/outputs/{job_id}/model.glb",
            "input_url": f"/outputs/{job_id}/input.png",
            "timings_sec": {k: round(v, 3) for k, v in timings.items()},
            "total_sec": round(total, 3),
        }
    )


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    remove_bg: bool = True,
) -> JSONResponse:
    """收一張圖，回傳生成好的 .glb 相對 URL 與耗時。"""
    if model is None:
        raise HTTPException(503, "engine still loading, retry shortly")

    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"cannot read image: {e}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    timings = image_to_glb(pil, job_dir, remove_bg=remove_bg)
    return _glb_response(job_id, timings)


@app.post("/api/generate-from-text")
async def generate_from_text(prompt: str = Form(...)) -> JSONResponse:
    """打字生 3D：文字 → SDXL-Turbo 生圖 → 沿用同一條 TripoSR 管線 → glb。"""
    if model is None:
        raise HTTPException(503, "engine still loading, retry shortly")
    prompt = (prompt or "").strip()
    if not prompt:
        raise HTTPException(400, "prompt is empty")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 1) 文字 → 圖（SDXL-Turbo）。存下來讓前端也能秀「生成的中間圖」。
    t = time.time()
    try:
        gen_img = text_to_image(prompt)
    except Exception as e:  # noqa: BLE001
        log.exception("text_to_image failed")
        raise HTTPException(500, f"text-to-image failed: {e}")
    gen_img.save(job_dir / "generated.png")
    t2i = time.time() - t

    # 2) 生成的圖 → 走現有圖生 3D 管線
    timings = image_to_glb(gen_img, job_dir, remove_bg=True)
    timings = {"text2img": t2i, **timings}
    resp = _glb_response(job_id, timings)
    return resp


# 靜態產物：/outputs/<job>/model.glb 直接讓前端抓
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
