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
from fastapi import FastAPI, File, HTTPException, UploadFile
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

# ---- 全域單例：模型與去背 session 只載一次 ----
model: TSR | None = None
rembg_session = None


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
    glb_path = job_dir / "model.glb"
    meshes[0].export(str(glb_path))
    timings["export"] = time.time() - t

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


# 靜態產物：/outputs/<job>/model.glb 直接讓前端抓
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
