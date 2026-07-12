# Bob 3D — 一鍵啟動
# 後端 FastAPI (port 8000) + 前端靜態站 (port 5173)
# 用法：在 D:\projects\meshy-clone 底下開 PowerShell 跑 .\start.ps1

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$engine = "D:\projects\meshy-clone-engine"
$py = Join-Path $engine ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "找不到 TripoSR 環境的 python：$py"
    exit 1
}

Write-Host "啟動後端 (FastAPI :8000) ..." -ForegroundColor Cyan
$env:TRIPOSR_DIR = $engine
Start-Process -FilePath $py -ArgumentList "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" `
    -WorkingDirectory (Join-Path $root "backend")

Start-Sleep -Seconds 2

Write-Host "啟動前端靜態站 (:5173) ..." -ForegroundColor Cyan
Start-Process -FilePath $py -ArgumentList "-m", "http.server", "5173" `
    -WorkingDirectory (Join-Path $root "frontend")

Start-Sleep -Seconds 1
Write-Host ""
Write-Host "✅ 已啟動" -ForegroundColor Green
Write-Host "   前端： http://127.0.0.1:5173" -ForegroundColor Yellow
Write-Host "   後端 health： http://127.0.0.1:8000/api/health" -ForegroundColor Yellow
Write-Host ""
Write-Host "注意：後端第一次啟動要載入模型（約 10-20 秒），health 顯示 loading 是正常的。" -ForegroundColor Gray
Start-Process "http://127.0.0.1:5173"
