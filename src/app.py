"""
src/app.py -- WindGuard AI API server with React UI dashboard.
Run with: uvicorn src.app:app --host 0.0.0.0 --port 8000
"""
import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .routes_detect import router as detect_router
from .routes_ask import router as ask_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("windguard")

app = FastAPI(
    title="WindGuard AI",
    description="Constrained object detection (RT-DETR, wind-turbine blade defects) + a minimal reasoning layer.",
    version="1.0.0",
)

app.include_router(detect_router)
app.include_router(ask_router)

# Paths for static files & docs
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
docs_dir = os.path.join(base_dir, "docs")
dist_dir = os.path.join(base_dir, "frontend", "dist")

if os.path.exists(docs_dir):
    app.mount("/docs_static", StaticFiles(directory=docs_dir), name="docs_static")

if os.path.exists(dist_dir):
    assets_dir = os.path.join(dist_dir, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/")
def read_root():
    index_file = os.path.join(dist_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "ok", "message": "WindGuard AI Backend API"}


@app.get("/docs/training_loss_plot.png")
def get_training_plot():
    plot_file = os.path.join(docs_dir, "training_loss_plot.png")
    if os.path.exists(plot_file):
        return FileResponse(plot_file)
    return JSONResponse(status_code=404, content={"detail": "Plot not found"})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
