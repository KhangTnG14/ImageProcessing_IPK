"""
api/main.py
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from setup import setup_models
from api.routes.predict import router as predict_router
from api.routes.models import router as models_router

app = FastAPI(
    title="ImgPK API",
    description="Nền tảng Computer Vision đa model.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, tags=["Prediction"])
app.include_router(models_router, tags=["System"])

# Serve thư mục static/
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def on_startup():
    print("\n[ImgPK] Server đang khởi động...")
    setup_models()
    print("[ImgPK] ✅ Sẵn sàng nhận request!\n")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse("static/index.html")