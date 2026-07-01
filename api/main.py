"""
api/main.py
-----------
Khởi động FastAPI app, gắn router, setup model Registry.
Chạy lệnh: uvicorn api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from setup import setup_models
from api.routes.predict import router as predict_router
from api.routes.models import router as models_router

# ── Khởi tạo app ──────────────────────────────────────────────────
app = FastAPI(
    title="ImgPK API",
    description=(
        "Nền tảng Computer Vision đa model.\n\n"
        "Upload ảnh → chọn model AI → nhận kết quả detection + ảnh annotated."
    ),
    version="1.0.0",
    docs_url="/docs",      # Swagger UI tự động sinh ra tại địa chỉ này
    redoc_url="/redoc",
)

# ── CORS: cho phép frontend (Phase 3) gọi API ─────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Phase 3 sẽ thu hẹp lại
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gắn các router ────────────────────────────────────────────────
app.include_router(predict_router, tags=["Prediction"])
app.include_router(models_router, tags=["System"])


# ── Sự kiện khởi động: đăng ký model khi server start ────────────
@app.on_event("startup")
async def on_startup():
    print("\n[ImgPK] Server đang khởi động...")
    setup_models()
    print("[ImgPK] ✅ Sẵn sàng nhận request!\n")


# ── Root endpoint ─────────────────────────────────────────────────
@app.get("/", tags=["System"], summary="Root")
def root():
    return {
        "app": "ImgPK",
        "status": "running",
        "docs": "/docs",
    }