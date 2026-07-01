"""
api/routes/models.py
--------------------
GET /models  → Danh sách model có trong hệ thống
GET /health  → Trạng thái server
"""

from fastapi import APIRouter
from core.registry import registry
from api.schemas import ModelsResponse, ModelInfo, HealthResponse

router = APIRouter()
APP_VERSION = "1.0.0-phase2"


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="Danh sách model AI có sẵn",
)
def list_models():
    """Trả về tất cả model đã đăng ký trong Registry kèm thông tin mô tả."""
    registry_data = registry.list_models()
    models_info = []

    for model_id, meta in registry_data.items():
        # Nếu model đã được load, lấy info chi tiết từ instance
        if meta["loaded"]:
            instance = registry.get(model_id)
            info = instance.get_info()
            models_info.append(ModelInfo(
                model_id=model_id,
                name=info.get("name", model_id),
                description=info.get("description", ""),
                num_classes=info.get("num_classes"),
                loaded=True,
            ))
        else:
            # Chưa load: chỉ trả thông tin cơ bản từ Registry
            models_info.append(ModelInfo(
                model_id=model_id,
                name=meta["class"],
                description="Chưa được load vào bộ nhớ.",
                loaded=False,
            ))

    return ModelsResponse(total=len(models_info), models=models_info)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái server",
)
def health_check():
    """Ping endpoint để kiểm tra server đang chạy và Registry hoạt động."""
    registry_data = registry.list_models()
    loaded_count = sum(1 for m in registry_data.values() if m["loaded"])

    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        registered_models=len(registry_data),
        loaded_models=loaded_count,
    )