"""
api/schemas.py
--------------
Định nghĩa cấu trúc dữ liệu cho request và response của API.
Pydantic tự động validate kiểu dữ liệu và sinh tài liệu API (Swagger).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ── Response schemas ──────────────────────────────────────────────

class DetectionItem(BaseModel):
    """Một đối tượng được phát hiện trong ảnh."""
    label: str = Field(example="person")
    confidence: float = Field(example=0.92, ge=0.0, le=1.0)
    bbox: List[int] = Field(example=[120, 80, 340, 420],
                            description="[x1, y1, x2, y2] tính bằng pixel")


class PredictResponse(BaseModel):
    """Kết quả trả về từ endpoint POST /predict."""
    success: bool
    model_name: str
    object_count: int
    processing_time_ms: float
    detections: List[DetectionItem]
    annotated_image_base64: Optional[str] = Field(
        default=None,
        description="Ảnh đã vẽ bounding box, encode base64 (JPEG). "
                    "Dùng trực tiếp trong thẻ <img src='data:image/jpeg;base64,...'>"
    )
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Thông tin một model trong Registry."""
    model_id: str
    name: str
    description: str
    num_classes: Optional[int] = None
    loaded: bool = Field(description="Model đang được load trong bộ nhớ chưa")


class ModelsResponse(BaseModel):
    """Danh sách tất cả model có trong hệ thống."""
    total: int
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Trạng thái hệ thống."""
    status: str
    version: str
    registered_models: int
    loaded_models: int