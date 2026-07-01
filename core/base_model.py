"""
core/base_model.py
------------------
Đây là "hợp đồng" (Abstract Base Class) mà mọi model AI trong ImgPK đều phải tuân theo.
Nguyên tắc: bất kỳ model mới nào cũng chỉ cần kế thừa class này và implement 2 hàm:
  - get_info()  → trả về thông tin mô tả model
  - predict()   → nhận ảnh vào, trả về kết quả chuẩn
Phần còn lại của hệ thống (API, Registry, UI) không cần biết bên trong model làm gì.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    """Một đối tượng được phát hiện trong ảnh."""
    label: str              # Tên nhãn, ví dụ: "person", "car"
    confidence: float       # Độ tin cậy, từ 0.0 đến 1.0
    bbox: List[int]         # Bounding box: [x1, y1, x2, y2] tính bằng pixel


@dataclass
class PredictionResult:
    """Kết quả đầy đủ trả về sau khi model xử lý một ảnh."""
    model_name: str                     # Tên model đã chạy
    detections: List[Detection]         # Danh sách các đối tượng phát hiện được
    object_count: int                   # Tổng số đối tượng
    processing_time_ms: float           # Thời gian xử lý (mili giây)
    annotated_image: Optional[np.ndarray] = field(default=None, repr=False)
    # ảnh đã được vẽ bounding box, dạng numpy array (BGR)

    def to_dict(self) -> dict:
        """Chuyển kết quả sang JSON-serializable dict để lưu hoặc trả về API."""
        return {
            "model_name": self.model_name,
            "object_count": self.object_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "detections": [
                {
                    "label": d.label,
                    "confidence": round(d.confidence, 4),
                    "bbox": d.bbox,
                }
                for d in self.detections
            ],
        }


class BaseModel(ABC):
    """
    Lớp trừu tượng (Abstract Base Class) cho tất cả model trong ImgPK.

    Cách thêm model mới:
    1. Tạo file mới trong thư mục /models/
    2. Kế thừa class BaseModel
    3. Implement hai hàm get_info() và predict()
    4. Đăng ký vào ModelRegistry
    → Không cần chỉnh sửa bất kỳ code nào khác.
    """

    @abstractmethod
    def get_info(self) -> dict:
        """
        Trả về thông tin mô tả của model.
        Ví dụ: {"name": "YOLOv8 General", "version": "8n", "description": "..."}
        """
        ...

    @abstractmethod
    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Nhận ảnh (numpy array BGR từ OpenCV), trả về PredictionResult.
        """
        ...
