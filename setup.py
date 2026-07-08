"""
setup.py
--------
Đây là file duy nhất bạn cần chỉnh sửa khi thêm model mới.
Toàn bộ quá trình thêm model mới:
  1. Tạo file model trong /models/
  2. Import class đó vào đây
  3. Gọi registry.register(...)
  → Xong. API và UI tự động nhận model mới.
"""

from core.registry import registry
from models.yolo_general import YoloGeneralModel
from models.facial_expression import FacialExpressionModel

def setup_models():
    """Đăng ký tất cả model vào Registry khi khởi động app."""

    registry.register("yolo_general", YoloGeneralModel)
    registry.register("facial_expression", FacialExpressionModel)


    # ── Thêm model mới vào đây trong tương lai ──
    # from models.license_plate import LicensePlateModel
    # registry.register("license_plate", LicensePlateModel)
    #
    # from models.leaf_disease import LeafDiseaseModel
    # registry.register("leaf_disease", LeafDiseaseModel)
    # ─────────────────────────────────────────────

    print(f"[Setup] Đã đăng ký {len(registry.list_models())} model(s).")
