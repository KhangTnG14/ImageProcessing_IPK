"""
core/registry.py
----------------
ModelRegistry là "danh bạ" trung tâm của ImgPK.
Tất cả model được đăng ký vào đây, và phần còn lại của hệ thống
chỉ cần hỏi Registry để lấy model — không cần biết model nằm ở đâu.

Cách hoạt động:
  - registry.register("yolo_general", YoloGeneralModel)   ← đăng ký
  - registry.get("yolo_general")                           ← lấy ra dùng
  - registry.list_models()                                 ← xem có gì

Nguyên tắc Open/Closed:
  Thêm model mới → chỉ cần gọi register() một lần.
  Không bao giờ phải sửa file này.
"""

from typing import Dict, Type
from core.base_model import BaseModel


class ModelRegistry:
    """Hệ thống đăng ký và quản lý tất cả model AI trong ImgPK."""

    def __init__(self):
        # Dictionary lưu: model_id (str) → class của model (chưa khởi tạo)
        # Dùng class thay vì instance để tránh load model khi chưa cần
        self._registry: Dict[str, Type[BaseModel]] = {}
        self._instances: Dict[str, BaseModel] = {}   # cache instance đã load

    def register(self, model_id: str, model_class: Type[BaseModel]) -> None:
        """
        Đăng ký một model vào hệ thống.

        Args:
            model_id: Tên định danh duy nhất, ví dụ: "yolo_general"
            model_class: Class của model (kế thừa BaseModel)
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(f"Model '{model_id}' phải kế thừa từ BaseModel.")
        self._registry[model_id] = model_class
        print(f"[Registry] ✅ Đã đăng ký model: '{model_id}'")

    def get(self, model_id: str) -> BaseModel:
        """
        Lấy instance của model theo ID. Tự động khởi tạo nếu chưa load.

        Args:
            model_id: Tên model đã đăng ký

        Returns:
            Instance của model, sẵn sàng để gọi predict()
        """
        if model_id not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Model '{model_id}' không tồn tại. "
                f"Các model hiện có: {available}"
            )

        # Lazy loading: chỉ khởi tạo model khi lần đầu được gọi
        if model_id not in self._instances:
            print(f"[Registry] 🔄 Đang load model '{model_id}'...")
            self._instances[model_id] = self._registry[model_id]()
            print(f"[Registry] ✅ Model '{model_id}' đã sẵn sàng.")

        return self._instances[model_id]

    def list_models(self) -> Dict[str, dict]:
        """
        Trả về danh sách tất cả model đã đăng ký kèm thông tin mô tả.
        """
        result = {}
        for model_id, model_class in self._registry.items():
            loaded = model_id in self._instances
            result[model_id] = {
                "loaded": loaded,
                "class": model_class.__name__,
            }
        return result

    def unload(self, model_id: str) -> None:
        """Giải phóng bộ nhớ của một model đang chạy."""
        if model_id in self._instances:
            del self._instances[model_id]
            print(f"[Registry] 🗑️  Đã unload model '{model_id}'.")


# Singleton: toàn bộ app dùng chung một Registry duy nhất
registry = ModelRegistry()
