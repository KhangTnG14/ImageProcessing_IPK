"""
pipeline.py
-----------
Pipeline chính của ImgPK.
Đây là điểm kết nối tất cả: nhận ảnh → chọn model → chạy → lưu kết quả.
API (Phase 2) và UI (Phase 3) đều sẽ gọi qua file này.
"""

import os
from typing import Optional, Tuple

from core.registry import registry
from core.base_model import PredictionResult
from utils.image_utils import load_image, save_annotated_image, save_json_result


class ImgPKPipeline:
    """
    Pipeline xử lý ảnh trung tâm của ImgPK.

    Sử dụng:
        pipeline = ImgPKPipeline()
        result, img_path, json_path = pipeline.run(
            image_path="photo.jpg",
            model_id="yolo_general",
        )
    """

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(
        self,
        image_path: str,
        model_id: str = "yolo_general",
        save_outputs: bool = True,
    ) -> Tuple[PredictionResult, Optional[str], Optional[str]]:
        """
        Chạy toàn bộ pipeline xử lý một ảnh.

        Args:
            image_path:   Đường dẫn đến file ảnh đầu vào
            model_id:     ID model muốn dùng (đã đăng ký trong Registry)
            save_outputs: Có lưu file kết quả ra disk không

        Returns:
            Tuple gồm:
              - PredictionResult  : object chứa toàn bộ kết quả
              - annotated_path    : đường dẫn ảnh đã annotate (hoặc None)
              - json_path         : đường dẫn file JSON kết quả (hoặc None)
        """
        print(f"\n{'='*50}")
        print(f"[Pipeline] 📸 Ảnh đầu vào : {image_path}")
        print(f"[Pipeline] 🤖 Model       : {model_id}")

        # Bước 1: Load ảnh
        image = load_image(image_path)
        h, w = image.shape[:2]
        print(f"[Pipeline] 📐 Kích thước  : {w}x{h} px")

        # Bước 2: Lấy model từ Registry và chạy prediction
        model = registry.get(model_id)
        result = model.predict(image)

        print(f"[Pipeline] ✅ Phát hiện   : {result.object_count} đối tượng")
        print(f"[Pipeline] ⏱️  Thời gian   : {result.processing_time_ms:.1f} ms")

        if result.detections:
            print("[Pipeline] 📋 Chi tiết:")
            for d in result.detections:
                print(f"           • {d.label:<20} conf={d.confidence:.1%}  bbox={d.bbox}")

        # Bước 3: Lưu kết quả (tuỳ chọn)
        annotated_path = None
        json_path = None

        if save_outputs and result.annotated_image is not None:
            annotated_path = save_annotated_image(
                result.annotated_image, image_path, self.output_dir
            )
            json_path = save_json_result(result, image_path, self.output_dir)
            print(f"[Pipeline] 💾 Ảnh output  : {annotated_path}")
            print(f"[Pipeline] 💾 JSON output : {json_path}")

        print(f"{'='*50}\n")
        return result, annotated_path, json_path
