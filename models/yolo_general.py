"""
models/yolo_general.py
----------------------
Model đầu tiên của ImgPK: YOLOv8 nhận diện đối tượng tổng quát.
Sử dụng weight `yolov8n.pt` (nano — nhỏ nhất, nhanh nhất, đủ dùng để demo).
Model này nhận diện được 80 loại đối tượng phổ biến (COCO dataset):
người, xe hơi, xe đạp, điện thoại, ghế, ly, v.v.

Để thêm model mới về sau (ví dụ nhận diện biển số xe):
  1. Copy file này
  2. Đổi tên class và model_path
  3. Đăng ký vào setup.py
  → Xong. Không cần chạm vào code nào khác.
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO

from core.base_model import BaseModel, Detection, PredictionResult


# Màu sắc cho từng class (dùng HSV để tự động phân biệt màu)
def _get_color(class_id: int) -> tuple:
    """Sinh màu BGR dựa trên class_id để các nhãn khác nhau có màu khác nhau."""
    hue = int((class_id * 37) % 180)
    color_hsv = np.uint8([[[hue, 220, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])


class YoloGeneralModel(BaseModel):
    """
    YOLOv8 Nano — nhận diện 80 loại đối tượng tổng quát (COCO).
    Weight tự động download về ~/.cache/ultralytics/ lần đầu chạy.
    """

    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.40   # Chỉ lấy detection có confidence >= 40%

    def __init__(self):
        # Load model khi khởi tạo instance
        self._model = YOLO(self.MODEL_PATH)
        self._class_names = self._model.names   # dict: {0: "person", 1: "bicycle", ...}

    def get_info(self) -> dict:
        return {
            "name": "YOLOv8 General Detection",
            "model_id": "yolo_general",
            "version": "YOLOv8n",
            "description": "Nhận diện 80 loại đối tượng tổng quát (COCO dataset). "
                           "Phù hợp để test pipeline và demo.",
            "num_classes": len(self._class_names),
            "confidence_threshold": self.CONFIDENCE_THRESHOLD,
        }

    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Xử lý ảnh và trả về kết quả detection.

        Args:
            image: numpy array BGR (định dạng chuẩn của OpenCV)

        Returns:
            PredictionResult chứa detections + ảnh đã annotate
        """
        start_time = time.perf_counter()

        # Chạy inference
        results = self._model(
            image,
            conf=self.CONFIDENCE_THRESHOLD,
            verbose=False,   # Tắt log của YOLO để console sạch hơn
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Parse kết quả
        detections = []
        annotated = image.copy()

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                # Lấy thông tin từng detection
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self._class_names.get(class_id, f"class_{class_id}")

                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                ))

                # Vẽ bounding box lên ảnh
                color = _get_color(class_id)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Vẽ label nền + chữ
                text = f"{label} {conf:.0%}"
                (text_w, text_h), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_h - baseline - 4),
                    (x1 + text_w + 4, y1),
                    color, -1
                )
                cv2.putText(
                    annotated, text,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA
                )

        return PredictionResult(
            model_name="YOLOv8 General Detection",
            detections=detections,
            object_count=len(detections),
            processing_time_ms=elapsed_ms,
            annotated_image=annotated,
        )
