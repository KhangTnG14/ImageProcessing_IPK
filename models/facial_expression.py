"""
models/facial_expression.py
----------------------------
Nhận diện cảm xúc khuôn mặt dùng OpenCV + ONNX model.
Không cần TensorFlow hay PyTorch — chạy hoàn toàn qua OpenCV DNN.

Pipeline:
  1. Dùng Haar Cascade detect vị trí khuôn mặt
  2. Crop từng khuôn mặt ra
  3. Đưa vào ONNX model → ra vector 8 cảm xúc
  4. Vẽ kết quả lên ảnh
"""

import os
import time
import cv2
import numpy as np

from core.base_model import BaseModel, Detection, PredictionResult

# Đường dẫn model — tính từ thư mục gốc project
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACE_MODEL = os.path.join(BASE_DIR, "weights", "haarcascade_frontalface_default.xml")
EMOT_MODEL = os.path.join(BASE_DIR, "weights", "emotion_model.onnx")

# 8 nhãn cảm xúc theo thứ tự output của model
EMOTION_LABELS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]

EMOTION_EMOJI = {
    "happiness": "😊", "sadness":  "😢", "anger":    "😠",
    "fear":      "😨", "surprise": "😲", "disgust":  "🤢",
    "neutral":   "😐", "contempt": "😤",
}

EMOTION_COLOR = {
    "happiness": (50,  205, 50),
    "sadness":   (200, 100, 50),
    "anger":     (50,   50, 220),
    "fear":      (180,  50, 180),
    "surprise":  (50,  200, 220),
    "disgust":   (50,  140,  80),
    "neutral":   (160, 160, 160),
    "contempt":  (100, 100, 200),
}


class FacialExpressionModel(BaseModel):

    def __init__(self):
        # Kiểm tra file model tồn tại
        for path, name in [(FACE_MODEL, "Face detector"), (EMOT_MODEL, "Emotion model")]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} không tìm thấy tại: {path}\n"
                    f"Chạy lệnh tải model trong hướng dẫn setup."
                )

        # Load face detector (Haar Cascade — cực nhanh)
        self._face_detector = cv2.CascadeClassifier(FACE_MODEL)

        # Load emotion model qua OpenCV DNN (đọc file ONNX)
        self._emotion_net = cv2.dnn.readNetFromONNX(EMOT_MODEL)

        print("[FacialExpression] ✅ Model loaded thành công.")

    def get_info(self) -> dict:
        return {
            "name": "Facial Expression Recognition",
            "model_id": "facial_expression",
            "version": "OpenCV DNN + ONNX",
            "description": (
                "Nhận diện 8 loại cảm xúc khuôn mặt: "
                "happy, sad, angry, fear, surprise, disgust, neutral, contempt. "
                "Hỗ trợ nhiều khuôn mặt trong cùng một ảnh. "
                "Chạy hoàn toàn offline, không cần TensorFlow."
            ),
            "emotions": EMOTION_LABELS,
            "num_classes": len(EMOTION_LABELS),
        }

    def predict(self, image: np.ndarray) -> PredictionResult:
        start_time = time.perf_counter()
        annotated  = image.copy()
        detections = []

        # Chuyển sang grayscale để detect khuôn mặt nhanh hơn
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bước 1: Detect tất cả khuôn mặt
        faces = self._face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),       # Bỏ qua mặt quá nhỏ
        )

        if len(faces) == 0:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return PredictionResult(
                model_name="Facial Expression Recognition",
                detections=[],
                object_count=0,
                processing_time_ms=elapsed_ms,
                annotated_image=annotated,
            )

        for (x, y, w, h) in faces:
            # Bước 2: Crop khuôn mặt, chuyển sang ảnh xám và resize về 64x64 (đúng định dạng của FER+ ONNX)
            face_crop = image[y:y+h, x:x+w]
            face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) # Chuyển sang ảnh xám 1 kênh
            
            blob = cv2.dnn.blobFromImage(
                face_gray,
                scalefactor=1.0,           # FER+ yêu cầu pixel từ 0.0 - 255.0, không chia cho 255
                size=(64, 64),             # Kích thước chuẩn 64x64
                mean=0.0,
                swapRB=False,
                crop=False,
            )


            # Bước 3: Chạy emotion model
            self._emotion_net.setInput(blob)
            outputs = self._emotion_net.forward()           # shape: (1, 8)
            scores  = outputs[0]

            # Softmax thủ công để ra xác suất
            exp_s  = np.exp(scores - np.max(scores))
            probs  = exp_s / exp_s.sum()

            top_idx    = int(np.argmax(probs))
            top_label  = EMOTION_LABELS[top_idx]
            top_conf   = float(probs[top_idx])
            emoji      = EMOTION_EMOJI.get(top_label, "")
            color      = EMOTION_COLOR.get(top_label, (160, 160, 160))

            detections.append(Detection(
                label=f"{emoji} {top_label}",
                confidence=round(top_conf, 4),
                bbox=[int(x), int(y), int(x+w), int(y+h)],
            ))

            # Bước 4: Vẽ kết quả
            cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)

            # Label chính phía trên
            main_text = f"{top_label} {top_conf:.0%}"
            (tw, th), baseline = cv2.getTextSize(
                main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            label_y = max(y - 8, th + 8)
            cv2.rectangle(
                annotated,
                (x, label_y - th - baseline - 4),
                (x + tw + 6, label_y),
                color, -1
            )
            cv2.putText(
                annotated, main_text,
                (x + 3, label_y - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )

            # Mini bar — top 3 cảm xúc phía dưới khuôn mặt
            top3_idx = np.argsort(probs)[::-1][:3]
            for i, idx in enumerate(top3_idx):
                emo      = EMOTION_LABELS[idx]
                pct      = float(probs[idx])
                emo_clr  = EMOTION_COLOR.get(emo, (160, 160, 160))
                row_y    = y + h + 6 + i * 16
                bar_fill = int(w * pct)

                cv2.rectangle(annotated, (x, row_y), (x+w, row_y+10), (40,40,40), -1)
                if bar_fill > 0:
                    cv2.rectangle(annotated, (x, row_y), (x+bar_fill, row_y+10), emo_clr, -1)
                cv2.putText(
                    annotated,
                    f"{emo[:3]} {pct:.0%}",
                    (x + w + 4, row_y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (210, 210, 210), 1, cv2.LINE_AA
                )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            model_name="Facial Expression Recognition",
            detections=detections,
            object_count=len(detections),
            processing_time_ms=elapsed_ms,
            annotated_image=annotated,
        )