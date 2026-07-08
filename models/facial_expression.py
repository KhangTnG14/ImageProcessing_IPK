"""
models/facial_expression.py
----------------------------
Model nhận diện cảm xúc khuôn mặt dùng DeepFace.
Phát hiện được: happy, sad, angry, fear, surprise, disgust, neutral.

DeepFace tự động:
  - Detect khuôn mặt trong ảnh (có thể nhiều khuôn mặt)
  - Phân tích cảm xúc cho từng khuôn mặt
  - Trả về confidence % cho từng loại cảm xúc

Không cần train — dùng model pretrained có sẵn trong DeepFace.
"""

import time
import cv2
import numpy as np

from core.base_model import BaseModel, Detection, PredictionResult

# Map cảm xúc sang emoji cho dễ đọc
EMOTION_EMOJI = {
    "happy":    "😊",
    "sad":      "😢",
    "angry":    "😠",
    "fear":     "😨",
    "surprise": "😲",
    "disgust":  "🤢",
    "neutral":  "😐",
}

# Màu BGR cho từng cảm xúc khi vẽ bounding box
EMOTION_COLOR = {
    "happy":    (50,  205, 50),   # Xanh lá
    "sad":      (205, 100, 50),   # Xanh dương đậm
    "angry":    (50,  50,  220),  # Đỏ
    "fear":     (180, 50,  180),  # Tím
    "surprise": (50,  200, 220),  # Vàng
    "disgust":  (50,  150, 100),  # Xanh rêu
    "neutral":  (160, 160, 160),  # Xám
}


class FacialExpressionModel(BaseModel):
    """
    Nhận diện cảm xúc khuôn mặt bằng DeepFace.
    Hỗ trợ nhiều khuôn mặt trong cùng một ảnh.
    """

    def __init__(self):
        # Import ở đây để tránh load TensorFlow khi không cần
        from deepface import DeepFace
        self._deepface = DeepFace
        # Warm-up: load model lần đầu (tải weights nếu chưa có)
        print("[FacialExpression] 🔄 Đang khởi tạo DeepFace model...")
        self._backend = "opencv"   # Backend detect khuôn mặt: nhanh và ổn định
        self._model_name = "Emotion"
        print("[FacialExpression] ✅ Sẵn sàng.")

    def get_info(self) -> dict:
        return {
            "name": "Facial Expression Recognition",
            "model_id": "facial_expression",
            "version": "DeepFace",
            "description": (
                "Nhận diện cảm xúc khuôn mặt: happy, sad, angry, "
                "fear, surprise, disgust, neutral. "
                "Hỗ trợ nhiều khuôn mặt trong cùng một ảnh."
            ),
            "emotions": list(EMOTION_EMOJI.keys()),
            "num_classes": len(EMOTION_EMOJI),
        }

    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Phân tích cảm xúc khuôn mặt trong ảnh.

        Args:
            image: numpy array BGR (chuẩn OpenCV)

        Returns:
            PredictionResult với mỗi Detection là một khuôn mặt
            label = tên cảm xúc dominant, confidence = % cảm xúc đó
        """
        start_time = time.perf_counter()

        annotated = image.copy()
        detections = []

        try:
            # DeepFace analyze: trả về list (một phần tử cho mỗi khuôn mặt)
            results = self._deepface.analyze(
                img_path=image,
                actions=["emotion"],        # Chỉ phân tích cảm xúc, bỏ age/gender/race
                detector_backend=self._backend,
                enforce_detection=False,    # Không raise lỗi nếu không tìm thấy mặt
                silent=True,
            )

            # Đảm bảo results luôn là list
            if isinstance(results, dict):
                results = [results]

            for face_data in results:
                # Lấy vùng khuôn mặt
                region = face_data.get("region", {})
                x = region.get("x", 0)
                y = region.get("y", 0)
                w = region.get("w", 0)
                h = region.get("h", 0)

                # Bỏ qua nếu không detect được khuôn mặt thật
                if w == 0 or h == 0:
                    continue

                bbox = [x, y, x + w, y + h]

                # Lấy cảm xúc dominant và score
                dominant_emotion = face_data.get("dominant_emotion", "neutral")
                emotion_scores   = face_data.get("emotion", {})
                confidence = emotion_scores.get(dominant_emotion, 0.0) / 100.0

                emoji  = EMOTION_EMOJI.get(dominant_emotion, "")
                label  = f"{emoji} {dominant_emotion}"
                color  = EMOTION_COLOR.get(dominant_emotion, (160, 160, 160))

                detections.append(Detection(
                    label=label,
                    confidence=round(confidence, 4),
                    bbox=bbox,
                ))

                # ── Vẽ lên ảnh ───────────────────────────────────
                # Bounding box khuôn mặt
                cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

                # Thanh cảm xúc nhỏ bên dưới bbox (top-3 cảm xúc)
                top3 = sorted(
                    emotion_scores.items(),
                    key=lambda kv: kv[1], reverse=True
                )[:3]

                # Label chính phía trên bbox
                main_text = f"{emoji}{dominant_emotion} {confidence:.0%}"
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

                # Mini bar chart top-3 cảm xúc phía dưới khuôn mặt
                bar_x = x
                bar_y = y + h + 6
                bar_max_w = w
                for i, (emo, score) in enumerate(top3):
                    pct     = score / 100.0
                    bar_w   = int(bar_max_w * pct)
                    emo_clr = EMOTION_COLOR.get(emo, (160, 160, 160))
                    row_y   = bar_y + i * 16

                    # Background bar
                    cv2.rectangle(
                        annotated,
                        (bar_x, row_y),
                        (bar_x + bar_max_w, row_y + 10),
                        (50, 50, 50), -1
                    )
                    # Filled bar
                    if bar_w > 0:
                        cv2.rectangle(
                            annotated,
                            (bar_x, row_y),
                            (bar_x + bar_w, row_y + 10),
                            emo_clr, -1
                        )
                    # Label
                    cv2.putText(
                        annotated,
                        f"{EMOTION_EMOJI.get(emo,'')}{emo[:3]} {score:.0f}%",
                        (bar_x + bar_max_w + 4, row_y + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (220, 220, 220), 1, cv2.LINE_AA
                    )

        except Exception as e:
            # Không crash app nếu DeepFace gặp lỗi với ảnh cụ thể
            print(f"[FacialExpression] ⚠️ Lỗi phân tích: {e}")

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            model_name="Facial Expression Recognition",
            detections=detections,
            object_count=len(detections),
            processing_time_ms=elapsed_ms,
            annotated_image=annotated,
        )