"""
utils/image_utils.py
--------------------
Các hàm tiện ích xử lý ảnh: đọc, lưu, validate.
Tách riêng ra đây để core và models không cần quan tâm đến đường dẫn file.
"""

import json
import os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from core.base_model import PredictionResult


SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def load_image(image_path: str) -> np.ndarray:
    """
    Đọc ảnh từ đường dẫn, trả về numpy array BGR (chuẩn OpenCV).

    Raises:
        FileNotFoundError: nếu file không tồn tại
        ValueError: nếu định dạng không được hỗ trợ hoặc không đọc được
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {image_path}")

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Định dạng '{path.suffix}' không được hỗ trợ. "
            f"Hỗ trợ: {SUPPORTED_FORMATS}"
        )

    # Dùng PIL để đọc (xử lý tốt hơn với ảnh có EXIF rotation)
    pil_img = Image.open(path).convert("RGB")
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


def save_annotated_image(
    annotated: np.ndarray,
    original_path: str,
    output_dir: str = "outputs",
) -> str:
    """
    Lưu ảnh đã annotate vào thư mục output.

    Returns:
        Đường dẫn file đã lưu
    """
    os.makedirs(output_dir, exist_ok=True)
    original_name = Path(original_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{original_name}_annotated_{timestamp}.jpg")
    cv2.imwrite(output_path, annotated)
    return output_path


def save_json_result(
    result: PredictionResult,
    original_path: str,
    output_dir: str = "outputs",
) -> str:
    """
    Lưu kết quả JSON ra file.

    Returns:
        Đường dẫn file JSON đã lưu
    """
    os.makedirs(output_dir, exist_ok=True)
    original_name = Path(original_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{original_name}_result_{timestamp}.json")

    data = result.to_dict()
    data["source_image"] = original_path

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path
