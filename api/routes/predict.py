"""
api/routes/predict.py
---------------------
Endpoint chính: POST /predict
Nhận file ảnh upload + model_id, chạy pipeline, trả kết quả JSON + ảnh base64.
"""

import base64
import tempfile
import os

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
import io

from pipeline import ImgPKPipeline
from api.schemas import PredictResponse, DetectionItem

router = APIRouter()
pipeline = ImgPKPipeline(output_dir="outputs")


def numpy_to_base64(image: np.ndarray) -> str:
    """Chuyển numpy array (BGR) sang chuỗi base64 JPEG."""
    # Chuyển BGR → RGB cho Pillow
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=88)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("utf-8")


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Phân tích ảnh bằng model AI",
    description=(
        "Upload một file ảnh và chọn model muốn dùng. "
        "Trả về danh sách đối tượng phát hiện được, "
        "confidence score, bounding box, và ảnh đã annotate dạng base64."
    ),
)
async def predict(
    file: UploadFile = File(..., description="File ảnh (JPG, PNG, BMP, WEBP)"),
    model_id: str = Form(default="yolo_general", description="ID của model muốn dùng"),
    return_image: bool = Form(default=True, description="Có trả về ảnh annotated không"),
):
    # Kiểm tra định dạng file
    allowed_types = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Định dạng '{file.content_type}' không được hỗ trợ. "
                   f"Hỗ trợ: JPG, PNG, BMP, WEBP."
        )

    # Đọc ảnh từ upload vào bộ nhớ
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="File ảnh rỗng.")
    if len(contents) > 20 * 1024 * 1024:  # Giới hạn 20MB
        raise HTTPException(status_code=413, detail="File quá lớn. Tối đa 20MB.")

    # Decode ảnh từ bytes → numpy array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Không thể đọc file ảnh.")

    # Lưu tạm ra disk để pipeline xử lý (pipeline.run() cần image_path)
    suffix = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result, _, _ = pipeline.run(
            image_path=tmp_path,
            model_id=model_id,
            save_outputs=False,   # API không cần lưu file, dùng base64
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")
    finally:
        os.unlink(tmp_path)  # Xoá file tạm dù có lỗi hay không

    # Chuyển detections sang schema
    detections = [
        DetectionItem(
            label=d.label,
            confidence=d.confidence,
            bbox=d.bbox,
        )
        for d in result.detections
    ]

    # Encode ảnh annotated sang base64 (nếu được yêu cầu)
    annotated_b64 = None
    if return_image and result.annotated_image is not None:
        annotated_b64 = numpy_to_base64(result.annotated_image)

    return PredictResponse(
        success=True,
        model_name=result.model_name,
        object_count=result.object_count,
        processing_time_ms=result.processing_time_ms,
        detections=detections,
        annotated_image_base64=annotated_b64,
    )