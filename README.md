# ImgPK — Phase 1: Core Engine

Nền tảng Computer Vision có kiến trúc mở rộng.

## Cài đặt

```bash
cd ImgPK
pip install -r requirements.txt
```

## Chạy test

```bash
python test_pipeline.py
```

## Chạy thủ công với ảnh của bạn

```python
from setup import setup_models
from pipeline import ImgPKPipeline

setup_models()

pipeline = ImgPKPipeline()
result, img_path, json_path = pipeline.run(
    image_path="đường/dẫn/đến/ảnh.jpg",
    model_id="yolo_general",
)

print(f"Phát hiện: {result.object_count} đối tượng")
print(f"Ảnh output: {img_path}")
```

## Thêm model mới (sau này)

1. Tạo file `models/ten_model_moi.py` — kế thừa `BaseModel`
2. Mở `setup.py` — thêm 2 dòng import + register
3. Xong. Không cần đụng vào code nào khác.

## Cấu trúc thư mục

```
ImgPK/
├── core/
│   ├── base_model.py     ← Hợp đồng cho mọi model (ABC)
│   └── registry.py       ← Danh bạ model
├── models/
│   └── yolo_general.py   ← Model đầu tiên: YOLOv8 General
├── utils/
│   └── image_utils.py    ← Đọc/lưu ảnh và JSON
├── outputs/              ← Kết quả được lưu ở đây
├── tests/                ← Ảnh dùng để test
├── pipeline.py           ← Pipeline trung tâm
├── setup.py              ← Đăng ký model
├── test_pipeline.py      ← Script kiểm tra
└── requirements.txt
```
