"""
test_pipeline.py
----------------
Script kiểm tra nhanh toàn bộ Phase 1.
Chạy lệnh: python test_pipeline.py

Nếu không có ảnh thật, script sẽ tự tạo một ảnh test synthetic.
Test sẽ kiểm tra:
  ✅ Registry hoạt động đúng
  ✅ Model load thành công
  ✅ Pipeline xử lý ảnh và trả về kết quả đúng cấu trúc
  ✅ File output được lưu vào thư mục outputs/
"""

import os
import sys
import json
import numpy as np

# Đảm bảo Python tìm thấy các module trong project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup import setup_models
from core.registry import registry
from pipeline import ImgPKPipeline


def create_test_image(save_path: str = "tests/test_sample.jpg") -> str:
    """
    Tạo ảnh test nếu chưa có ảnh thật.
    Tải ảnh mẫu từ URL công khai hoặc tạo ảnh random.
    """
    os.makedirs("tests", exist_ok=True)

    if os.path.exists(save_path):
        print(f"[Test] 📁 Dùng ảnh có sẵn: {save_path}")
        return save_path

    # Thử tải ảnh mẫu từ internet
    try:
        import urllib.request
        # Ảnh sample từ Ultralytics (ảnh bus với nhiều người — test tốt cho YOLO)
        url = "https://ultralytics.com/images/bus.jpg"
        print(f"[Test] 🌐 Đang tải ảnh mẫu từ {url} ...")
        urllib.request.urlretrieve(url, save_path)
        print(f"[Test] ✅ Đã tải ảnh mẫu: {save_path}")
        return save_path
    except Exception as e:
        print(f"[Test] ⚠️  Không tải được ảnh online ({e}). Tạo ảnh synthetic...")

    # Fallback: tạo ảnh ngẫu nhiên (YOLO sẽ không detect gì nhiều, nhưng pipeline vẫn chạy)
    import cv2
    img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (50, 120, 200), -1)
    cv2.rectangle(img, (350, 150), (580, 380), (200, 80, 50), -1)
    cv2.imwrite(save_path, img)
    print(f"[Test] ✅ Đã tạo ảnh synthetic: {save_path}")
    return save_path


def run_tests():
    print("\n" + "="*60)
    print("  ImgPK — Phase 1 Test Suite")
    print("="*60)

    passed = 0
    failed = 0

    # ─── Test 1: Setup và Registry ───────────────────────────────
    print("\n[TEST 1] Khởi động Registry và đăng ký model...")
    try:
        setup_models()
        models = registry.list_models()
        assert "yolo_general" in models, "Model 'yolo_general' chưa được đăng ký!"
        print(f"         ✅ PASS — Registry có {len(models)} model: {list(models.keys())}")
        passed += 1
    except Exception as e:
        print(f"         ❌ FAIL — {e}")
        failed += 1

    # ─── Test 2: Model Info ───────────────────────────────────────
    print("\n[TEST 2] Lấy thông tin model...")
    try:
        model = registry.get("yolo_general")
        info = model.get_info()
        assert "name" in info
        assert "num_classes" in info
        print(f"         ✅ PASS — {info['name']} | {info['num_classes']} classes")
        passed += 1
    except Exception as e:
        print(f"         ❌ FAIL — {e}")
        failed += 1

    # ─── Test 3: Pipeline xử lý ảnh ──────────────────────────────
    print("\n[TEST 3] Chạy Pipeline xử lý ảnh...")
    try:
        test_image = create_test_image()
        pipeline = ImgPKPipeline(output_dir="outputs")
        result, img_out, json_out = pipeline.run(
            image_path=test_image,
            model_id="yolo_general",
            save_outputs=True,
        )

        assert result is not None
        assert result.object_count >= 0
        assert result.processing_time_ms > 0
        assert isinstance(result.detections, list)
        print(f"         ✅ PASS — Phát hiện {result.object_count} đối tượng "
              f"trong {result.processing_time_ms:.1f}ms")
        passed += 1
    except Exception as e:
        print(f"         ❌ FAIL — {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # ─── Test 4: Kiểm tra file output ────────────────────────────
    print("\n[TEST 4] Kiểm tra file output...")
    try:
        assert img_out and os.path.exists(img_out), f"Không tìm thấy ảnh output: {img_out}"
        assert json_out and os.path.exists(json_out), f"Không tìm thấy JSON output: {json_out}"

        with open(json_out, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "model_name" in data
        assert "detections" in data
        assert "object_count" in data
        assert "processing_time_ms" in data

        print(f"         ✅ PASS — Ảnh: {img_out}")
        print(f"                  JSON: {json_out}")
        passed += 1
    except Exception as e:
        print(f"         ❌ FAIL — {e}")
        failed += 1

    # ─── Test 5: Cấu trúc JSON đúng format ───────────────────────
    print("\n[TEST 5] Kiểm tra cấu trúc JSON output...")
    try:
        result_dict = result.to_dict()
        required_keys = {"model_name", "object_count", "processing_time_ms", "detections"}
        assert required_keys.issubset(result_dict.keys())

        if result_dict["detections"]:
            d = result_dict["detections"][0]
            assert "label" in d and "confidence" in d and "bbox" in d
            assert len(d["bbox"]) == 4

        print(f"         ✅ PASS — JSON đúng format. Preview:")
        print(f"         {json.dumps(result_dict, indent=10, ensure_ascii=False)[:400]}...")
        passed += 1
    except Exception as e:
        print(f"         ❌ FAIL — {e}")
        failed += 1

    # ─── Tổng kết ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  KẾT QUẢ: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("  🎉 Phase 1 hoàn chỉnh! Sẵn sàng cho Phase 2.")
    else:
        print(f"  ⚠️  {failed} test(s) thất bại. Kiểm tra lại.")
    print("="*60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
