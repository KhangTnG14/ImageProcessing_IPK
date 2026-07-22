# download_emotion_model.py
# Chạy: python download_emotion_model.py

import urllib.request
import os

os.makedirs("weights", exist_ok=True)

# Model emotion từ kho chính thức của onnx model zoo
# Đây là FER+ model — nhận diện 8 cảm xúc, file ~33MB
url = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
dst = "weights/emotion_model.onnx"

print("Đang tải emotion model (~33MB) từ ONNX Model Zoo...")
print("Vui lòng chờ...\n")

try:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=120) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dst, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    bars = int(pct / 2)
                    print(f"  [{'█' * bars}{'░' * (50 - bars)}] {pct:.0f}%", end="\r")

    print(f"\n\n Tải xong: {dst}")
    print(f"   Kích thước: {os.path.getsize(dst) / 1024 / 1024:.1f} MB")

except Exception as e:
    print(f"\n Lỗi: {e}")
    print("\nHãy tải thủ công theo hướng dẫn bên dưới.")