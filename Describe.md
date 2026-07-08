# Describe.md

## Phần 1 - Cây thư mục đầy đủ và chi tiết

```text
ImageProcessing_IPK/
├── .git/                                 # Thư mục Git của project
├── api/                                  # API layer bằng FastAPI
│   ├── __init__.py                       # Đánh dấu thư mục api là package Python
│   ├── main.py                           # Entry point của server FastAPI
│   ├── schemas.py                        # Pydantic schema cho request/response API
│   └── routes/                           # Các endpoint API phân theo chức năng
│       ├── __init__.py                   # Đánh dấu routes là package Python
│       ├── models.py                     # Endpoint: /models, /health
│       └── predict.py                    # Endpoint: /predict
├── core/                                 # Thành phần cốt lõi của hệ thống
│   ├── __init__.py                       # Đánh dấu thư mục core là package Python
│   ├── base_model.py                     # Định nghĩa abstract class BaseModel và cấu trúc PredictionResult
│   └── registry.py                       # Hệ thống đăng ký model (ModelRegistry)
├── models/                               # Chứa các implementation model AI
│   ├── __init__.py                       # Đánh dấu thư mục models là package Python
│   └── yolo_general.py                   # Model YOLOv8 nhận diện đối tượng tổng quát
├── outputs/                              # Thư mục lưu kết quả xử lý ảnh
│   └── test_sample_result_20260702_033018.json
├── static/                               # Tài nguyên frontend tĩnh
│   └── index.html                        # Giao diện web đơn giản phục vụ API
├── tests/                                # Thư mục chứa dữ liệu/ảnh dùng cho test
│   └── test_sample.jpg                   # Ảnh mẫu cho kiểm thử
├── utils/                                # Các hàm tiện ích dùng chung
│   ├── __init__.py                       # Đánh dấu thư mục utils là package Python
│   └── image_utils.py                    # Đọc ảnh, lưu ảnh annotated, lưu JSON kết quả
├── pipeline.py                           # Pipeline trung tâm: load ảnh -> chọn model -> chạy inference -> lưu output
├── setup.py                              # File đăng ký model để hệ thống nhận diện model mới
├── test_pipeline.py                      # Script test toàn bộ pipeline và thư mục outputs
├── requirements.txt                      # Danh sách các thư viện Python cần cài đặt
├── README.md                             # Tài liệu hướng dẫn sử dụng project
├── yolov8n.pt                            # File trọng số YOLOv8 nano dùng cho inference
└── __pycache__/                          # Cache biên dịch Python (tự tạo)
```

### Mô tả từng thư mục và file chính

- api/
  - Chứa toàn bộ hệ thống API web cho project.
  - main.py: khởi tạo ứng dụng FastAPI, đăng ký router, phục vụ static files và chạy startup setup.
  - schemas.py: định nghĩa cấu trúc dữ liệu cho API bằng Pydantic.
  - routes/predict.py: endpoint nhận ảnh upload, chạy inference và trả về kết quả JSON + ảnh annotate dạng base64.
  - routes/models.py: cung cấp endpoint thông tin model và trạng thái hệ thống.

- core/
  - Chứa logic nền tảng cho việc xây dựng và quản lý model.
  - base_model.py: định nghĩa interface chuẩn cho mọi model AI trong hệ thống.
  - registry.py: hệ thống đăng ký model theo kiểu singleton, giúp dễ dàng thêm model mới mà không sửa nhiều file.

- models/
  - Chứa các implementation model thực tế.
  - yolo_general.py: model YOLOv8 dùng để phát hiện các đối tượng tổng quát trên ảnh.

- utils/
  - Chứa các hàm hỗ trợ xử lý ảnh và file.
  - image_utils.py: đọc ảnh từ disk, lưu ảnh đã annotate, lưu kết quả JSON.

- outputs/
  - Nơi lưu trữ các kết quả đầu ra sau khi chạy pipeline, ví dụ ảnh đã đánh dấu và file JSON.

- static/
  - Chứa giao diện web tĩnh để phục vụ truy cập nhanh từ trình duyệt.

- tests/
  - Chứa dữ liệu mẫu phục vụ kiểm thử.

- pipeline.py
  - Là lớp orchestration trung tâm của project.
  - Đóng vai trò kết nối các thành phần: nhận ảnh, lấy model từ registry, chạy prediction, lưu output.

- setup.py
  - File cấu hình ban đầu để đăng ký các model có sẵn vào hệ thống.

- test_pipeline.py
  - Script kiểm tra pipeline từ đầu đến cuối, bao gồm load model, chạy inference và kiểm tra file output.

- requirements.txt
  - Liệt kê thư viện cần thiết như FastAPI, Ultralytics, OpenCV, numpy, Pillow.

- README.md
  - Tài liệu hướng dẫn cài đặt, chạy thử và cách mở rộng model mới.

- yolov8n.pt
  - File trọng số pretrained của YOLOv8 nano, dùng để thực hiện nhận diện đối tượng.

---

## Phần 2 - Mô tả project bằng text

Project này là một nền tảng xử lý ảnh bằng trí tuệ nhân tạo, tập trung vào bài toán object detection (nhận diện đối tượng trong ảnh) bằng mô hình YOLOv8. Nó được thiết kế theo hướng module hóa, dễ mở rộng, có thể chạy như một pipeline độc lập, hoặc thông qua API web bằng FastAPI.

Điểm nổi bật của project là kiến trúc phân tầng rõ ràng:

- lớp model: encapsulate logic inference của từng mô hình AI;
- lớp core: định nghĩa interface chuẩn và registry để quản lý model;
- lớp pipeline: điều phối toàn bộ quy trình xử lý ảnh;
- lớp API: cho phép người dùng upload ảnh và nhận kết quả qua HTTP.

Khi chạy, hệ thống sẽ đọc ảnh đầu vào, chọn mô hình phù hợp từ registry, chạy inference để tìm các đối tượng, vẽ bounding box lên ảnh, tính độ tin cậy và lưu lại kết quả dưới dạng ảnh annotated và file JSON. Ngoài ra, project còn cung cấp giao diện web tĩnh và tài liệu API Swagger/Redoc để người dùng dễ dàng thử nghiệm.

Về mặt mục tiêu, đây là một project phù hợp để làm nền tảng demo, nghiên cứu hoặc phát triển thêm các mô hình chuyên sâu hơn như nhận diện biển số xe, phát hiện vật thể riêng trong ngành công nghiệp, y tế hoặc nông nghiệp.
