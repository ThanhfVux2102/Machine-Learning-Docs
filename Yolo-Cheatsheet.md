# YOLOv8 Training Command Explanation

## 🔹 Câu lệnh mẫu
```bash
!yolo task=detect mode=train model=yolov8s.pt data={dataset.location}/data.yaml epochs=25 imgsz=800
```

---

## 1. `!`
- Trong Colab, dấu `!` có nghĩa là chạy lệnh **shell command** trực tiếp từ notebook.

---

## 2. `yolo`
- Command line tool của **Ultralytics YOLOv8** (có sau khi `pip install ultralytics`).

---

## 3. `task=detect`
- Xác định **nhiệm vụ (task)**:
  - `detect` → Object Detection (bạn đang dùng).
  - `segment` → Instance Segmentation.
  - `classify` → Image Classification.
  - `pose` → Pose Estimation.

Ở đây: YOLO được dùng cho **Object Detection**.

---

## 4. `mode=train`
- Chọn **chế độ (mode)**:  
  - `train` → Huấn luyện (training).  
  - `val` → Đánh giá (validation).  
  - `predict` → Suy luận (inference).  
  - `export` → Xuất model để deploy.

Ở đây: YOLO đang ở chế độ **train**.

---

## 5. `model=yolov8s.pt`
- Chỉ định **model YOLOv8** để train.  
- `yolov8s.pt` = YOLOv8 **Small** (nhẹ, tốc độ nhanh, chính xác vừa).  
- Các biến thể khác:  
  - `yolov8n.pt` → Nano (nhỏ nhất, nhanh nhất).  
  - `yolov8m.pt` → Medium.  
  - `yolov8l.pt` → Large.  
  - `yolov8x.pt` → X-Large (chính xác nhất, chậm nhất).

---

## 6. `data={dataset.location}/data.yaml`
- **Dataset config file (YAML)**, mô tả dữ liệu train/val:  
  ```yaml
  train: images/train
  val: images/val
  nc: 2
  names: ['cat', 'dog']
  ```
- `{dataset.location}` là biến Python trỏ đến thư mục dataset (ví dụ `/content/datasets/mydata`).

---

## 7. `epochs=25`
- Số vòng lặp train (epoch).  
- 1 epoch = duyệt qua toàn bộ dataset 1 lần.  
- Ở đây: YOLO train **25 lần qua dataset**.

---

## 8. `imgsz=800`
- Kích thước ảnh được resize trước khi đưa vào model.  
- Giá trị phổ biến:  
  - `320` → nhanh, ít chính xác.  
  - `640` → mặc định, cân bằng.  
  - `800–1280` → chính xác hơn (đặc biệt với vật thể nhỏ), nhưng chậm hơn và tốn RAM GPU.

Ở đây: YOLO resize ảnh thành **800x800**.

---

## 📌 Tóm lại
Câu lệnh trên nghĩa là:  
> "Huấn luyện một model YOLOv8 Small (`yolov8s.pt`) cho bài toán **object detection**, dùng dataset trong `{dataset.location}/data.yaml`, train trong **25 epochs**, resize ảnh thành **800x800** trước khi đưa vào model."
