
# Hướng Dẫn Xây Dựng Fully Connected Network (FCN)

## 1. **FCN là gì?**
Fully Connected Network (FCN) là một loại neural network, trong đó mỗi neuron trong một lớp sẽ kết nối với tất cả các neuron trong lớp tiếp theo. FCN thường được sử dụng cho các bài toán phân loại (classification) và hồi quy (regression).

**Các bước xây dựng FCN**:
- **Input Layer**: Lớp đầu vào, nơi dữ liệu đầu vào được đưa vào mô hình.
- **Hidden Layers**: Các lớp ẩn, nơi mô hình học các đặc trưng từ dữ liệu.
- **Output Layer**: Lớp đầu ra, nơi kết quả dự đoán của mô hình được tạo ra.

### **Cách xây dựng mô hình FCN:**
1. **Chuẩn bị Dữ liệu**: Đảm bảo dữ liệu đã được tiền xử lý và chuyển đổi thành dạng có thể đưa vào mô hình.
2. **Xây dựng Mô hình**:
   - **Input Layer**: Chỉ định số lượng đặc trưng (features) trong dữ liệu đầu vào.
   - **Hidden Layers**: Thêm các lớp ẩn với hàm kích hoạt **ReLU** hoặc **tanh**.
   - **Output Layer**: Tùy thuộc vào bài toán, sử dụng hàm kích hoạt như **sigmoid** (cho binary classification), **softmax** (cho multiclass classification), hoặc không có activation (cho regression).
3. **Tùy chỉnh Loss Function và Optimizer**: Chọn loss function và optimizer phù hợp với bài toán.

### **Sample Code cho FCN**:

```python
# Build the model
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),  # Input layer with ReLU activation
    Dense(32, activation='relu'),  # Hidden layer with ReLU activation
    Dense(5, activation='softmax')  # Output layer for multiclass classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',  # Suitable for multiclass classification
              metrics=['accuracy'])
```

**Giải thích Code**:
- **Dense Layer**: Mỗi lớp **Dense** có các neuron được kết nối với tất cả các neuron trong lớp tiếp theo.
- **Activation Functions**:
  - **ReLU** (Rectified Linear Unit): Dùng cho các lớp ẩn để giúp mô hình học được các đặc trưng không tuyến tính.
  - **Softmax**: Dùng cho lớp đầu ra trong bài toán phân loại nhiều lớp (multiclass classification).
- **Loss Function**: `sparse_categorical_crossentropy` dùng cho phân loại nhiều lớp, phù hợp với output integer labels.
- **Optimizer**: `Adam`, một optimizer hiệu quả và phổ biến trong deep learning.

---

# Sự khác biệt giữa Output Layer, Loss Function và y Labels trong FCN

Trong mô hình Fully Connected Network (FCN), các yếu tố quan trọng để cấu hình mô hình cho các bài toán khác nhau là:

1. **Output layer (và activation)** 
2. **Loss function**
3. **y labels trong dữ liệu**

Dưới đây là sự giải thích về cách tùy chỉnh các yếu tố này trong các bài toán **Binary Classification**, **Multiclass Classification**, **Multi-label Classification**, và **Regression**.

---

## 1. Output Layer (và Activation)

### **a. Binary Classification**

- **Mục đích**: Phân loại dữ liệu thành 2 lớp (ví dụ: spam hoặc không spam).
- **Output Layer**: 1 neuron, với **sigmoid activation**.
  - **Lý do**: Vì có 2 lớp, ta chỉ cần 1 neuron để dự đoán xác suất của lớp 1 (lớp còn lại sẽ là lớp đối diện, ví dụ: lớp 0).
  - **Activation**: `sigmoid` sẽ giúp chuyển đổi giá trị đầu ra về phạm vi [0, 1], đại diện cho xác suất của lớp 1.

```python
Dense(1, activation='sigmoid')
```

### **b. Multiclass Classification**

- **Mục đích**: Phân loại dữ liệu thành nhiều lớp (ví dụ: phân loại ảnh thành các loại động vật khác nhau).
- **Output Layer**: `num_classes` neurons, với **softmax activation**.
  - **Lý do**: Vì có nhiều lớp, `softmax` giúp tính toán xác suất cho từng lớp. Xác suất của tất cả các lớp cộng lại bằng 1, do đó mỗi lớp có thể được chọn như là lớp có xác suất cao nhất.

```python
Dense(num_classes, activation='softmax')
```

### **c. Multi-label Classification**

- **Mục đích**: Một mẫu có thể thuộc nhiều lớp (ví dụ: một bức ảnh có thể có cả "chó", "mèo" và "chim").
- **Output Layer**: `num_labels` neurons, với **sigmoid activation** cho mỗi neuron.
  - **Lý do**: Mỗi lớp được tính độc lập với nhau, vì vậy mỗi neuron sử dụng activation `sigmoid`, cho phép output từ 0 hoặc 1, biểu thị việc có hay không có một label cụ thể.

```python
Dense(num_labels, activation='sigmoid')
```

### **d. Regression**

- **Mục đích**: Dự đoán giá trị liên tục (ví dụ: dự đoán giá nhà, nhiệt độ).
- **Output Layer**: 1 neuron, **activation=None** hoặc không dùng activation.
  - **Lý do**: Ta không cần hạn chế giá trị đầu ra, vì giá trị liên tục có thể nằm trong bất kỳ khoảng giá trị nào, không cần phải chuyển đổi bằng sigmoid hay softmax.

```python
Dense(1)  # hoặc Dense(1, activation=None)
```

---

## 2. Loss Function

### **a. Binary Classification**

- **Loss Function**: `binary_crossentropy`
  - **Lý do**: Đây là bài toán phân loại nhị phân, nơi ta sử dụng cross-entropy để tính toán sự khác biệt giữa giá trị thực tế và dự đoán (0 hoặc 1).

```python
loss='binary_crossentropy'
```

### **b. Multiclass Classification**

- **Loss Function**: `sparse_categorical_crossentropy` (nếu y là dạng integer) hoặc `categorical_crossentropy` (nếu y là dạng one-hot encoding).
  - **Lý do**: Sử dụng cross-entropy cho việc phân loại nhiều lớp. Khi có nhiều lớp, ta tính toán xác suất cho mỗi lớp và cross-entropy giúp đánh giá sự sai lệch giữa phân phối xác suất thực tế và dự đoán.

```python
loss='sparse_categorical_crossentropy'  # với y là integer
# hoặc
loss='categorical_crossentropy'  # với y là one-hot
```

### **c. Multi-label Classification**

- **Loss Function**: `binary_crossentropy` cho từng label.
  - **Lý do**: Mỗi label được tính toán độc lập, do đó ta dùng `binary_crossentropy` để đánh giá từng label riêng biệt (chỉ có 2 giá trị: có hoặc không).

```python
loss='binary_crossentropy'
```

### **d. Regression**

- **Loss Function**: `mean_squared_error (MSE)` hoặc `mean_absolute_error (MAE)`.
  - **Lý do**: Trong bài toán hồi quy, ta cần tính toán sự khác biệt giữa giá trị dự đoán và giá trị thực tế dưới dạng số liên tục, do đó MSE hoặc MAE là phù hợp.

```python
loss='mean_squared_error'  # hoặc 'mean_absolute_error'
```

---

## 3. y Labels trong Dữ Liệu

### **a. Binary Classification**

- **y Labels**: 1 cột với các giá trị là 0 hoặc 1.
  - **Lý do**: Chỉ có 2 lớp, mỗi mẫu dữ liệu sẽ được gán nhãn là 0 hoặc 1.

### **b. Multiclass Classification**

- **y Labels**: Một cột với các giá trị là integer từ 0 đến `num_classes-1`, hoặc sử dụng dạng one-hot encoding với mỗi lớp là 1 cột.
  - **Lý do**: Mỗi mẫu dữ liệu sẽ được phân vào một lớp duy nhất trong các lớp có sẵn.

### **c. Multi-label Classification**

- **y Labels**: Một ma trận với mỗi label được đánh dấu là 0 hoặc 1 (one-hot cho mỗi label).
  - **Lý do**: Một mẫu có thể thuộc nhiều lớp cùng lúc, vì vậy mỗi label trong ma trận có thể là 0 hoặc 1, tùy thuộc vào sự hiện diện của label đó.

### **d. Regression**

- **y Labels**: Một giá trị liên tục.
  - **Lý do**: Dữ liệu đầu ra trong bài toán hồi quy là các giá trị số thực, chẳng hạn như giá trị dự đoán của một ngôi nhà hoặc nhiệt độ.

---

## Tóm tắt

| **Yếu tố**              | **Binary Classification**                      | **Multiclass Classification**               | **Multi-label Classification**              | **Regression**                            |
|-------------------------|-----------------------------------------------|---------------------------------------------|---------------------------------------------|-------------------------------------------|
| **Output Layer**        | `Dense(1, activation='sigmoid')`              | `Dense(num_classes, activation='softmax')`  | `Dense(num_labels, activation='sigmoid')`   | `Dense(1)`                                |
| **Loss Function**       | `binary_crossentropy`                         | `sparse_categorical_crossentropy`           | `binary_crossentropy`                      | `mean_squared_error`                      |
| **y Labels**            | 0 hoặc 1                                      | Integer từ 0 đến `num_classes-1` hoặc one-hot | 0 hoặc 1 cho mỗi label                     | Giá trị liên tục                          |
