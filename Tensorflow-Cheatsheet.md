# 🎯 TensorFlow Dataset Pipeline – Tham số & Cách chọn (Cheatsheet)

## 1. `.map(function, num_parallel_calls=...)`

| Tham số              | Khi nào cần chỉnh                              | Cách chọn                                                                 |
|----------------------|------------------------------------------------|---------------------------------------------------------------------------|
| `function`           | **Bắt buộc**                                   | Hàm xử lý từng phần tử (normalize, augment, tokenize, scale...)          |
| `num_parallel_calls` | Khi dataset lớn hoặc xử lý nặng trong `map()`  | Dùng `tf.data.AUTOTUNE` để tối ưu hiệu suất                              |

> 💡 Tip: Luôn dùng `num_parallel_calls=tf.data.AUTOTUNE` khi xử lý ảnh, text hoặc augment nhiều.

---

## 2. `.shuffle(buffer_size, seed=None, reshuffle_each_iteration=True)`

| Tham số                    | Khi nào cần chỉnh                                           | Cách chọn                                                                 |
|----------------------------|-------------------------------------------------------------|---------------------------------------------------------------------------|
| `buffer_size`              | Rất quan trọng để tránh overfitting                        | - Bằng số mẫu (VD: `ds_info.splits['train'].num_examples`)                |
| `seed`                     | Muốn kết quả reproducible                                  | Chọn số cụ thể như `seed=42`                                              |
| `reshuffle_each_iteration` | Nếu muốn dữ liệu shuffle lại mỗi epoch                     | Mặc định là `True`, chỉ tắt khi debug                                     |

> ⚠️ Shuffle quá yếu → model học theo thứ tự và overfit.

---

## 3. `.batch(batch_size, drop_remainder=False)`

| Tham số          | Khi nào cần chỉnh                                     | Cách chọn                                               |
|------------------|-------------------------------------------------------|---------------------------------------------------------|
| `batch_size`     | Luôn cần chỉ định                                     | - GPU yếu: 16–32 <br> - GPU mạnh: 64–128+               |
| `drop_remainder` | Muốn các batch có shape đồng nhất                     | `True` nếu dùng RNN/LSTM hoặc TPU yêu cầu fixed shape   |

> ✅ Nếu dùng mô hình yêu cầu input cố định → nên dùng `drop_remainder=True`.

---

## 4. `.prefetch(buffer_size)`

| Tham số          | Khi nào cần chỉnh                   | Cách chọn                          |
|------------------|-------------------------------------|------------------------------------|
| `buffer_size`    | Muốn tăng tốc độ training           | Dùng `tf.data.AUTOTUNE`            |

> 💡 Nên luôn có `.prefetch(tf.data.AUTOTUNE)` ở cuối pipeline.

---

## 5. `.repeat(count=None)`

| Tham số     | Khi nào cần chỉnh                         | Cách chọn                           |
|-------------|-------------------------------------------|-------------------------------------|
| `count`     | Khi bạn muốn dataset tự lặp lại nhiều lần | Nếu dùng trong `.fit(epochs=...)`, để mặc định là vô hạn (`None`) |

---

## 🧠 Nguyên tắc tư duy chọn tham số:

| Câu hỏi                        | Gợi ý hành động                                       |
|-------------------------------|--------------------------------------------------------|
| Dataset lớn hay nhỏ?          | Dataset lớn → dùng `shuffle(buffer_size lớn)`         |
| GPU có chậm không?            | Dùng `prefetch(tf.data.AUTOTUNE)`                     |
| Có cần xử lý song song?       | Dùng `map(..., num_parallel_calls=tf.data.AUTOTUNE)`  |
| Mô hình cần shape cố định?    | Dùng `drop_remainder=True` trong `.batch()`           |

---

## 🔁 Cấu trúc pipeline tiêu chuẩn:

```python
ds_train = ds_train.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.shuffle(buffer_size=10000, seed=42)
ds_train = ds_train.batch(32, drop_remainder=True)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
```

---

# 📒 Keras Sequential Handbook - Categorized

## 🔹 `model = Sequential([...])`

### 1. Sequential Model
- **Concept**: Linear stack of layers.  
- **Use when**: simple, single input/output.  
- **Don’t use when**: skip connections, multiple inputs/outputs.  

**Parameter choices**:  
- **layers**: Dense (tabular), Conv2D (images), Conv1D/LSTM (sequences).  
- **Input(shape=...)**: must match data (e.g., `(28,28,1)` for MNIST).  

---

#### Dense Layer ( included by .layers)
```python
layers.Dense(units, activation=None, use_bias=True, kernel_initializer="glorot_uniform")
```

**Parameter choices**:  
- **units**:  
  - Hidden layers → 32–512.  
  - Output → depends on task (1 for binary/regression, #classes for multi-class).  
- **activation**:  
  - Hidden → `relu`.  
  - Output → `sigmoid` (binary), `softmax` (multi-class), `linear` (regression).  
- **kernel_initializer**:  
  - `he_normal` (ReLU).  
  - `glorot_uniform` (default).  
- **use_bias**: keep True (unless BatchNorm follows).  

---

## 🔹 `model.compile(optimizer=..., loss=..., metrics=...)`

### 2. Loss Functions
- **Binary** → `BinaryCrossentropy(from_logits=...)`.  
- **Multi-class** →  
  - One-hot → `CategoricalCrossentropy(from_logits=...)`.  
  - Integer labels → `SparseCategoricalCrossentropy(from_logits=...)`.  
- **Multi-label** → `BinaryCrossentropy`.  
- **Regression** → `MSE` or `MAE`.  

**Key parameters**:  
- `from_logits`: True if no activation on output.  
- `label_smoothing`: 0.1–0.2 to reduce overconfidence.  
- `reduction`: how to aggregate loss (`auto` by default).  

---

### 3. Metrics
- **Binary classification**: `accuracy`, `AUC`, `Precision/Recall`.  
- **Multi-class**:  
  - `SparseCategoricalAccuracy` (integer labels).  
  - `CategoricalAccuracy` (one-hot).  
- **Regression**: `mae`, `mse`, `RMSE`, `R2` (custom).  

👉 **Loss = optimized, Metrics = monitoring only.**  

---

### 4. Optimizers
- **Adam** → `learning_rate=1e-3`.  
- **AdamW** → add weight decay (`1e-4`).  
- **SGD+Momentum** → `lr=1e-2 ~ 1e-1`, `momentum=0.9`.  
- **RMSprop** → often for RNN/audio.  

**Tips**:  
- Use `ReduceLROnPlateau` to lower LR dynamically.  
- Use `clipnorm=1.0` to avoid exploding gradients.  
- Try LR schedules (CosineDecay, ExponentialDecay).  

---

## 🔹 `model.fit(x, y, ...)`

### 5. Fit
```python
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    shuffle=True,
    class_weight=None,
    callbacks=[...],
    verbose=1
)
```

**Parameter choices**:  
- **epochs**: 20–50 + `EarlyStopping(patience=5)`.  
- **batch_size**: 32 (default), 64/128 if GPU strong.  
- **validation_split**: 0.2 if no dedicated val set.  
- **shuffle**: True (default), False for time-series.  
- **class_weight**: adjust for imbalanced data.  
- **callbacks**:  
  - EarlyStopping  
  - ModelCheckpoint  
  - ReduceLROnPlateau  
- **verbose**: 1 (detailed), 2 (compact).  

---

# ✅ Final Flow
1. **Build** → `Sequential` + layers (Section 1–2).  
2. **Compile** → loss, metrics, optimizer (Section 3–5).  
3. **Fit** → train model with data (Section 6).  
