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
| `reshuffle_each_iteration`| Nếu muốn dữ liệu shuffle lại mỗi epoch                     | Mặc định là `True`, chỉ tắt khi debug                                     |

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
