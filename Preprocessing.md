## 📘 Preprocessing Cheatsheet by Data Type

| 📂 Data Type | ⚙️ Main Preprocessing Steps | 🛠️ Common Methods / Functions | 🔍 Recognition Trick |
|-------------|-----------------------------|-------------------------------|-----------------------|
| 🖼️ Image | 1. Resize<br>2. Normalize pixel values<br>3. Data Augmentation<br>4. Encode labels | `tf.image.resize()`<br>`tf.cast(img, tf.float32)/255`<br>`ImageDataGenerator`,<br>`tf.image.random_flip_left_right`<br>`LabelEncoder`, `to_categorical()` | `.jpg`, `.png`, shape `(H, W, C)` |
| 📄 Text | 1. Lowercase<br>2. Remove punctuation / stopwords<br>3. Tokenize<br>4. Padding<br>5. Vectorization (TF-IDF / BERT) | `.lower()`, `re.sub()`<br>`nltk.corpus.stopwords`, `re`<br>`Tokenizer().fit_on_texts()`<br>`pad_sequences()`<br>`TfidfVectorizer`, `BERTTokenizer` | Files like `.txt`, `.csv` containing sentences or paragraphs |
| 📊 Tabular | 1. Handle missing values<br>2. Encode categorical variables<br>3. Scale numerical features<br>4. Feature selection | `SimpleImputer`, `dropna()`<br>`OneHotEncoder`, `LabelEncoder`, `get_dummies()`<br>`StandardScaler`, `MinMaxScaler`<br>`SelectKBest` | CSV/XLS with multiple columns and data types |
| 📈 Time-Series | 1. Parse timestamp<br>2. Resampling<br>3. Normalize<br>4. Sliding Window | `pd.to_datetime()`<br>`df.resample('D').mean()`<br>`StandardScaler`, `MinMaxScaler`<br>`sliding_window_view` | Has `"timestamp"` column, time-sequenced data |
| 🔊 Audio | 1. Feature extraction (MFCC, Spectrogram)<br>2. Normalize<br>3. Noise reduction<br>4. Sampling | `librosa.feature.mfcc()`<br>`librosa.amplitude_to_db()`<br>`scipy.signal`,<br>`noisereduce.reduce_noise()`<br>`librosa.resample()` | `.wav`, `.mp3` files or waveform arrays |
| 🎞️ Video | 1. Extract frames<br>2. Resize frames<br>3. Normalize<br>4. Sampling | `cv2.VideoCapture().read()`<br>`cv2.resize()`<br>`frame/255.0`<br>Sampling by FPS or time | `.mp4`, `.avi` files or image sequences |



## Goal for pipeline for every Data Type :

| ✅ **Step**                | ✅ **Purpose / Description**                                                                 |
|---------------------------|----------------------------------------------------------------------------------------------|
| **1. Load the dataset**   | Read and import the raw data into the pipeline (from disk, database, web, etc.)              |
| **2. Normalize the data** | Standardize or scale the data (e.g. pixel values to [0, 1], text tokenization, feature scaling) |
| **3. Shuffle the data**   | Randomize the order of data to prevent model overfitting or learning from sequence bias      |
| **4. Batch the data**     | Divide the data into fixed-size groups (batches) to enable efficient training/inference      |
| **5. Prefetch the data**  | Load batches ahead of time during training to improve pipeline throughput and efficiency     |


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

