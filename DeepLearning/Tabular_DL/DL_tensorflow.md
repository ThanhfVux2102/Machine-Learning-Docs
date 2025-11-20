# Deep Learning Workflow for Tabular Data with TensorFlow

Mục tiêu: Dùng **TensorFlow (Keras)** để xây dựng một pipeline Deep Learning (MLP) cho dữ liệu tabular (ví dụ CIC-IDS, credit scoring, churn, employee performance…) và so sánh tư duy với ML truyền thống.

---

## 1. Tổng quan workflow

Cho một bài toán classification với data dạng bảng:

1. **Hiểu & load data**

   - Đọc file (CSV, Parquet…) bằng pandas.
   - Xem sơ bộ: số mẫu, số cột, kiểu dữ liệu, phân bố nhãn.

2. **Tiền xử lý (preprocessing)**

   - Xoá cột vô nghĩa (ID, timestamp nếu không dùng).
   - Xử lý giá trị thiếu (NaN, inf).
   - Tách features X và labels y.
   - Encode label (nếu là string).
   - Scale feature số (StandardScaler / MinMaxScaler).
   - Chia train / validation / test.

3. **Đưa dữ liệu vào TensorFlow**

   - Chuyển X, y sang TensorFlow tensors.
   - Tạo Dataset cho tabular.
   - Tạo DataLoader cho batch + shuffle (với TensorFlow, sử dụng `tf.data.Dataset`).

4. **Định nghĩa model (MLP)**

   - Dùng Keras API với `tf.keras.Model`.
   - Sử dụng `Dense`, activation (ReLU), Dropout, v.v.

5. **Khai báo loss + optimizer**

   - Classification: `tf.keras.losses.SparseCategoricalCrossentropy`.
   - Optimizer: `tf.keras.optimizers.Adam` (thường dùng).

6. **Training loop**

   - Cho từng epoch:
     - Train: loop qua train_dataset, forward → loss → backward → optimizer.step().
     - Eval: loop qua val_dataset, tính loss + accuracy để theo dõi overfitting.

7. **Đánh giá trên test set**

   - Load best checkpoint.
   - Tính test loss, accuracy và các metric khác (F1, ROC-AUC… nếu cần).

8. **Lưu & deploy model**
   - `model.save("best_model.h5")`.
   - Khi deploy:
     - Load lại model + scaler + encoder.
     - Tiền xử lý input giống hệt lúc train.
     - Gọi `model(x)` để predict.

---

## 2. Chuẩn bị thư viện

Ví dụ code:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
```

---

## 3. Load & tiền xử lý dữ liệu tabular

Ví dụ với file `data.csv`, cột nhãn tên là "label", có vài cột không dùng như "Flow ID", "Timestamp":

```python
# 1. Load data
df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df["label"].value_counts())  # phân bố nhãn
```

### 3.1. Xoá cột không cần thiết

```python
cols_drop = ["Flow ID", "Timestamp"]  # đổi theo dataset thật
df = df.drop(columns=[c for c in cols_drop if c in df.columns])
```

### 3.2. Xử lý missing / inf

```python
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()  # đơn giản: bỏ các dòng có NaN (có thể dùng fillna nếu muốn)
```

### 3.3. Tách features và label

```python
X = df.drop(columns=["label"])  # đổi "label" theo tên cột nhãn
y = df["label"]
```

### 3.4. Encode label

Nếu nhãn dạng string (e.g. "Normal", "Attack", "DoS"), dùng `LabelEncoder`:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # np.array các số 0,1,2,...
```

### 3.5. Chia train / val / test

```python
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.2,
    random_state=42,
    stratify=y_train_val
)
```

### 3.6. Scale features số

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
```

---

## 4. Dataset & DataLoader trong TensorFlow

### 4.1. Tạo Dataset

```python
import tensorflow as tf

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

train_dataset = train_dataset.batch(256).shuffle(10000)
val_dataset = val_dataset.batch(256)
test_dataset = test_dataset.batch(256)
```

---

## 5. Định nghĩa MLP model với Keras

### 5.1. Thiết kế kiến trúc

```python
class MLP(tf.keras.Model):
    def __init__(self, in_dim, num_classes):
        super(MLP, self).__init__()
        self.dense1 = Dense(128, activation='relu')
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(64, activation='relu')
        self.dropout2 = Dropout(0.3)
        self.out = Dense(num_classes)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.out(x)

# Khởi tạo model
num_features = X_train.shape[1]
num_classes  = len(np.unique(y_encoded))

model = MLP(num_features, num_classes)
```

---

## 6. Loss function & optimizer

### 6.1. Chọn loss

Classification multi-class: dùng `tf.keras.losses.SparseCategoricalCrossentropy`.

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```

### 6.2. Chọn optimizer

Dùng Adam:

```python
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
```

---

## 7. Training loop chi tiết

### 7.1. Hàm train một epoch

```python
def train_one_epoch(model, dataset, optimizer, loss_fn):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total = 0

    for X_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            logits = model(X_batch, training=True)
            loss = loss_fn(y_batch, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss += loss.numpy() * X_batch.shape[0]
        preds = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(preds, y_batch), tf.float32))
        epoch_accuracy += accuracy.numpy()
        total += X_batch.shape[0]

    return epoch_loss / total, epoch_accuracy / total
```

### 7.2. Hàm evaluate

```python
@tf.function
def eval_one_epoch(model, dataset, loss_fn):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total = 0

    for X_batch, y_batch in dataset:
        logits = model(X_batch, training=False)
        loss = loss_fn(y_batch, logits)

        epoch_loss += loss.numpy() * X_batch.shape[0]
        preds = tf.argmax(logits, axis=1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(preds, y_batch), tf.float32))
        epoch_accuracy += accuracy.numpy()
        total += X_batch.shape[0]

    return epoch_loss / total, epoch_accuracy / total
```

---

## 8. Đánh giá trên test set

```python
model.load_weights("best_mlp.h5")
test_loss, test_acc = eval_one_epoch(model, test_dataset, loss_fn)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
```

Nếu muốn F1-score, precision, recall, dùng thêm sklearn:

```python
from sklearn.metrics import classification_report

all_preds = []
all_targets = []

for X_batch, y_batch in test_dataset:
    logits = model(X_batch, training=False)
    preds = tf.argmax(logits, axis=1)
    all_preds.append(preds.numpy())
    all_targets.append(y_batch.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

print(classification_report(all_targets, all_preds))
```

---

## 9. Lưu & sử dụng lại model

Lưu:

```python
model.save("best_mlp.h5")
```

Load lại để infer:

```python
model = tf.keras.models.load_model("best_mlp.h5")
model.eval()
```

Khi dùng với input mới:

1. Áp dụng cùng scaler đã fit trước đó:
   - `x_new_scaled = scaler.transform(x_new)`
2. Chuyển sang tensor:
   - `x_tensor = tf.convert_to_tensor(x_new_scaled, dtype=tf.float32)`
3. Gọi model:

```python
logits = model(x_tensor)
preds = tf.argmax(logits, axis=1)
```

---

## 10. Tài liệu tham khảo (TensorFlow docs)

- TensorFlow documentation: https://www.tensorflow.org/api_docs
- `tf.keras.Model`: https://www.tensorflow.org/api_docs/python/tf/keras/Model
- `tf.keras.layers.Dense`: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
- `tf.keras.losses.SparseCategoricalCrossentropy`: https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy
- `tf.data.Dataset`: https://www.tensorflow.org/api_docs/python/tf/data/Dataset
