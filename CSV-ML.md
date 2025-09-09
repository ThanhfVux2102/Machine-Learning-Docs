# 📝 Tutorial: Machine Learning Pipeline with Scikit-learn

## 1. Data Splitting

### `train_test_split` (from `sklearn.model_selection`)
- Chia dữ liệu thành tập **train** và **test**.

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- **`test_size`**: tỷ lệ dữ liệu test (ví dụ: 0.2 = 20%).
- **`random_state`**: seed cố định, đảm bảo tái lập kết quả.
- **`stratify`**: nếu = `y`, giữ đúng tỷ lệ nhãn khi chia train/test.

---

## 2. Data Normalization / Scaling

Chuẩn hóa dữ liệu số giúp mô hình hội tụ nhanh hơn, tránh dominance của feature lớn.

### Phương pháp phổ biến:
- **`StandardScaler`**: chuẩn hóa dữ liệu về mean=0, std=1.  
- **`MinMaxScaler`**: scale dữ liệu về [0,1].  
- **`RobustScaler`**: ít nhạy với outliers, dựa trên median và IQR.  
- **`Normalizer`**: chuẩn hóa theo vector norm (L1, L2).  

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
```

---

## 3. Encoding (Categorical Features)

Biến đổi dữ liệu dạng chuỗi thành số.

### Các phương pháp:
- **`OneHotEncoder`**: tạo dummy variables (0/1) cho từng hạng mục.
- **`LabelEncoder`**: ánh xạ mỗi giá trị thành 1 số nguyên (chỉ dùng cho target hoặc khi feature có thứ tự).
- **`OrdinalEncoder`**: gán số theo thứ tự cho feature dạng ordinal (ví dụ: small < medium < large).
- **`Target Encoding`** (ngoài sklearn, dùng thư viện `category_encoders`): thay hạng mục bằng mean của target.

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
```

---

## 4. ColumnTransformer

Kết hợp nhiều bước xử lý cho từng loại cột.

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='drop'
)
```

- **`remainder='drop'`**: bỏ cột không xử lý.
- **`remainder='passthrough'`**: giữ nguyên cột không xử lý.

---

## 5. Model Options

### a) Classification
- **`SGDClassifier(loss="log_loss")`**: Logistic Regression với Gradient Descent.
- **`LogisticRegression`**: mô hình logistic regression chuẩn.
- **`RandomForestClassifier`**: cây quyết định ensemble, mạnh mẽ, ít cần scaling.
- **`SVC` (Support Vector Machine)**: tốt với dữ liệu không tuyến tính.
- **`KNeighborsClassifier`**: dựa trên khoảng cách, đơn giản nhưng hiệu quả.

### b) Regression
- **`SGDRegressor`**: Linear Regression bằng Gradient Descent.
- **`LinearRegression`**: mô hình tuyến tính chuẩn.
- **`RandomForestRegressor`**: ensemble trees cho regression.
- **`SVR`**: Support Vector Regression.
- **`KNeighborsRegressor`**: tương tự KNN nhưng cho regression.

---

## 6. Pipeline

Dùng `Pipeline` để gom tất cả bước lại.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42))
])
```

- **`steps`**: list `(tên bước, object)`.
- Các bước đầu là preprocessing, bước cuối là model.

---

## 7. Evaluation Metrics

### Classification
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

- **`accuracy_score`**: tỷ lệ dự đoán đúng.
- **`confusion_matrix`**: ma trận dự đoán.
- **`classification_report`**: Precision, Recall, F1-score.

### Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

- **`MSE`**: trung bình bình phương sai số.
- **`MAE`**: trung bình sai số tuyệt đối.
- **`R2`**: hệ số xác định (1 là tốt nhất).

---

## 8. Visualization

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

---

## ✅ Summary

- **Splitting**: `train_test_split` để chia dữ liệu.  
- **Scaling**: chọn giữa Standard, MinMax, Robust, Normalizer.  
- **Encoding**: chọn OneHot, Label, Ordinal, hoặc Target Encoding.  
- **Model**: chọn Classification (SGD, Logistic, RF, SVM, KNN) hoặc Regression (SGDRegressor, Linear, RF, SVR, KNN).  
- **Pipeline**: gom các bước lại để code gọn và ít lỗi.  
- **Evaluation**: metrics cho classification hoặc regression.  
- **Visualization**: confusion matrix, learning curve, feature importance.  
