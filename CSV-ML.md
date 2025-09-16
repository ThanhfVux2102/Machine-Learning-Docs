# 📝 Tutorial: Machine Learning Pipeline with Scikit‑learn

## 1) Data Splitting

### `train_test_split` (from `sklearn.model_selection`)
Chia dữ liệu thành tập **train** và **test**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- **`test_size`**: tỷ lệ dữ liệu test (ví dụ: `0.2` = 20%).
- **`random_state`**: seed cố định, đảm bảo tái lập kết quả.
- **`stratify`**: nếu = `y`, giữ đúng tỷ lệ nhãn khi chia train/test.

---

## 2) Data Normalization / Scaling

Chuẩn hoá dữ liệu số giúp mô hình hội tụ nhanh hơn và tránh việc feature có thang đo lớn lấn át feature khác.

**Phương pháp phổ biến:**
- `StandardScaler`: chuẩn hoá về mean=0, std=1.
- `MinMaxScaler`: đưa dữ liệu về [0, 1].
- `RobustScaler`: bền với outliers (dựa trên median & IQR).
- `Normalizer`: chuẩn hoá theo vector norm (L1/L2) trên từng hàng (mẫu).

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
```

---

## 3) Encoding (Categorical Features)

Biến đổi dữ liệu dạng chuỗi thành số:

- `OneHotEncoder`: tạo dummy variables (0/1) cho từng hạng mục.
- `LabelEncoder`: ánh xạ hạng mục → số nguyên (chủ yếu dùng cho **target**).
- `OrdinalEncoder`: gán số theo thứ tự (small < medium < large).
- Target Encoding (ngoài sklearn – dùng `category_encoders`).

```python
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
```

---

## 4) ColumnTransformer

Kết hợp nhiều bước xử lý cho từng loại cột trong **một** đối tượng.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = [...]      # ví dụ: ['Total Cases','Total Deaths',...]
categorical_features = [...]  # ví dụ: ['Region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features),
    ],
    remainder='drop'  # hoặc 'passthrough' nếu muốn giữ lại cột khác
)
```

---

## 5) Model Options



### a) Classification
- `SGDClassifier(loss="log_loss")` – Logistic Regression với Gradient Descent.
- `LogisticRegression`
- `RandomForestClassifier`
- `SVC`
- `KNeighborsClassifier`
# 📝 SGDClassifier Cheatsheet (scikit-learn)

Bạn đang dùng **SGDClassifier** trong thư viện `sklearn.linear_model`.\
Đây là một classifier dùng **Stochastic Gradient Descent (SGD)** để tối
ưu các mô hình tuyến tính như **Logistic Regression, SVM**, v.v.

------------------------------------------------------------------------

## 📌 Ví dụ code

``` python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)
```

------------------------------------------------------------------------

## 📌 Ý nghĩa từng tham số

### `loss="log_loss"`

-   Chỉ định hàm mất mát (loss function).\
-   `"log_loss"` nghĩa là dùng **Logistic Regression** (tương đương phân
    loại nhị phân/multi-class bằng sigmoid/softmax).

**Một số lựa chọn khác:** - `"hinge"` → SVM tuyến tính.\
- `"squared_error"` → Linear Regression.\
- `"modified_huber"` → robust classification (ít nhạy với outlier).

------------------------------------------------------------------------

### `max_iter=1000`

-   Số lần lặp tối đa (epochs) qua toàn bộ dữ liệu.\
-   SGD có thể **dừng sớm** nếu hội tụ trước khi đạt tới số lần lặp này.

------------------------------------------------------------------------

### `tol=1e-3`

-   Ngưỡng dừng (tolerance).\
-   Nếu độ cải thiện nhỏ hơn `1e-3` trong một số vòng lặp liên tiếp,
    thuật toán sẽ **dừng sớm** (early stopping).\
-   Giá trị nhỏ hơn (ví dụ `1e-4`) → yêu cầu hội tụ kỹ hơn nhưng chạy
    lâu hơn.

------------------------------------------------------------------------

### `random_state=42`

-   Đặt seed để việc ngẫu nhiên trong SGD (chọn sample, shuffle) **tái
    lập được**.\
-   Nếu bỏ qua, mỗi lần chạy có thể cho kết quả hơi khác nhau.

------------------------------------------------------------------------

## 📌 Một số tham số quan trọng khác

-   **`penalty`**: loại regularization (`"l2"` mặc định, `"l1"`,
    `"elasticnet"`) → giúp tránh overfitting.\
-   **`alpha`**: hệ số regularization (mặc định `0.0001`), tăng lên →
    regularize mạnh hơn.\
-   **`learning_rate`**: cách cập nhật step size (`"optimal"`,
    `"constant"`, `"invscaling"`, `"adaptive"`).\
-   **`eta0`**: learning rate ban đầu (dùng khi
    `learning_rate="constant"` hoặc `"invscaling"`).

------------------------------------------------------------------------

## 📌 Các giá trị của `loss` trong SGDClassifier

### 🔹 Phân loại (Classification)

-   `"hinge"` → Linear SVM (Support Vector Machine) với hinge loss.\
-   `"log_loss"` → Logistic Regression (multinomial hoặc binary
    classification).\
-   `"modified_huber"` → robust version của hinge loss (ít nhạy cảm với
    outlier).\
-   `"squared_hinge"` → hinge loss bình phương (SVM biến thể).

### 🔹 Hồi quy (Regression)

-   `"squared_error"` → Linear Regression (mse loss).\
-   `"huber"` → Huber loss (lai giữa squared error và absolute error,
    robust với outlier).\
-   `"epsilon_insensitive"` → Linear SVR với epsilon-insensitive loss.\
-   `"squared_epsilon_insensitive"` → SVR với squared
    epsilon-insensitive loss.

------------------------------------------------------------------------

## 📌 Khi nào chọn cái nào?

-   **Classification**
    -   `hinge`: muốn huấn luyện SVM tuyến tính.\
    -   `log_loss`: logistic regression (có xác suất output).\
    -   `modified_huber`: classification nhưng có tolerance với outlier.
-   **Regression**
    -   `squared_error`: hồi quy tuyến tính.\
    -   `huber`: hồi quy có outlier.\
    -   `epsilon_insensitive`: hồi quy theo style SVR (chỉ quan tâm sai
        số \> epsilon).

------------------------------------------------------------------------



------------------------------------------------------------------------

### b) Regression
- `SGDRegressor`
- `LinearRegression`
- `RandomForestRegressor`
- `SVR`
- `KNeighborsRegressor`

---

## 6) Pipeline

Gom tất cả bước lại để tránh rò rỉ dữ liệu và code gọn hơn.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

clf = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42))
])
```

---

## 7) Evaluation Metrics

### Classification
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
```

### Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
```

### Visualization (Confusion Matrix)
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
```

---

## 8) Handling Missing Values (2 cách chính)

### Cách A – **Loại bỏ** (Deletion)
Dùng khi dữ liệu thiếu rất ít hoặc cột/hàng không quan trọng.
```python
# Xoá hàng có bất kỳ NaN
df_drop_rows = df.dropna()

# Xoá cột thiếu quá nhiều (ví dụ > 40%)
thresh = int(0.6 * len(df))       # giữ cột có >= 60% giá trị
df_drop_cols = df.dropna(axis=1, thresh=thresh)
```

### Cách B – **Điền giá trị** (Imputation)
Giữ lại tối đa dữ liệu, phù hợp với Worldometer.
```python
from sklearn.impute import SimpleImputer
import numpy as np

# Tự động tách loại cột
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

# Điền cho numeric: median (ít bị ảnh hưởng bởi outlier)
df[num_cols] = SimpleImputer(strategy='median').fit_transform(df[num_cols])

# Điền cho categorical: most_frequent
if len(cat_cols) > 0:
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])

# Kiểm tra còn thiếu không
print(df.isnull().sum().sort_values(ascending=False).head(10))
```

> Lưu ý: Với các cột định danh như `Country`, thường **không dùng làm feature**; nếu dùng `Region/Continent` thì cần One‑Hot Encoding.

---

## 9) Correlation Matrix & Heatmap (để chọn lọc đặc trưng)

Dùng để phát hiện:
- Feature tương quan mạnh/yếu với nhãn (`DeathRate` hoặc `DeathRateCategory`).
- Cặp features tương quan quá cao với nhau → cân nhắc giữ một để giảm trùng lặp.

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ví dụ tạo nhãn liên tục và phân lớp
df['DeathRate'] = df['Total Deaths'] / df['Total Cases']
def categorize(rate):
    if rate < 0.01: return 0
    elif rate < 0.03: return 1
    return 2
df['DeathRateCategory'] = df['DeathRate'].apply(categorize)

# Ma trận tương quan cho numeric + DeathRate
num_for_corr = df.select_dtypes(include=['int64','float64']).columns.tolist()
corr = df[num_for_corr + ['DeathRate']].corr(method='pearson')

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap='coolwarm', center=0)
plt.title('Correlation heatmap')
plt.show()
```

**Gợi ý chọn lọc feature:**
- Giữ feature có |corr với `DeathRate` cao.
- Loại 1 trong 2 feature nếu chúng tương quan rất cao (ví dụ |corr| > 0.9).
- Tránh đưa cột đã tính sẵn theo đầu người (`Tot Cases/1M pop`, `Deaths/1M pop`, `Tests/1M pop`) nếu đã có tổng số tương ứng, để giảm multicollinearity.

---

## 10) Tóm tắt nhanh
- **Splitting**: `train_test_split` (nhớ `stratify=y` cho bài toán phân lớp mất cân bằng).
- **Missing values**: ưu tiên **Imputation** (median / most_frequent); chỉ xoá khi thiếu quá ít.
- **Scaling**: `StandardScaler` cho numeric (đặc biệt khi tối ưu bằng Gradient Descent).
- **Encoding**: `OneHotEncoder` cho categorical (nếu dùng).
- **Feature selection**: dùng **correlation matrix + heatmap** và/hoặc `SelectKBest`, `RFE`.
- **Evaluation**: `accuracy`, `confusion_matrix`, `classification_report` cho classification.

