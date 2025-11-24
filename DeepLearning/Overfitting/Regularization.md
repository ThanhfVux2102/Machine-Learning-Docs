
# Machine Learning Regularization & Cross-Validation Cheatsheet

## 1. **Regularization Overview**
Regularization là kỹ thuật giúp giảm thiểu **overfitting** trong mô hình học máy bằng cách thêm vào một phần bồi dưỡng (penalty) vào hàm mất mát (loss function).

### 1.1 **L2 Regularization (Ridge Regression)**
- **Mục tiêu:** Giảm thiểu giá trị trọng số mà không làm chúng bằng 0 hoàn toàn.
- **Hàm mất mát:**
  \[
  	ext{Loss} = 	ext{MSE} + \lambda \sum_{i=1}^{n} w_i^2
  \]
- **Ứng dụng:** Khi bạn muốn giảm thiểu sự ảnh hưởng của các trọng số lớn mà không loại bỏ tính năng.

#### **Ưu điểm:**
- Đơn giản và dễ áp dụng.
- Giúp giảm thiểu overfitting mà không làm mất đi bất kỳ tính năng nào.
- Thường hiệu quả trong các bài toán có nhiều tính năng có mối quan hệ với nhau.

#### **Nhược điểm:**
- Không thực hiện lựa chọn tính năng (feature selection), nghĩa là vẫn giữ lại tất cả các tính năng, kể cả những tính năng không quan trọng.

#### **Ví dụ:**
```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)  
ridge.fit(X_train, y_train)
```

### 1.2 **L1 Regularization (Lasso Regression)**
- **Mục tiêu:** Giúp loại bỏ các tính năng không quan trọng bằng cách đưa trọng số của chúng về 0.
- **Hàm mất mát:**
  \[
  	ext{Loss} = 	ext{MSE} + \lambda \sum_{i=1}^{n} |w_i|
  \]
- **Ứng dụng:** Khi bạn cần thực hiện **feature selection** (lựa chọn tính năng).

#### **Ưu điểm:**
- Thực hiện **feature selection**, giúp loại bỏ các tính năng không quan trọng.
- Làm giảm độ phức tạp của mô hình bằng cách giữ lại các tính năng quan trọng.

#### **Nhược điểm:**
- Có thể loại bỏ quá nhiều tính năng nếu không điều chỉnh \( \lambda \) cẩn thận.
- Không hiệu quả khi các tính năng có mối quan hệ mạnh với nhau, vì nó có thể loại bỏ một tính năng quan trọng dù có liên quan đến các tính năng khác.

#### **Ví dụ:**
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

### 1.3 **Elastic Net**
- **Mục tiêu:** Kết hợp ưu điểm của cả L1 và L2 regularization.
- **Hàm mất mát:**
  \[
  	ext{Loss} = 	ext{MSE} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2
  \]
- **Ứng dụng:** Khi bạn không chắc chắn chọn giữa L1 hay L2, và bạn muốn kết hợp cả hai.

#### **Ưu điểm:**
- Kết hợp khả năng lựa chọn tính năng của L1 và khả năng giảm thiểu trọng số của L2.
- Thích hợp cho bài toán có nhiều tính năng và tính năng phụ thuộc lẫn nhau.

#### **Nhược điểm:**
- Cần phải điều chỉnh cẩn thận hai tham số \( \lambda_1 \) và \( \lambda_2 \).
- Có thể phức tạp hơn trong việc tối ưu tham số so với L1 và L2 đơn lẻ.

#### **Ví dụ:**
```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.7)
elastic_net.fit(X_train, y_train)
```

### 1.4 **Dropout (Trong Neural Networks)**
- **Mục tiêu:** Giảm thiểu overfitting trong mạng neural bằng cách bỏ qua một số neurons trong quá trình huấn luyện.

#### **Ưu điểm:**
- Giảm sự phụ thuộc vào một số neurons, giúp mô hình học được các đặc trưng tổng quát hơn.
- Tăng khả năng tổng quát của mạng neural.

#### **Nhược điểm:**
- Cần thời gian huấn luyện lâu hơn vì phải thực hiện thêm bước ngẫu nhiên trong quá trình huấn luyện.
- Có thể gây khó khăn trong việc điều chỉnh các tham số như tỷ lệ dropout.

#### **Ví dụ trong Keras (Deep Learning):**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
model = Sequential([
    Dense(128, input_dim=8, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

### 1.5 **Early Stopping**
- **Mục tiêu:** Dừng huấn luyện sớm khi mô hình không còn cải thiện trên tập kiểm tra.

#### **Ưu điểm:**
- Giúp tránh overfitting và tiết kiệm thời gian huấn luyện.
- Đảm bảo mô hình không được huấn luyện quá mức.

#### **Nhược điểm:**
- Cần một tập validation tốt để có thể theo dõi sự cải thiện của mô hình.
- Có thể dừng quá sớm, dẫn đến mô hình chưa học đủ.

#### **Ví dụ trong Keras (Deep Learning):**
```python
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopping])
```

---

## 2. **Cross-Validation**

### 2.1 **Cross-Validation là gì?**
Cross-validation là một kỹ thuật giúp kiểm tra hiệu suất của mô hình một cách chính xác hơn. Kỹ thuật này chia dữ liệu thành các phần (folds), huấn luyện mô hình trên một phần và kiểm tra trên phần còn lại.

### 2.2 **Kỹ thuật k-Fold Cross-Validation**
- **Mục tiêu:** Chia dữ liệu thành **k phần**, huấn luyện trên **k-1 phần** và kiểm tra trên phần còn lại. Quá trình này lặp lại **k lần**.
- **Ưu điểm:**
  - Giảm thiểu overfitting.
  - Cung cấp một đánh giá mô hình chính xác hơn.
- **Nhược điểm:**
  - Tốn thời gian vì mô hình phải được huấn luyện nhiều lần.

#### **Ví dụ:**
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)

# Sử dụng k-fold cross-validation (5-fold)
cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_scores = -cv_scores  # Chuyển từ MSE âm sang dương
print(f"Mean CV Score: {cv_scores.mean():.3f}")
```

### 2.3 **Leave-One-Out Cross-Validation (LOO)**
- **Mục tiêu:** Mỗi mẫu trong dữ liệu được sử dụng làm tập kiểm tra một lần duy nhất. Kỹ thuật này rất tốn kém nhưng có thể cung cấp độ chính xác cao.

#### **Ưu điểm:**
- Đánh giá chính xác mô hình, đặc biệt khi dữ liệu nhỏ.

#### **Nhược điểm:**
- Tốn nhiều thời gian và tài nguyên tính toán.

#### **Ví dụ:**
```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
cv_scores = cross_val_score(ridge, X_train, y_train, cv=loo, scoring='neg_mean_squared_error')
cv_scores = -cv_scores
print(f"Mean LOO CV Score: {cv_scores.mean():.3f}")
```

### 2.4 **Stratified k-Fold Cross-Validation**
- **Mục tiêu:** Đảm bảo rằng mỗi fold có phân phối tỷ lệ lớp giống với toàn bộ tập dữ liệu, đặc biệt hữu ích cho các bài toán phân loại không cân bằng.

#### **Ưu điểm:**
- Giúp cải thiện độ chính xác cho các bài toán phân loại với dữ liệu không cân bằng.

#### **Nhược điểm:**
- Cũng tốn thời gian và tài nguyên tính toán như k-Fold.

#### **Ví dụ:**
```python
from sklearn.model_selection import StratifiedKFold
stratified_kfold = StratifiedKFold(n_splits=5)

for train_idx, test_idx in stratified_kfold.split(X_train, y_train):
    X_train_fold, X_test_fold = X_train[train_idx], X_train[test_idx]
    y_train_fold, y_test_fold = y_train[train_idx], y_train[test_idx]
    ridge.fit(X_train_fold, y_train_fold)
    print(f"Test Score: {ridge.score(X_test_fold, y_test_fold)}")
```

---

## 3. **Lựa chọn Regularization và Cross-Validation**

### 3.1 **Khi nào sử dụng L2 Regularization?**
- Dữ liệu có nhiều tính năng liên quan (correlated features).
- Không cần lựa chọn tính năng (feature selection).
- Mô hình bị overfitting vì quá phức tạp.

### 3.2 **Khi nào sử dụng L1 Regularization?**
- Khi bạn muốn lựa chọn tính năng tự động (feature selection).
- Dữ liệu có nhiều tính năng không quan trọng hoặc dư thừa.

### 3.3 **Khi nào sử dụng Elastic Net?**
- Khi bạn không chắc chắn giữa L1 và L2 và muốn kết hợp cả hai.
- Dữ liệu có mối quan hệ giữa các tính năng nhưng bạn vẫn muốn loại bỏ tính năng dư thừa.

### 3.4 **Khi nào sử dụng Cross-Validation?**
- Để đánh giá mô hình một cách chính xác hơn và tránh overfitting.
- Khi bạn muốn kiểm tra hiệu suất của mô hình trên các phần khác nhau của dữ liệu.
- Sử dụng cross-validation khi bạn có một bộ dữ liệu nhỏ hoặc không chắc chắn về hiệu suất mô hình.

---

## 4. **Tổng kết**
- **L2 Regularization**: Thích hợp khi bạn muốn giảm thiểu độ phức tạp của mô hình mà không loại bỏ tính năng.
- **L1 Regularization**: Phù hợp khi bạn muốn tự động loại bỏ các tính năng không quan trọng.
- **Elastic Net**: Kết hợp ưu điểm của cả L1 và L2.
- **Cross-Validation**: Làm giảm overfitting và đánh giá chính xác hơn hiệu suất của mô hình.
  - **k-Fold**: Phổ biến cho hầu hết các bài toán.
  - **Leave-One-Out (LOO)**: Dùng khi bạn muốn đánh giá mô hình cực kỳ chính xác.
  - **Stratified k-Fold**: Dùng cho bài toán phân loại khi dữ liệu không cân bằng.

---

