# Core Deep Learning Training Loop (with PyTorch)

This note summarizes the core concepts that appear in **every** deep learning training loop, especially when using **PyTorch**.

---

## 0. Minimal Training Loop (PyTorch)

```python
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # 1. Forward pass
        y_pred = model(x_batch)

        # 2. Compute loss
        loss = loss_fn(y_pred, y_batch)

        # 3. Backward pass (backprop)
        loss.backward()

        # 4. Update parameters
        optimizer.step()
        optimizer.zero_grad()
```

Tất cả các khái niệm bên dưới đều xoay quanh đoạn code này.

---

## 1. Model & Parameters (Weights, Bias)

- **Model** trong PyTorch thường là class kế thừa `torch.nn.Module`.
- **Parameters** là các tensor kiểu `torch.nn.Parameter` được model tự động đăng ký và sẽ được optimizer cập nhật.

Ví dụ:

```python
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)  # has weights & bias inside

    def forward(self, x):
        return self.fc(x)

model = MyNet()
params = list(model.parameters())
```

**Tài liệu tham khảo:**

- PyTorch `nn.Module`: vào trang chính [pytorch.org](https://pytorch.org) → Docs → PyTorch → API → `torch.nn`
- Mục `nn.Parameter` trong phần `torch.nn` docs

---

## 2. Forward Pass

**Forward pass** = gửi input đi qua các layer của model để lấy output (logits, prediction,…).

Trong PyTorch, bạn cài đặt bằng hàm `forward`, và gọi bằng `model(x)`:

```python
y_pred = model(x_batch)   # forward pass
```

**Model** là một đồ thị tính toán (computation graph) được xây dựng trong lúc forward, để phục vụ cho autograd ở bước backward sau.

**Ref docs:**

- PyTorch Tutorials → “Learning PyTorch with Examples”
- PyTorch Tutorials → “Neural Networks”

---

## 3. Loss Function

**Loss function** đo “độ sai” giữa `y_pred` và `y_true` (label).  

Một số loss phổ biến:

- Regression: `nn.MSELoss`
- Classification (multi-class): `nn.CrossEntropyLoss`  
- Binary classification: `nn.BCEWithLogitsLoss`  

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y_true)
```

**Ref docs:**

- PyTorch `torch.nn` → mục “Loss Functions”
- PyTorch Tutorials → “Training a Classifier”

---

## 4. Backward Pass (Backprop) & Autograd

**Backward pass / backprop** = tự động tính gradient của loss với từng parameter bằng **autograd**.

Trong PyTorch:

```python
loss.backward()  # compute d(loss)/d(param) for all params with requires_grad=True
```

PyTorch sẽ:

1. Ghi lại tất cả operation trong forward thành một **computation graph**.
2. Khi gọi `backward()`, nó duyệt ngược graph và dùng **chain rule** để tính gradient.

Nếu cần gradient w.r.t inputs cụ thể, có thể dùng `torch.autograd.grad`.

**Ref docs:**

- PyTorch Docs → “Autograd mechanics”
- PyTorch Tutorials → “Autograd: Automatic Differentiation”

---

## 5. Gradient, Gradient Descent, Optimizer

Sau khi `loss.backward()` xong, mỗi parameter `p` sẽ có `p.grad`.  

**Optimizer** (SGD, Adam, …) sẽ dùng gradient này để cập nhật weights:

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# in training loop
loss.backward()
optimizer.step()        # update weights
optimizer.zero_grad()   # clear old gradients
```

- **Gradient descent** (và các biến thể mini-batch, SGD) là thuật toán tối ưu để giảm loss bằng cách đi ngược hướng gradient.

**Ref docs:**

- PyTorch Docs → `torch.optim`
- PyTorch Tutorials → “Optimization Loop” / “What is torch.optim?”

---

## 6. Learning Rate, Batch Size, Epoch

Ba hyperparameter cực core:

- **Learning rate (lr)**  
  Bước nhảy mỗi lần update:  
  \[
  w_{	ext{new}} = w_{	ext{old}} - \eta \cdot rac{\partial L}{\partial w}
  \]  
  LR lớn → học nhanh nhưng dễ “nổ”; LR nhỏ → ổn định nhưng chậm.

- **Batch size**  
  Số mẫu trong 1 mini-batch để tính loss và gradient.  
  Liên quan trực tiếp tới biến thể mini-batch gradient descent.

- **Epoch**  
  1 epoch = 1 lần duyệt hết toàn bộ tập train qua các mini-batch.

**Ref docs / blog:**

- Từ khóa để tra thêm:  
  - “learning rate and batch size relationship”  
  - “epoch vs batch size vs iteration”

---

## 7. Activation Functions

Activation function tạo **phi tuyến** giúp mạng học quan hệ phức tạp.

Một số thường dùng:

- `nn.ReLU`, `nn.LeakyReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Softmax` (thường dùng ở cuối cho phân loại, hoặc dùng implicit qua `CrossEntropyLoss`)

Ví dụ:

```python
self.net = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

Tất cả đều nằm trong `torch.nn` hoặc `torch.nn.functional` (mục Non-linear Activations).

**Ref docs:**

- PyTorch Docs → “Non-linear Activations (weighted sum, non-linearity)” trong `torch.nn`

---

## 8. Metrics (Accuracy, MAE, ...)

**Loss** dùng để train, **metric** dùng để đánh giá.

- Classification: **accuracy**, precision, recall, F1, AUC…
- Regression: **MAE**, RMSE, R²…

Ví dụ (simple accuracy cho classification):

```python
with torch.no_grad():
    logits = model(x_val)
    preds = logits.argmax(dim=1)
    acc = (preds == y_val).float().mean().item()
```

Thường dùng lib ngoài:

- `torchmetrics`
- `sklearn.metrics`

**Ref docs:**

- `sklearn.metrics` (Scikit-learn docs)
- `torchmetrics` docs

---

## 9. Train / Val / Test Split

Chia dữ liệu thành 3 phần:

- **Train**: dùng để model học (update weights).
- **Validation (val)**: dùng để chọn hyperparameter, early stopping, theo dõi overfitting.
- **Test**: dùng một lần cuối cùng để report kết quả (không đụng vào khi tune).

Thực hiện bằng:

- `sklearn.model_selection.train_test_split`
- Hoặc tự chia, rồi wrap thành `torch.utils.data.Dataset` + `DataLoader`.

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
```

**Ref docs:**

- Scikit-learn Docs → `model_selection.train_test_split`
- PyTorch Docs → `torch.utils.data.Dataset`, `DataLoader`

---

## 10. Computation Graph & Autograd (PyTorch View)

Trong PyTorch:

- Mọi operation trên tensor có `requires_grad=True` sẽ tạo nên **computation graph** (DAG).  
- Các node lá (leaf) thường là parameters / input.  
- Các node root là output / loss.  
- `loss.backward()` = autograd duyệt graph ngược lại, áp dụng **reverse-mode automatic differentiation** (backprop).

**Ref docs:**

- PyTorch Docs → “Autograd mechanics”
- PyTorch Tutorials → “Autograd: Automatic Differentiation”

---

## 11. Big Picture: Everything Ties Back to the Training Loop

Tóm gọn:

1. **Model + Parameters** (`nn.Module`, `nn.Parameter`)
2. **Forward pass** → output
3. **Loss function** → scalar loss
4. **Backward pass (autograd)** → gradients
5. **Optimizer (gradient descent variants)** → update parameters
6. Lặp lại qua nhiều **epoch**, với **mini-batch**, dùng **activation functions**, **metrics** và **train/val/test split** để đảm bảo model **generalize** tốt.

Nếu bạn hiểu kỹ từng mục ở trên, bạn đã nắm **core của Deep Learning**; sau đó chỉ cần thay **kiến trúc model** (CNN, RNN, Transformer, …) là làm được NLP, CV, tabular, v.v.
