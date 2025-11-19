# Core Deep Learning Training Loop (with PyTorch)

This note summarizes the core concepts that appear in **every** deep learning training loop, especially when using **PyTorch**.  
Ở từng bước mình đều ghi rõ **Goal (mục tiêu)** để bạn nắm rõ “bước này để làm gì”.

---

## 0. Minimal Training Loop (PyTorch)

```python
for epoch in range(num_epochs):
    for x_batch, y_batch in dataloader:
        # 0. (optional) Clear old gradients
        optimizer.zero_grad()

        # 1. Forward pass
        y_pred = model(x_batch)

        # 2. Compute loss
        loss = loss_fn(y_pred, y_batch)

        # 3. Backward pass (backprop)
        loss.backward()

        # 4. Update parameters
        optimizer.step()
```

### Goal of each step

- **optimizer.zero_grad()**  
  → Goal: Xoá toàn bộ gradient cũ (`param.grad`) từ batch trước, để batch hiện tại **không bị cộng dồn nhầm**.

- **Forward pass (`y_pred = model(x_batch)`)**  
  → Goal: Tính **output (prediction/logits)** của model với batch input hiện tại và **xây computation graph** để sau này autograd dùng cho backward.

- **Loss computation (`loss = loss_fn(y_pred, y_batch)`)**  
  → Goal: Biến sự khác biệt giữa **dự đoán** và **nhãn thật** thành **1 số scalar** (loss) để tối ưu (minimize).

- **Backward pass (`loss.backward()`)**  
  → Goal: Dùng autograd để tính **gradient của loss theo từng parameter** (`∂loss/∂w`) và lưu vào `param.grad`.

- **Optimizer step (`optimizer.step()`)**  
  → Goal: Dùng `param.grad` để **cập nhật lại weights** (gradient descent / Adam / …), giúp model dự đoán tốt hơn ở lần sau.

---

## 1. Model & Parameters (Weights, Bias)

- **Goal:** Định nghĩa **hàm f(x; θ)** – tức là kiến trúc mạng (layers, connections, activation, …) và tập tham số **θ = {weights, biases}** mà ta sẽ *học* trong quá trình train.

- **Model** trong PyTorch thường là class kế thừa `torch.nn.Module`.
- **Parameters** là các tensor kiểu `torch.nn.Parameter` được model đăng ký tự động và sẽ được optimizer cập nhật.

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

**Docs / articles:**

- `torch.nn` overview  
  https://pytorch.org/docs/stable/nn.html
- `torch.nn.Module` API  
  https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Note “Modules — PyTorch”  
  https://pytorch.org/docs/stable/notes/modules.html
- Tutorial “Build the Neural Network”  
  https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

---

## 2. Forward Pass

- **Goal:**  
  - Tính **output** của model (logits / probability / prediction) từ input.  
  - Đồng thời để autograd **xây computation graph** (mỗi phép toán được ghi lại) cho bước backward.

Trong PyTorch, forward được hiện thực bằng hàm `forward`, còn gọi bằng `model(x)`:

```python
y_pred = model(x_batch)   # forward pass
```

**Docs:**

- Tutorial “Build the Neural Network”  
  https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- Tutorial “Neural Networks”  
  https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

---

## 3. Loss Function

- **Goal:** Đo **mức độ sai** giữa dự đoán của model và nhãn thật, gom lại thành **một số scalar** để dùng cho tối ưu (minimize).

Một số loss phổ biến:

- Regression: `nn.MSELoss`
- Classification (multi-class): `nn.CrossEntropyLoss`  
- Binary classification: `nn.BCEWithLogitsLoss`  

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(y_pred, y_true)
```

**Docs:**

- Loss Functions in `torch.nn`  
  https://pytorch.org/docs/stable/nn.html#loss-functions
- Tutorial “Training a Classifier”  
  https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

---

## 4. Backward Pass (Backprop) & Autograd

- **Goal backward:**  
  - Tính **gradient của loss theo từng parameter** (`∂loss/∂θ`) bằng chain rule.  
  - Lưu gradient vào `param.grad` để optimizer dùng.

- **Goal autograd:**  
  - Tự động hoá việc tính đạo hàm, không phải code backprop bằng tay.  
  - Quản lý **computation graph** và thứ tự nhân chain rule.

Trong PyTorch:

```python
loss.backward()  # compute d(loss)/d(param) for all params with requires_grad=True
```

Autograd sẽ:

1. Ghi lại operation trong forward thành **computation graph**.
2. Duyệt ngược graph, dùng chain rule để tính gradient và cộng vào `param.grad`.

**Docs:**

- Autograd package  
  https://pytorch.org/docs/stable/autograd.html
- Tutorial “Automatic differentiation with torch.autograd”  
  https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
- Tutorial “A Gentle Introduction to torch.autograd”  
  https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

---

## 5. Gradient, Optimizer, Gradient Descent

- **Goal gradient:**  
  - Cho biết nếu **tăng/giảm** một weight tí xíu thì **loss thay đổi thế nào** (hướng & độ lớn).

- **Goal optimizer:**  
  - Dùng gradient để **tìm bộ weights tốt hơn** (giảm loss), theo một thuật toán tối ưu cụ thể (SGD, Adam, …).

Ví dụ:

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# in training loop
optimizer.zero_grad()     # clear old gradients
y_pred = model(x_batch)
loss = loss_fn(y_pred, y_batch)
loss.backward()           # compute gradients
optimizer.step()          # update weights
```

- `optimizer.step()` thường làm kiểu:  
  \\( w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w} \\)
- `optimizer.zero_grad()` dọn `w.grad` về 0 sau khi đã dùng xong, chuẩn bị cho batch mới.

**Docs:**

- `torch.optim` docs  
  https://pytorch.org/docs/stable/optim.html
- Tutorial “Optimizing Model Parameters”  
  https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- Adam optimizer  
  https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

---

## 6. Learning Rate, Batch Size, Epoch

- **Goal learning rate (lr):**  
  - Điều chỉnh **bước nhảy** khi cập nhật weights. LR quá lớn → dễ “nổ”; quá nhỏ → học chậm.

- **Goal batch size:**  
  - Quyết định **mức độ nhiễu** của gradient (batch nhỏ → noisy, batch lớn → ổn định hơn) và lượng memory dùng.

- **Goal epoch:**  
  - Đo số lần model **quét hết tập train**; thường dùng để lên lịch learning rate, early stopping…

**Docs / bài đọc:**

- “Optimizing Model Parameters” (giải thích iter / epoch / batch)  
  https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- RealPython: train/test split & cơ bản training loop  
  https://realpython.com/train-test-split-python-data/

---

## 7. Activation Functions

- **Goal:** Tạo **phi tuyến** trong mạng; nếu không có activation phi tuyến, nhiều layer Linear chồng nhau vẫn chỉ tương đương 1 Linear duy nhất.

Các activation phổ biến:

- `nn.ReLU`, `nn.LeakyReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Softmax` (thường dùng ở cuối classification)

Ví dụ:

```python
self.net = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
```

**Docs:**

- Non-linear Activations in `torch.nn`  
  https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity

---

## 8. Metrics (Accuracy, MAE, ...)

- **Goal:** Cung cấp **thước đo dễ hiểu** để đánh giá model (ngoài loss), ví dụ: accuracy, F1, MAE,…

Loss dùng để **tối ưu**; metric dùng để **đọc hiểu kết quả** và so sánh model.

Ví dụ (simple accuracy):

```python
with torch.no_grad():
    logits = model(x_val)
    preds = logits.argmax(dim=1)
    acc = (preds == y_val).float().mean().item()
```

**Docs:**

- Scikit-learn model evaluation  
  https://scikit-learn.org/stable/modules/model_evaluation.html
- TorchMetrics docs  
  https://lightning.ai/docs/torchmetrics/stable/index.html

---

## 9. Train / Val / Test Split

- **Goal:**  
  - **Train:** cho model học.  
  - **Validation:** theo dõi overfitting, chọn hyperparameter, chọn model.  
  - **Test:** đánh giá cuối cùng, không đụng vào khi tune.

Ví dụ với scikit-learn:

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

**Docs:**

- `train_test_split`  
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
- RealPython hướng dẫn chi tiết  
  https://realpython.com/train-test-split-python-data/

---

## 10. Computation Graph & Autograd (PyTorch View)

- **Goal computation graph:**  
  - Lưu lại **chuỗi phép toán** của forward để biết phải nhân chain rule như thế nào khi backward.

- **Goal autograd trên graph:**  
  - Từ `loss`, đi ngược graph → tính gradient của từng node (layer, operation) một cách **tự động**.

Trong PyTorch:

- Tensors với `requires_grad=True` được track trong graph.
- Mỗi operation sinh ra tensor mới kèm `grad_fn` (hàm dùng để tính gradient ngược).
- `loss.backward()` kích hoạt quá trình reverse-mode automatic differentiation.

**Docs:**

- Autograd docs  
  https://pytorch.org/docs/stable/autograd.html
- Autograd quickstart  
  https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

---

## 11. Big Picture: Mục tiêu của cả vòng lặp train

Nếu gom lại:

1. **Model + Parameters**  
   → Goal: Định nghĩa hàm f(x; θ) có thể học được.

2. **Forward pass**  
   → Goal: Tính output + build computation graph.

3. **Loss function**  
   → Goal: Đo sai số thành 1 số scalar để tối ưu.

4. **Backward pass (autograd)**  
   → Goal: Tính gradient của loss theo từng θ.

5. **Optimizer step**  
   → Goal: Cập nhật θ theo hướng giảm loss (gradient descent).

6. **Repeat over many epochs**  
   → Goal: Làm model càng ngày càng **generalize tốt** trên dữ liệu mới, theo dõi bằng **metrics** trên **val/test**.

Hiểu được **Goal của từng step** trong file này là bạn đã nắm rất rõ **core training loop của Deep Learning**. Sau đó chỉ cần thay **kiến trúc model** (CNN, RNN, Transformer, …) là áp dụng được cho CV, NLP, tabular, time series, v.v.
