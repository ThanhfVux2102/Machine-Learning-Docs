# Deep Learning Workflow for Tabular Data with PyTorch

Mục tiêu: Dùng **PyTorch** để build một pipeline Deep Learning (MLP) cho dữ liệu tabular (ví dụ CIC-IDS, credit scoring, churn, employee performance…) và so sánh tư duy với ML truyền thống.

---

## 1. Tổng quan workflow

Cho một bài toán classification với data dạng bảng:

1. Hiểu & load data
   - Đọc file (CSV, Parquet…) bằng pandas.
   - Xem sơ bộ: số mẫu, số cột, kiểu dữ liệu, phân bố nhãn.

2. Tiền xử lý (preprocessing)
   - Xoá cột vô nghĩa (ID, timestamp nếu không dùng).
   - Xử lý giá trị thiếu (NaN, inf).
   - Tách features X và labels y.
   - Encode label (nếu là string).
   - Scale feature số (StandardScaler / MinMaxScaler).
   - Chia train / validation / test.

3. Đưa dữ liệu vào PyTorch
   - Chuyển X, y sang torch.Tensor.
   - Tạo lớp Dataset custom cho tabular.
   - Tạo DataLoader để batch + shuffle.

4. Định nghĩa model (MLP)
   - Kế thừa nn.Module.
   - Dùng nn.Linear, activation (ReLU), Dropout, v.v.

5. Khai báo loss + optimizer
   - Classification: nn.CrossEntropyLoss.
   - Optimizer: torch.optim.Adam (thường dùng).

6. Training loop
   - Cho từng epoch:
     - Train: loop qua train_loader, forward → loss → backward → optimizer.step().
     - Eval: loop qua val_loader, tính loss + accuracy để theo dõi overfitting.

7. Đánh giá trên test set
   - Load best checkpoint.
   - Tính test loss, accuracy và các metric khác (F1, ROC-AUC… nếu cần).

8. Lưu & deploy model
   - torch.save(model.state_dict(), "best_model.pth").
   - Khi deploy:
     - Load lại model + scaler + encoder.
     - Tiền xử lý input giống hệt lúc train.
     - Gọi model(x) để predict.

---

## 2. Chuẩn bị thư viện

Ví dụ code:

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

---

## 3. Load & tiền xử lý dữ liệu tabular

Ví dụ với file data.csv, cột nhãn tên là "label", có vài cột không dùng như "Flow ID", "Timestamp":

    # 1. Load data
    df = pd.read_csv("data.csv")

    print(df.head())
    print(df.info())
    print(df["label"].value_counts())  # phân bố nhãn

### 3.1. Xoá cột không cần thiết

    cols_drop = ["Flow ID", "Timestamp"]  # đổi theo dataset thật
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])

### 3.2. Xử lý missing / inf

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()  # đơn giản: bỏ các dòng có NaN (có thể dùng fillna nếu muốn)

### 3.3. Tách features và label

    X = df.drop(columns=["label"])  # đổi "label" theo tên cột nhãn
    y = df["label"]

### 3.4. Encode label

Nếu nhãn dạng string (e.g. "Normal", "Attack", "DoS"), dùng LabelEncoder:

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)  # np.array các số 0,1,2,...

### 3.5. Chia train / val / test

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

### 3.6. Scale features số

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

---

## 4. Dataset & DataLoader trong PyTorch

### 4.1. Lớp TabularDataset

    class TabularDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.long)  # classification

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

### 4.2. Tạo dataset & dataloader

    train_ds = TabularDataset(X_train, y_train)
    val_ds   = TabularDataset(X_val, y_val)
    test_ds  = TabularDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False)

DataLoader giúp:

- Tự động batching,
- Shuffle dữ liệu,
- Dễ dàng lặp qua batch trong training loop.

---

## 5. Định nghĩa MLP model với nn.Module

### 5.1. Thiết kế kiến trúc

    class MLP(nn.Module):
        def __init__(self, in_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes)
            )

        def forward(self, x):
            return self.net(x)

Khởi tạo model & chọn device:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_features = X_train.shape[1]
    num_classes  = len(np.unique(y_encoded))

    model = MLP(num_features, num_classes).to(device)

---

## 6. Loss function & optimizer

### 6.1. Chọn loss

Classification multi-class: dùng nn.CrossEntropyLoss.

    criterion = nn.CrossEntropyLoss()

### 6.2. Chọn optimizer

Dùng Adam:

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4  # L2 regularization nhẹ
    )

---

## 7. Training loop chi tiết

### 7.1. Hàm train một epoch

    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 1) Forward
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # 2) Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 3) Thống kê
            running_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        return epoch_loss, epoch_acc

### 7.2. Hàm evaluate

    @torch.no_grad()
    def eval_one_epoch(model, loader, criterion):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            running_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        epoch_loss = running_loss / total
        epoch_acc  = correct / total
        return epoch_loss, epoch_acc

### 7.3. Vòng lặp nhiều epoch + lưu best model

    num_epochs = 30
    best_val_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_mlp.pth")

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

---

## 8. Đánh giá trên test set

    model.load_state_dict(torch.load("best_mlp.pth"))
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion)
    print("Test loss:", test_loss)
    print("Test accuracy:", test_acc)

Nếu muốn F1-score, precision, recall, dùng thêm sklearn:

    from sklearn.metrics import classification_report

    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    print(classification_report(all_targets, all_preds))

---

## 9. Lưu & sử dụng lại model

Lưu:

    torch.save(model.state_dict(), "best_mlp.pth")

Load lại để infer:

    in_dim = num_features
    num_classes = len(np.unique(y_encoded))
    model = MLP(in_dim, num_classes)
    model.load_state_dict(torch.load("best_mlp.pth", map_location="cpu"))
    model.eval()

Khi dùng với input mới:

1. Áp dụng cùng scaler đã fit trước đó:
   - x_new_scaled = scaler.transform(x_new)
2. Chuyển sang tensor:
   - x_tensor = torch.tensor(x_new_scaled, dtype=torch.float32)
3. Gọi model:

    with torch.no_grad():
        logits = model(x_tensor)
        preds = logits.argmax(dim=1)

Nếu ban đầu label được encode bằng LabelEncoder, dùng le.inverse_transform(preds.numpy()) để map lại sang tên class gốc.

---

## 10. Tài liệu tham khảo (PyTorch docs)

- PyTorch documentation (trang tổng): https://pytorch.org/docs/stable/index.html
- Tensors: https://pytorch.org/docs/stable/tensors.html
- torch.nn (layers, loss…): https://pytorch.org/docs/stable/nn.html
- nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- nn.CrossEntropyLoss: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- DataLoader: https://pytorch.org/docs/stable/data.html
- Optimizers (torch.optim): https://pytorch.org/docs/stable/optim.html
