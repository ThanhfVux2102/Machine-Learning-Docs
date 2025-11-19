# 🔥 Flow Chuẩn NLP Deep Learning với Transformer (Level 3)

Đây là **flow chuẩn industry** text → tokenizer → encodings (input_ids, attention_mask, …) → Dataset → DataLoader/Trainer → model (BERT, …)

---

# **STEP 1 — Chuẩn bị dữ liệu (Dataset Preparation)**

### ✔ Thu thập / tạo dataset

- Định dạng: CSV / JSON / TXT
- Phải có: `text`, `label`

### ✔ Tiền xử lý nhẹ

Không cần làm các bước NLP truyền thống:

- stopword removal
- stemming
- lemmatization
- bỏ dấu tiếng Việt

Chỉ cần:

- loại ký tự lỗi
- chuẩn hóa khoảng trắng
- bỏ emoji / HTML nếu không cần

**Transformers hoạt động tốt với dữ liệu gần-thô.**

---

# **STEP 2 — Train / Validation / Test Split**

Tỉ lệ đề xuất:

- Train: **70%**
- Validation: **15%**
- Test: **15%**

Nếu dataset mất cân bằng → sử dụng stratify.

---

# **STEP 3 — Tokenization bằng HuggingFace**

Tokenize đúng kiểu của model:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tok = tokenizer(
    "I love NLP",
    padding="max_length",
    truncation=True,
    max_length=128
)
```

Tokenizer tạo ra:

- `input_ids`
- `attention_mask`
- `token_type_ids` (chỉ BERT)

---

# **STEP 4 — Tạo Dataset cho PyTorch**

```python
class MyDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
```

---

# **STEP 5 — Load mô hình Transformer**

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_CLASSES
)
```

Có thể thay:

- `distilbert-base-uncased`
- `roberta-base`
- `xlm-roberta-base`

---

# **STEP 6 — Fine-tuning với Trainer API**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
trainer.train()
```

### ⚙ Hyperparameters chuẩn:

- Learning rate: **2e-5 hoặc 3e-5**
- Epoch: **2–4**
- Batch size: **16–32**

---

# **STEP 7 — Evaluate**

Dùng các metric:

- Accuracy
- Precision / Recall / F1
- Macro-F1 (nếu mất cân bằng)
- Confusion Matrix

---

# **STEP 8 — Inference Pipeline**

```python
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return torch.argmax(probs).item()
```

---

# **STEP 9 — Save mô hình**

```python
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
```

---

# **STEP 10 — Deploy API (optional)**

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/predict")
def classify(text: str):
    return {"label": predict(text)}
```

UI có thể build bằng:

- Streamlit
- Gradio
- Next.js

---

# **STEP 11 — Viết README / Report**

Bao gồm:

- Giới thiệu bài toán
- Dataset
- Kiến trúc mô hình
- Hyperparameters
- Training logs
- Kết quả đánh giá
- Error analysis
- Future work

---

# 🎯 **TÓM TẮT NGẮN GỌN FLOW LEVEL 3**

1. Chuẩn bị dữ liệu
2. Chia train/val/test
3. Tokenization
4. Tạo dataset
5. Load model
6. Fine-tune
7. Evaluate
8. Inference
9. Save
10. Deploy (optional)


---


