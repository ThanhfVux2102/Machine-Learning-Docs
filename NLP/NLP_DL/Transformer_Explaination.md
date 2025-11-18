# 📘 Chi Tiết Từng Step + Cách Chọn Tham Số + Chọn Model Cho NLP Transformer Level 3

Tài liệu này giải thích rõ từng bước trong FLOW NLP Transformer Level 3, đồng thời hướng dẫn cách chọn mô hình, chọn tham số, và gợi ý setup cho từng dạng bài.

---

# 🟦 STEP 1 — Chuẩn Bị Dữ Liệu (Dataset Preparation)

### ✔ Mục đích  
Tạo nguồn dữ liệu sạch và đúng format, nhưng **không xử lý quá nhiều**, vì Transformer tự hiểu ngữ cảnh.

### ✔ Chỉ cần làm:
- xóa ký tự lỗi  
- normalize spacing  
- bỏ tag HTML  
- loại emoji nếu bài toán không cần  

### ✔ Không cần làm:
- bỏ stopwords  
- stemming  
- lemmatization  
- lower-case bắt buộc (một số model phân biệt hoa–thường)  

### 🔥 Tips:  
Nếu dữ liệu là social media → cần lọc bớt ký tự spam như: @#$%^&*, URL.

---

# 🟩 STEP 2 — Train / Validation / Test Split

### ✔ Tỉ lệ chuẩn
- **70 / 15 / 15** (dữ liệu lớn)  
- **80 / 10 / 10** (dữ liệu nhỏ)  

### ✔ Quan trọng: Stratify
Nếu là classification → luôn stratify theo label.

---

# 🟦 STEP 3 — Tokenization (HuggingFace)

Transformer dùng **subword tokenization**, ví dụ “playing” → “play” + “##ing”.

### ✔ Cách chọn max_length:
- 64 → comment, tweet  
- 128 → review, mô tả ngắn  
- 256 → báo, mô tả dài  
- 512 → tài liệu, report  

### 🔥 Lưu ý:  
max_length càng lớn → RAM càng tốn → training chậm hơn.

---

# 🟨 STEP 4 — Tạo Dataset PyTorch

Dataset phải trả về:
- input_ids  
- attention_mask  
- labels  

Với NER → trả về labels theo từng token (sequence labeling).

---

# 🟧 STEP 5 — Load Model Transformer

### ✔ Khi nào chọn BERT-base?
- bài toán tiếng Anh  
- dữ liệu trung bình  
- accuracy ổn định  
- mô hình phổ thông  

### ✔ Khi nào chọn DistilBERT?
- máy yếu  
- muốn tốc độ nhanh  
- xây app mobile/web  
- inference realtime  

### ✔ Khi nào chọn RoBERTa?
- cần accuracy cao  
- dữ liệu phức tạp  
- GPU mạnh hơn  

### ✔ Khi nào chọn XLM-R?
- dữ liệu tiếng Việt  
- multilingual  
- văn bản lai nhiều ngôn ngữ  

---

# 🟦 STEP 6 — Fine-tuning Model

### ✔ Chọn Learning Rate
- **2e-5** → chuẩn nhất  
- **3e-5** → nhanh hơn, dễ overfit  
- **5e-5** → chỉ dùng cho model nhỏ như DistilBERT  

### ✔ Batch size
- 16 → GPU 8GB  
- 32 → GPU 12GB  
- 64 → GPU 24GB+  

### ✔ Epoch
- 2 → khi dataset lớn  
- 3 → chuẩn  
- 4 → khi dataset nhỏ  

### ✔ Warmup ratio
- 10% số step → giúp training mượt hơn

### ✔ Weight Decay
- 0.01 → chuẩn cho Transformer  

### 🔥 Mẹo quan trọng:
RoBERTa thường cần nhiều epoch hơn BERT.

---

# 🟦 STEP 7 — Model Evaluation

### Classification
- Accuracy  
- Precision / Recall  
- Macro-F1 **(quan trọng nhất nếu mất cân bằng)**  
- Confusion Matrix  

### NER
- F1 theo entity-level  

### Sentence Similarity
- Cosine similarity  
- Spearman correlation  

---

# 🟦 STEP 8 — Inference Pipeline

### ✔ Mục tiêu  
Tạo hàm dự đoán đơn giản, nhận input → trả label.

### ✔ Lưu ý:
- Luôn tokenize với padding/truncation  
- Transformer trả logits → softmax → label  

---

# 🟦 STEP 9 — Save Model

Bao gồm:
```
config.json  
pytorch_model.bin  
tokenizer.json  
special_tokens_map.json  
```

### ✔ Khi cần ONNX?
- chạy CPU  
- chạy real-time  
- deploy mobile  

---

# 🟦 STEP 10 — Deploy API

### Framework:
- **FastAPI** → production  
- **Streamlit/Gradio** → demo nhanh  

### ✔ Có cần GPU server không?
Chỉ khi:
- model > 200M params  
- tốc độ dự đoán > 50 req/s  

---

# 🟦 STEP 11 — README / Report

Nội dung chuẩn:
- mô tả task  
- mô tả dataset  
- EDA  
- lý do chọn mô hình  
- hyperparameters  
- training logs  
- evaluation metrics  
- confusion matrix  
- error analysis  
- future work  

---

# 📌 CÁCH CHỌN MODEL THEO BÀI TOÁN

### ✔ Sentiment Analysis
- DistilBERT → nếu muốn nhanh  
- BERT-base → stable  
- RoBERTa → accuracy cao nhất  

### ✔ Hate Speech / Toxic Comments
- RoBERTa  
- XLM-R (nếu có tiếng Việt)  

### ✔ Text Classification (news/product review)
- BERT-base  
- XLM-R (multilingual)  

### ✔ NER
- BERT-base  
- XLM-R → tốt cho tiếng Việt  

### ✔ Semantic Similarity
- SBERT  
- Siamese-BERT  

---

# 📌 CÁCH CHỌN THAM SỐ NHANH (CHEAT-SHEET)

| Thành phần | Giá trị chuẩn |
|-----------|---------------|
| Learning rate | **2e-5** |
| Batch size | **16** |
| Epoch | **3** |
| Max length | **128** |
| Warmup | **10% steps** |
| Weight decay | **0.01** |

---


