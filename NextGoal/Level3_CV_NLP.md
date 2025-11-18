# Level 3 Summary for NLP & Computer Vision (CV)

## 📘 NLP Level 3 — Transformer-based (BERT, DistilBERT, RoBERTa)

### 🔹 Mục tiêu của Level 3
- Xây dựng dự án **chuẩn AI-engineer** bằng cách sử dụng các mô hình **Transformer hiện đại**.
- Tạo một pipeline đầy đủ: preprocessing → fine-tuning → evaluation → inference → (optional) deploy.

---

## 🚀 Các kỹ thuật cốt lõi
### ✔ Transformer-based Models
- BERT-base
- DistilBERT
- RoBERTa-base
- XLM-RoBERTa (nếu cần đa ngôn ngữ)

### ✔ Kỹ năng cần có
- Tokenization bằng HuggingFace
- Fine-tuning mô hình với Trainer API hoặc PyTorch Lightning
- Attention Mask & Padding
- Evaluation chuyên nghiệp:
  - Accuracy
  - F1-score
  - Confusion Matrix
- Save/Load model
- Inference pipeline
- (Optional) Deploy bằng FastAPI hoặc Flask

---

## 🧪 Các loại project phù hợp Level 3
- Text Classification (Spam/Emotion/Intent)
- NER (Named Entity Recognition)
- Semantic Similarity (Siamese BERT)
- Topic Classification
- Question Answering (SQuAD-style)

---

# 📙 CV Level 3 — Transfer Learning CNN (EfficientNet, ResNet, VGG16)

### 🔹 Mục tiêu của Level 3
- Sử dụng **modern CNN architectures** để giải bài toán thị giác máy tính với hiệu suất mạnh mẽ.
- Tương đương “đẹp” như NLP Level 3.

---

## 🚀 Các kỹ thuật cốt lõi
### ✔ Transfer Learning
- ResNet50 / ResNet101
- EfficientNet-B0 → B5
- MobileNetV2 / V3
- DenseNet121
- VGG16 (ít dùng nhưng dễ hiểu)

### ✔ Kỹ năng cần có
- Loading pretrained weights
- Freezing & Unfreezing layers
- Custom head classifier
- Data Augmentation (Albumentations hoặc torchvision)
- Training + Validation loops
- Early stopping + model checkpoint
- Evaluation:
  - Accuracy
  - F1-score
  - Confusion Matrix
  - ROC/AUC nếu cần

---

## 🧪 Các loại project phù hợp Level 3
- Image Classification (Leaf disease, Animal species, Product defects)
- Face attributes classification (emotion, age group)
- Simple multi-class problems (food, fashion, vehicles)
- Landmark classification

---

# ⭐ Mục tiêu chung của Level 3 (NLP & CV)
| Yêu cầu | NLP Level 3 | CV Level 3 |
|--------|--------------|-------------|
| Deep Learning | ✔ Transformer | ✔ CNN (Transfer Learning) |
| Có mô hình industry | ✔ BERT | ✔ ResNet/EfficientNet |
| Evaluation chuẩn | ✔ F1, CM | ✔ F1, CM |
| Data thật/phức tạp | ✔ | ✔ |
| Deploy (optional) | FastAPI | FastAPI/Streamlit |
| Đẹp trong portfolio | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

# 🎯 Tóm tắt cuối

- NLP: BERT → đẹp, mạnh, industry standard  
- CV: ResNet/EfficientNet → chuyên nghiệp và dễ triển khai  

