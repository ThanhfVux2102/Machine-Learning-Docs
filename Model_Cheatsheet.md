
# 🧠 Machine Learning Model Comparison Cheat Sheet

So sánh các model Machine Learning theo **mục đích – kiểu dữ liệu – khi nào dùng – khi nào không dùng – độ phức tạp**.

---

# 1️⃣ Supervised Learning (Học có giám sát)

## 1.1 Bức tranh tổng quát

| Model | Mục đích chính | Kiểu đầu ra | Độ phức tạp |
|-------|----------------|-------------|-------------|
| **Linear Regression** | Dự đoán giá trị liên tục | Số thực | Thấp |
| **Logistic Regression** | Phân loại nhị phân / multi-class | Nhãn lớp | Thấp–TB |
| **Decision Tree** | Phân loại / hồi quy, mô hình if–else | Cả số & lớp | TB |
| **Random Forest** | Ensemble nhiều cây → giảm overfit | Cả số & lớp | Cao |
| **SVM (Linear)** | Ranh giới phân tách tối ưu | Nhãn lớp | TB–Cao |
| **KNN** | Phân loại theo “hàng xóm gần nhất” | Nhãn lớp | Thấp–TB |

---

## 1.2 Khi nào nên dùng / không nên dùng

### 🔹 Linear Regression
- **Dùng khi:** Quan hệ gần tuyến tính, cần mô hình dễ giải thích.  
- **Không dùng khi:** Dữ liệu phi tuyến mạnh, nhiều outlier.

### 🔹 Logistic Regression
- **Dùng khi:** Bài toán phân loại có ranh giới tương đối tuyến tính, cần xác suất.  
- **Không dùng khi:** Quan hệ phi tuyến phức tạp.

### 🔹 Decision Tree
- **Dùng khi:** Cần mô hình dễ hiểu, logic if–else.  
- **Không dùng khi:** Dataset nhỏ + nhiều nhiễu → dễ overfit.

### 🔹 Random Forest
- **Dùng khi:** Cần độ chính xác cao, dữ liệu tabular nhiều feature.  
- **Không dùng khi:** Cần mô hình cực nhanh/nhẹ hoặc cần giải thích sâu.

### 🔹 SVM (Linear SVM)
- **Dùng khi:** Dữ liệu nhiều chiều (như TF-IDF), phân tách gần tuyến tính.  
- **Không dùng khi:** Dataset quá lớn hoặc nhiều lớp phức tạp.

### 🔹 KNN
- **Dùng khi:** Dataset nhỏ, làm baseline nhanh.  
- **Không dùng khi:** Dữ liệu nhiều chiều (hiệu ứng “lời nguyền chiều”).

---

# 2️⃣ Unsupervised Learning (Học không giám sát)

| Model | Mục đích chính | Kiểu dữ liệu phù hợp | Độ phức tạp |
|-------|----------------|----------------------|-------------|
| **K-Means** | Chia K cụm tương tự nhau | Dữ liệu số, cụm convex | Thấp–TB |
| **PCA** | Giảm chiều, visualization | Dữ liệu số có tương quan | TB |
| **Hierarchical Clustering** | Cụm phân cấp (dendrogram) | Dataset nhỏ–vừa | TB–Cao |

### Khi nên dùng:

#### 🔹 K-Means
- Khi muốn phân nhóm khách hàng, topic, hành vi người dùng.  
- Không phù hợp khi cụm méo mó hoặc nhiều outlier.

#### 🔹 PCA
- Khi feature quá nhiều → giảm chiều trước khi train model.  
- Không phù hợp khi cần giữ nguyên ý nghĩa từng feature gốc.

#### 🔹 Hierarchical Clustering
- Khi cần hiểu quan hệ phân cấp giữa các nhóm dữ liệu.  
- Không phù hợp với dataset lớn (O(n²) khoảng cách).

---

# 3️⃣ Deep Learning (Học sâu)

| Model | Mục đích chính | Dữ liệu phù hợp | Độ phức tạp |
|--------|----------------|-----------------|-------------|
| **CNN** | Xử lý ảnh/video | Ảnh, video | Rất cao |
| **RNN (LSTM/GRU)** | Dữ liệu chuỗi | Text, speech, time series | Rất cao |
| **GAN** | Sinh dữ liệu mới | Ảnh, âm thanh | Rất cao |

### Khi nên dùng:
- Dữ liệu rất lớn và phức tạp (ảnh, giọng nói, văn bản dài).  
- Khi các model truyền thống không đủ mạnh.

### Khi không nên dùng:
- Dataset nhỏ.  
- Cần giải thích rõ ràng từng feature.  
- Không có GPU hoặc thời gian train hạn chế.

---

# 4️⃣ Tóm tắt chọn model nhanh theo bài toán

- **Dự đoán giá trị số:** Linear Regression, Random Forest, XGBoost.  
- **Phân loại văn bản (TF-IDF):** Linear SVM, Logistic Regression, Naive Bayes.  
- **Phân nhóm dữ liệu:** K-Means / Hierarchical.  
- **Giảm chiều, visualize:** PCA.  
- **Ảnh/video:** CNN.  
- **Chuỗi thời gian, text:** RNN/LSTM/GRU hoặc Transformer.  
- **Sinh ảnh:** GAN.

---


