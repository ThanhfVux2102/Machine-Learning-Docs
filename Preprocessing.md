# 📘 ML Preprocessing + Feature Engineering Cheatsheet

This guide combines **data preprocessing steps by type** and **general feature engineering strategies** into one reference.

---

## 🧹 Preprocessing Cheatsheet by Data Type

| 📂 Data Type | ⚙️ Main Preprocessing Steps | 🛠️ Common Methods / Functions | 🏗️ Frameworks | 🔍 Recognition Trick |
|--------------|-----------------------------|-------------------------------|---------------|-----------------------|
| 🖼️ Image | 1. Resize<br>2. Normalize pixel values<br>3. Data Augmentation<br>4. Encode labels | `tf.image.resize()`<br>`tf.cast(img, tf.float32)/255`<br>`ImageDataGenerator`<br>`tf.image.random_flip_left_right`<br>`LabelEncoder`, `to_categorical()` | **TensorFlow/Keras**, **PyTorch** (`torchvision`, `albumentations`), **OpenCV**, **YOLO (Ultralytics)**, Detectron2, MMDetection | `.jpg`, `.png`, shape `(H, W, C)` |
| 📄 Text | 1. Lowercase<br>2. Remove punctuation / stopwords<br>3. Tokenize<br>4. Padding<br>5. Vectorization (TF-IDF / BERT) | `.lower()`, `re.sub()`<br>`nltk.corpus.stopwords`, `re`<br>`Tokenizer().fit_on_texts()`<br>`pad_sequences()`<br>`TfidfVectorizer`, `BERTTokenizer` | **Hugging Face Transformers**, spaCy, NLTK, TensorFlow/Keras (`TextVectorization`) | Files like `.txt`, `.csv` containing sentences or paragraphs |
| 📊 Tabular (pandas) | 1. Handle missing values<br>2. Encode categorical variables<br>3. Scale numerical features<br>4. Feature selection<br>5. Merge data | `SimpleImputer`, `dropna()`<br>`OneHotEncoder`, `LabelEncoder`, `get_dummies()`<br>`StandardScaler`, `MinMaxScaler`<br>`SelectKBest` | **scikit-learn**, pandas, XGBoost/LightGBM/CatBoost | CSV/XLS with multiple columns and data types |
| 📈 Time-Series | 1. Parse timestamp<br>2. Resampling<br>3. Normalize<br>4. Sliding Window | `pd.to_datetime()`<br>`df.resample('D').mean()`<br>`StandardScaler`, `MinMaxScaler`<br>`sliding_window_view` | **sktime**, **darts**, prophet, statsmodels, pandas | Has `"timestamp"` column, time-sequenced data |
| 🔊 Audio | 1. Feature extraction (MFCC, Spectrogram)<br>2. Normalize<br>3. Noise reduction<br>4. Sampling | `librosa.feature.mfcc()`<br>`librosa.amplitude_to_db()`<br>`scipy.signal`<br>`noisereduce.reduce_noise()`<br>`librosa.resample()` | **librosa**, **torchaudio (PyTorch)**, SpeechBrain, ESPnet, OpenAI Whisper | `.wav`, `.mp3` files or waveform arrays |
| 🎞️ Video | 1. Extract frames<br>2. Resize frames<br>3. Normalize<br>4. Sampling | `cv2.VideoCapture().read()`<br>`cv2.resize()`<br>`frame/255.0`<br>Sampling by FPS or time | **OpenCV**, **PyTorchVideo**, torchvision, decord, moviepy, **YOLO + trackers (ByteTrack, StrongSORT)** | `.mp4`, `.avi` files or image sequences |

---

## 🏗️ General Pipeline Steps (All Data Types)

| ✅ Step                | ✅ Purpose / Description                                                                 |
|------------------------|-------------------------------------------------------------------------------------------|
| **1. Load the dataset**   | Read/import raw data (disk, DB, web API, etc.)                                          |
| **2. Normalize the data** | Standardize or scale (pixel [0,1], tokenization, feature scaling, etc.)                 |
| **3. Shuffle the data**   | Randomize order to reduce bias and prevent overfitting                                  |
| **4. Batch the data**     | Split into fixed-size groups for efficient training/inference                           |
| **5. Prefetch the data**  | Load batches ahead during training for faster throughput                                |

---

## 🧠 Feature Engineering Cheatsheet (Cross-Domain)

### 1. Domain-driven Features
Use domain knowledge to create meaningful features.  
- Real estate: `house_age`, `since_renovation`, `living_ratio`  
- Finance: `debt_to_income`, `savings_rate`  
- Healthcare: `BMI`, `blood_pressure_ratio`  
- E-commerce: `avg_order_value`, `days_since_last_purchase`

---

### 2. Transformations
Fix skewness & non-linear relationships.  
- Log transform: `log(price)`, `log(income)`  
- Square root: `sqrt(accident_counts)`  
- Polynomial: `age²`, `temperature²`

---

### 3. Encoding Categorical Data
Turn categories into usable numbers.  
- One-hot encoding: good for linear models.  
- Label encoding: simple integer map (OK for trees).  
- Frequency / target encoding: good for high-cardinality.  
- Group rare categories into “Other”.

---

### 4. Interaction Features
Combine features to capture hidden effects.  
- Real estate: `sqft_living × bathrooms`, `waterfront × view`  
- Retail: `discount × season`  
- Credit scoring: `age × income`

---

### 5. Aggregations & Binning
Summarize and bucket values.  
- Binning: age → `young/adult/senior`, price → `low/medium/high`  
- Aggregations: average price per zipcode, median income per region, customer purchase frequency

---

## 🚦 Workflow for Feature Engineering

1. **Understand raw features** (meaning, type, skew, distribution).  
2. **Brainstorm features** (domain knowledge + math).  
3. **Check correlations** (heatmap, scatterplots).  
4. **Iteratively test** (cross-validation, metrics).  
5. **Keep what helps, drop what doesn’t**.

---

## ✅ Rules of Thumb

- High correlation with target, low with others → **keep**  
- High correlation with target & another feature → **drop one** (linear models)  
- Low correlation with target & others → **candidate to drop** (test first)  
- Weak features can help in **non-linear models** → **don’t drop blindly**

---
