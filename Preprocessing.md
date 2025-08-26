## 📘 Preprocessing Cheatsheet by Data Type

# Preprocessing Cheatsheet by Data Type

| 📂 Data Type | ⚙️ Main Preprocessing Steps | 🛠️ Common Methods / Functions | 🏗️ Frameworks | 🔍 Recognition Trick |
|--------------|-----------------------------|-------------------------------|---------------|-----------------------|
| 🖼️ Image | 1. Resize<br>2. Normalize pixel values<br>3. Data Augmentation<br>4. Encode labels | `tf.image.resize()`<br>`tf.cast(img, tf.float32)/255`<br>`ImageDataGenerator`<br>`tf.image.random_flip_left_right`<br>`LabelEncoder`, `to_categorical()` | **TensorFlow/Keras**, **PyTorch** (`torchvision`, `albumentations`), OpenCV | `.jpg`, `.png`, shape `(H, W, C)` |
| 📄 Text | 1. Lowercase<br>2. Remove punctuation / stopwords<br>3. Tokenize<br>4. Padding<br>5. Vectorization (TF-IDF / BERT) | `.lower()`, `re.sub()`<br>`nltk.corpus.stopwords`, `re`<br>`Tokenizer().fit_on_texts()`<br>`pad_sequences()`<br>`TfidfVectorizer`, `BERTTokenizer` | **Hugging Face Transformers**, spaCy, NLTK, TensorFlow/Keras (`TextVectorization`) | Files like `.txt`, `.csv` containing sentences or paragraphs |
| 📊 Tabular (pandas) | 1. Handle missing values<br>2. Encode categorical variables<br>3. Scale numerical features<br>4. Feature selection<br>5. Merge data | `SimpleImputer`, `dropna()`<br>`OneHotEncoder`, `LabelEncoder`, `get_dummies()`<br>`StandardScaler`, `MinMaxScaler`<br>`SelectKBest` | **scikit-learn**, pandas, XGBoost/LightGBM/CatBoost | CSV/XLS with multiple columns and data types |
| 📈 Time-Series | 1. Parse timestamp<br>2. Resampling<br>3. Normalize<br>4. Sliding Window | `pd.to_datetime()`<br>`df.resample('D').mean()`<br>`StandardScaler`, `MinMaxScaler`<br>`sliding_window_view` | **sktime**, **darts**, prophet, statsmodels, pandas | Has `"timestamp"` column, time-sequenced data |
| 🔊 Audio | 1. Feature extraction (MFCC, Spectrogram)<br>2. Normalize<br>3. Noise reduction<br>4. Sampling | `librosa.feature.mfcc()`<br>`librosa.amplitude_to_db()`<br>`scipy.signal`<br>`noisereduce.reduce_noise()`<br>`librosa.resample()` | **librosa**, **torchaudio (PyTorch)**, SpeechBrain, ESPnet, OpenAI Whisper | `.wav`, `.mp3` files or waveform arrays |
| 🎞️ Video | 1. Extract frames<br>2. Resize frames<br>3. Normalize<br>4. Sampling | `cv2.VideoCapture().read()`<br>`cv2.resize()`<br>`frame/255.0`<br>Sampling by FPS or time | **OpenCV**, **PyTorchVideo**, torchvision, decord, moviepy | `.mp4`, `.avi` files or image sequences |



## Goal for pipeline for every Data Type :

| ✅ **Step**                | ✅ **Purpose / Description**                                                                 |
|---------------------------|----------------------------------------------------------------------------------------------|
| **1. Load the dataset**   | Read and import the raw data into the pipeline (from disk, database, web, etc.)              |
| **2. Normalize the data** | Standardize or scale the data (e.g. pixel values to [0, 1], text tokenization, feature scaling) |
| **3. Shuffle the data**   | Randomize the order of data to prevent model overfitting or learning from sequence bias      |
| **4. Batch the data**     | Divide the data into fixed-size groups (batches) to enable efficient training/inference      |
| **5. Prefetch the data**  | Load batches ahead of time during training to improve pipeline throughput and efficiency     |


