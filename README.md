## 📘 Preprocessing Cheatsheet by Data Type

| 📂 Data Type | ⚙️ Main Preprocessing Steps | 🛠️ Common Methods / Functions | 🔍 Recognition Trick |
|-------------|-----------------------------|-------------------------------|-----------------------|
| 🖼️ Image | 1. Resize<br>2. Normalize pixel values<br>3. Data Augmentation<br>4. Encode labels | `tf.image.resize()`<br>`tf.cast(img, tf.float32)/255`<br>`ImageDataGenerator`,<br>`tf.image.random_flip_left_right`<br>`LabelEncoder`, `to_categorical()` | `.jpg`, `.png`, shape `(H, W, C)` |
| 📄 Text | 1. Lowercase<br>2. Remove punctuation / stopwords<br>3. Tokenize<br>4. Padding<br>5. Vectorization (TF-IDF / BERT) | `.lower()`, `re.sub()`<br>`nltk.corpus.stopwords`, `re`<br>`Tokenizer().fit_on_texts()`<br>`pad_sequences()`<br>`TfidfVectorizer`, `BERTTokenizer` | Files like `.txt`, `.csv` containing sentences or paragraphs |
| 📊 Tabular | 1. Handle missing values<br>2. Encode categorical variables<br>3. Scale numerical features<br>4. Feature selection | `SimpleImputer`, `dropna()`<br>`OneHotEncoder`, `LabelEncoder`, `get_dummies()`<br>`StandardScaler`, `MinMaxScaler`<br>`SelectKBest` | CSV/XLS with multiple columns and data types |
| 📈 Time-Series | 1. Parse timestamp<br>2. Resampling<br>3. Normalize<br>4. Sliding Window | `pd.to_datetime()`<br>`df.resample('D').mean()`<br>`StandardScaler`, `MinMaxScaler`<br>`sliding_window_view` | Has `"timestamp"` column, time-sequenced data |
| 🔊 Audio | 1. Feature extraction (MFCC, Spectrogram)<br>2. Normalize<br>3. Noise reduction<br>4. Sampling | `librosa.feature.mfcc()`<br>`librosa.amplitude_to_db()`<br>`scipy.signal`,<br>`noisereduce.reduce_noise()`<br>`librosa.resample()` | `.wav`, `.mp3` files or waveform arrays |
| 🎞️ Video | 1. Extract frames<br>2. Resize frames<br>3. Normalize<br>4. Sampling | `cv2.VideoCapture().read()`<br>`cv2.resize()`<br>`frame/255.0`<br>Sampling by FPS or time | `.mp4`, `.avi` files or image sequences |
