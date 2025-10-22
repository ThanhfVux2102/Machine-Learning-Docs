# 🧹 Text Preprocessing Flow with NLTK

This document describes a **standard preprocessing flow** for text data
using Python and NLTK.

------------------------------------------------------------------------

## 🔄 Step-by-Step Flow

### 1️⃣ Remove line breaks, tabs, and extra spaces

Use regex to normalize spacing:

``` python
import re
text = re.sub(r'\s+', ' ', text)
```

👉 Helps prevent tokenization errors.

------------------------------------------------------------------------

### 2️⃣ Convert to lowercase

``` python
text = text.lower()
```

👉 Normalizes words (e.g., "Apple" and "apple" are treated as the same).

------------------------------------------------------------------------

### 3️⃣ Remove emojis, special characters, numbers, and punctuation

Use regex and emoji filtering:

``` python
# Remove punctuation and special characters
text = re.sub(r'[^\w\s]', '', text)

# Remove emoji
import re
emoji_pattern = re.compile("["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags
    "]+", flags=re.UNICODE)
text = emoji_pattern.sub(r'', text)
```

------------------------------------------------------------------------

### 4️⃣ Tokenize (split text into words)

``` python
from nltk.tokenize import word_tokenize
tokens = word_tokenize(text)
```

------------------------------------------------------------------------

### 5️⃣ Remove stopwords

``` python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if w not in stop_words]
```

------------------------------------------------------------------------

### 6️⃣ Apply stemming or lemmatization

Choose **one**, not both.

#### 🔹 Stemming

``` python
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
tokens = [stemmer.stem(w) for w in tokens]
```

#### 🔹 Lemmatization

``` python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(w) for w in tokens]
```

------------------------------------------------------------------------

## ✅ Summary of Optimal Flow

1.  Normalize whitespace and remove line breaks\
2.  Lowercase text\
3.  Remove emojis and special characters\
4.  Tokenize text\
5.  Remove stopwords\
6.  Apply stemming or lemmatization

------------------------------------------------------------------------

## 📘 Notes

-   Use `nltk.download('punkt')`, `nltk.download('stopwords')`, and
    `nltk.download('wordnet')` before running the preprocessing code.
-   Avoid doing both stemming and lemmatization simultaneously.
-   This pipeline works best for English text.
