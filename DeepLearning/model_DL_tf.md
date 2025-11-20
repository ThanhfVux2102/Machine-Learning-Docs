
# Deep Learning Models in TensorFlow

## 1. Fully Connected Networks (FCN)

### Description
Fully Connected Networks (FCNs), also known as MLP (Multilayer Perceptrons), consist of layers where every neuron is connected to every other neuron in the next layer. These models are simple yet powerful for a variety of classification tasks.

### Sample Code

```python
import tensorflow as tf

class MLP(tf.keras.Model):
    def __init__(self, in_dim, num_classes):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.out = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.out(x)

# Initialize model
model = MLP(in_dim=20, num_classes=3)
```

### Advantages
- Simple to implement and train.
- Works well with structured/tabular data (e.g., CIC-IDS, credit scoring, etc.).

### Disadvantages
- Not effective for spatial (image) or sequential (text, time series) data.

### Suitable for
- **Tabular data**, **structured data**, **classification problems**.

---

## 2. Convolutional Neural Networks (CNN)

### Description
CNNs are specialized for processing grid-like data, especially images. They use convolutional layers to capture spatial hierarchies of features in an image.

### Sample Code

```python
class CNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.fc(x)

# Initialize model
model = CNN(num_classes=10)
```

### Advantages
- Excellent for image data processing and tasks that involve spatial relationships.
- Can automatically learn spatial hierarchies without manual feature extraction.

### Disadvantages
- Not suitable for sequential or tabular data.
- Requires large amounts of labeled data to achieve optimal performance.

### Suitable for
- **Image classification**, **object detection**, **segmentation**, **video analysis**.

---

## 3. Recurrent Neural Networks (RNN)

### Description
RNNs are designed for sequence data and are capable of maintaining hidden states that capture temporal dependencies between steps in a sequence.

### Sample Code

```python
class RNN_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(RNN_Model, self).__init__()
        self.rnn = tf.keras.layers.SimpleRNN(128, return_sequences=False)
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.rnn(x)
        return self.fc(x)

# Initialize model
model = RNN_Model(num_classes=5)
```

### Advantages
- Suitable for sequential data (e.g., time series, speech, text).
- Can model temporal dependencies and patterns over time.

### Disadvantages
- Struggles with long sequences due to vanishing gradients.
- Slower training compared to CNNs for large datasets.

### Suitable for
- **Time series forecasting**, **speech recognition**, **text processing**, **sequence generation**.

---

## 4. Long Short-Term Memory (LSTM)

### Description
LSTMs are a special kind of RNN that solves the vanishing gradient problem by using gates to control the flow of information through the network.

### Sample Code

```python
class LSTM_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(LSTM_Model, self).__init__()
        self.lstm = tf.keras.layers.LSTM(128)
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.lstm(x)
        return self.fc(x)

# Initialize model
model = LSTM_Model(num_classes=5)
```

### Advantages
- Can handle long-term dependencies in sequence data.
- More stable and capable of modeling complex sequences compared to vanilla RNNs.

### Disadvantages
- Requires more computation and memory than traditional RNNs.
- Still can struggle with very long sequences.

### Suitable for
- **Time series forecasting**, **text generation**, **machine translation**.

---

## 5. Gated Recurrent Unit (GRU)

### Description
GRUs are a variation of LSTMs that have fewer parameters and are simpler to train. They also solve the vanishing gradient problem by using gates to control the flow of information.

### Sample Code

```python
class GRU_Model(tf.keras.Model):
    def __init__(self, num_classes):
        super(GRU_Model, self).__init__()
        self.gru = tf.keras.layers.GRU(128)
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.gru(x)
        return self.fc(x)

# Initialize model
model = GRU_Model(num_classes=5)
```

### Advantages
- Similar performance to LSTMs but with fewer parameters and faster training.
- Suitable for sequence data.

### Disadvantages
- Less flexible than LSTM for modeling very long-term dependencies.
- Requires large amounts of data for effective training.

### Suitable for
- **Time series forecasting**, **speech recognition**, **sequence modeling**.

---

## 6. Transformer

### Description
The Transformer model is designed to handle sequential data with better efficiency than RNNs by using attention mechanisms to process entire sequences at once.

### Sample Code

```python
class Transformer(tf.keras.Model):
    def __init__(self, num_classes):
        super(Transformer, self).__init__()
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=128)
        self.fc = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        x = self.attention(x, x)  # self-attention
        return self.fc(x)

# Initialize model
model = Transformer(num_classes=5)
```

### Advantages
- Highly efficient for sequence-to-sequence tasks, like machine translation.
- Allows for parallel processing of data, reducing training time compared to RNNs.

### Disadvantages
- Requires large datasets to perform optimally.
- Can be computationally expensive due to the attention mechanism.

### Suitable for
- **NLP tasks** (e.g., machine translation, text generation), **time series forecasting**.

---

## 7. Autoencoders

### Description
Autoencoders are used for unsupervised learning. They learn to encode data in a compressed form and then reconstruct it. They are often used for anomaly detection and dimensionality reduction.

### Sample Code

```python
class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.layers.Dense(128, activation='relu')
        self.decoder = tf.keras.layers.Dense(784, activation='sigmoid')  # Example for image reconstruction

    def call(self, x):
        x = self.encoder(x)
        return self.decoder(x)

# Initialize model
model = Autoencoder()
```

### Advantages
- Great for unsupervised learning, especially for tasks like anomaly detection and dimensionality reduction.
- Can be used for denoising and data compression.

### Disadvantages
- May require extensive pre-processing to work effectively on complex data types like images and text.
- Limited ability to handle large datasets compared to supervised learning models.

### Suitable for
- **Anomaly detection**, **data compression**, **denoising**, **dimensionality reduction**.

---

## Summary

| Model Type              | Best For                              | Suitable Data Types            | Advantages                                    | Disadvantages                                  |
|-------------------------|---------------------------------------|---------------------------------|-----------------------------------------------|------------------------------------------------|
| **Fully Connected Networks (FCN)** | Simple classification tasks         | Tabular data                    | Easy to implement, fast training              | Not effective for spatial or sequential data   |
| **Convolutional Neural Networks (CNN)** | Image classification, Object detection | Image data                      | Excellent for spatial data, learns hierarchies | Requires large labeled data                   |
| **Recurrent Neural Networks (RNN)** | Sequence modeling, Time series       | Text, Speech, Time series        | Good for sequences, captures temporal dependencies | Struggles with long sequences                 |
| **Long Short-Term Memory (LSTM)** | Time series, Text generation         | Text, Time series               | Handles long-term dependencies better than RNN | More computationally expensive than RNN       |
| **Gated Recurrent Unit (GRU)** | Sequence modeling                    | Text, Time series               | Faster training, fewer parameters than LSTM   | Less flexible for very long sequences         |
| **Transformer**          | NLP tasks, Machine translation        | Text, Time series               | Efficient and scalable for long sequences     | Computationally expensive, requires large data |
| **Autoencoder**          | Unsupervised learning, Anomaly detection | Images, Text                    | Useful for compression, anomaly detection     | Requires clean and well-preprocessed data     |
