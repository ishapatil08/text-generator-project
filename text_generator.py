import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import os

# ---------------------------
# STEP 1: Load Text Dataset
# ---------------------------

text = """
Machine learning is fascinating.
Text generation using neural networks is powerful.
Deep learning allows computers to learn patterns in data.
"""

text = text.lower()

# ---------------------------
# STEP 2: Create Character Mapping
# ---------------------------

chars = sorted(list(set(text)))
char_to_int = {c: i for i, c in enumerate(chars)}
int_to_char = {i: c for i, c in enumerate(chars)}

# ---------------------------
# STEP 3: Prepare Dataset
# ---------------------------

seq_length = 40
X = []
y = []

for i in range(0, len(text) - seq_length):
    seq_in = text[i:i + seq_length]
    seq_out = text[i + seq_length]
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])

X = np.array(X)
y = to_categorical(y, num_classes=len(chars))

# Normalize input
X = X / float(len(chars))

# ---------------------------
# STEP 4: Build Model
# ---------------------------

model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], 1)))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Reshape for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ---------------------------
# STEP 5: Train Model
# ---------------------------

model.fit(X, y, epochs=20, batch_size=64)

# ---------------------------
# STEP 6: Generate Text
# ---------------------------

start_index = np.random.randint(0, len(X)-1)
pattern = X[start_index]
pattern = pattern.reshape(1, seq_length, 1)

generated_text = ""

for i in range(200):
    prediction = model.predict(pattern, verbose=0)
    index = np.argmax(prediction)
    result = int_to_char[index]
    generated_text += result

    pattern = np.append(pattern[:,1:,:], [[[index/float(len(chars))]]], axis=1)

print("\nGenerated Text:\n")
print(generated_text)
