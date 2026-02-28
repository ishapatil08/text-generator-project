Project Overview

This project implements a Text Generator using a Recurrent Neural Network (LSTM).
The model learns character-level patterns from a given text dataset and generates new text automatically based on the learned patterns.

Text generation is a Natural Language Processing (NLP) task where a model predicts the next character (or word) in a sequence after being trained on sample text data.

🎯 Objectives

Load and preprocess text data

Convert characters into numerical format

Create input-output training sequences

Train an LSTM neural network

Generate new text using the trained model

🛠 Technologies Used

Python

NumPy

TensorFlow / Keras

LSTM (Recurrent Neural Network)

⚙️ Methodology
 Data Preprocessing

The input text is converted to lowercase and unique characters are identified.

 Character Mapping

Each character is mapped to a numerical value using dictionaries.

 Sequence Creation

Fixed-length input sequences are created along with corresponding target outputs.

 Model Building

An LSTM model is built using:

LSTM layer

Dense output layer with softmax activation

Model Training

The model is trained using categorical cross-entropy loss and Adam optimizer.

 Text Generation

After training, the model predicts characters sequentially to generate new text.

 Output

The trained model generates new text sequences based on learned patterns from the dataset.