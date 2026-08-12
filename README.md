Emotion Classification using Bidirectional GRU

A Deep Learning–based NLP model that classifies a user’s text into one of six emotions:

Sadness · Joy · Love · Anger · Fear · Surprise

The project uses the dair-ai/emotion dataset from Hugging Face and compares Simple RNN, LSTM, and GRU architectures before selecting a Bidirectional GRU for the final model.

Project Overview

Given a text input such as:

“I am extremely happy with my results today!”

the model predicts the corresponding emotion:

joy

The goal of the project is to understand how different recurrent neural network architectures perform for short-text emotion classification and build an efficient model capable of capturing contextual information from both directions of a sentence.

Dataset

The project uses the dair-ai/emotion dataset available on Hugging Face.

The dataset contains text samples belonging to six emotion categories:

Label	Emotion
0	Sadness
1	Joy
2	Love
3	Anger
4	Fear
5	Surprise

The dataset provides separate:

* Training set
* Validation set
* Test set

The original splits are preserved to avoid data leakage.

NLP Preprocessing

The text data is converted into numerical sequences before being passed to the neural network.

Tokenization

A Keras tokenizer is used with:

* Vocabulary size: 10,000 words
* OOV token: <OOV>
* Maximum sequence length: 50 tokens

Words outside the vocabulary are mapped to the OOV token.

Each text sequence is converted into integer token IDs and then padded/truncated to a fixed length of 50.

Raw Text
   ↓
Tokenization
   ↓
Integer Sequence
   ↓
Padding / Truncation
   ↓
Fixed-length sequence of 50 tokens

Model Experiments

Three recurrent architectures were compared:

1. Simple RNN

A basic recurrent architecture used as a baseline for sequential text processing.

2. LSTM

LSTM uses gated memory mechanisms to handle long-term dependencies and reduce the limitations of standard RNNs.

3. GRU

GRU provides gated recurrent processing with a simpler architecture than LSTM, using fewer gates and no separate cell state.

After comparing the architectures, GRU was selected for the final model based on the experimental results and its balance between sequence modeling capability and computational complexity.

Final Model Architecture

The final model uses a Bidirectional GRU with a 300-dimensional embedding representation.

Input Text
    ↓
Tokenizer
    ↓
Sequence Length = 50
    ↓
Embedding Layer
    ↓
300-dimensional Word Embeddings
    ↓
Bidirectional GRU
    ↓
Dropout (0.5)
    ↓
Dense Layer
    ↓
Softmax
    ↓
6 Emotion Classes

Why Bidirectional GRU?

A standard GRU processes the sequence in one direction. A Bidirectional GRU processes the sequence in both forward and backward directions.

This allows the model to use contextual information from both preceding and following words when determining the emotion expressed by a sentence.

For example:

"I am not happy with this result"

Understanding the relationship between words such as “not” and “happy” requires contextual information across the sequence. Bidirectional processing helps the model capture such relationships.

Word Embeddings

The model uses an embedding dimension of 300.

Instead of representing every word as a sparse one-hot vector, the embedding layer learns a dense numerical representation for each token.

Conceptually:

"happy"
   ↓
Token ID
   ↓
Embedding Layer
   ↓
[0.21, -0.14, 0.37, ..., 0.08]
        300 dimensions

These embeddings are learned during model training.

Regularization

A Dropout rate of 0.5 is used in the final architecture to reduce overfitting.

During training, dropout randomly disables a portion of the neurons, forcing the network to learn more robust representations rather than relying heavily on specific neurons.

Training

The final Bidirectional GRU model is trained using:

* Epochs: 32
* Batch size: 32
* Validation set: Hugging Face validation split
* Output classes: 6
* Output activation: Softmax

The validation set is used during training to monitor generalization, while the test set is kept separate for final evaluation.

Technologies Used

* Python
* TensorFlow / Keras
* Hugging Face Datasets
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab

Clone the repository:

git clone https://github.com/harshitraj1236/emotion.git
cd Emotion-Classification

Install the dependencies:

pip install -r requirements.txt

Example Prediction

Input:

"I am so happy that I finally achieved my goal!"

Output:

Predicted Emotion: Joy

Another example:

"I am scared about what will happen next."

Output:

Predicted Emotion: Fear

Conclusion

This project demonstrates the use of recurrent neural networks for NLP-based emotion classification. Simple RNN, LSTM, and GRU architectures were compared, and a Bidirectional GRU with 300-dimensional embeddings and 0.5 dropout was selected as the final architecture.

The model predicts one of six emotions from user-provided text: sadness, joy, love, anger, fear, and surprise.