#https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional,SimpleRNN
from tensorflow.keras import Input

reviews = pd.read_csv('IMDB Dataset.csv')
reviews.head()
	
reviews['sentiment'] = np.where(reviews['sentiment'] == 'positive', 1, 0)

sentences = reviews['review'].to_numpy()
labels = reviews['sentiment'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25)

vocab_size = 10000
oov_tok = "<OOV>"
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
tokenizer.word_counts
tokenizer.word_docs
train_sequences = tokenizer.texts_to_sequences(X_train)
sequence_length = 200
train_padded = pad_sequences(train_sequences, maxlen=sequence_length, padding='post', truncating='post')
test_sequences = tokenizer.texts_to_sequences(X_test)
test_padded = pad_sequences(test_sequences, maxlen=sequence_length, padding='post', truncating='post')

model = Sequential()

embedding_dim = 16
model.add(Input(shape=(sequence_length,)))
model.add(Embedding(vocab_size, embedding_dim))
lstm_out = 32
model.add(Bidirectional(LSTM(lstm_out)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

	
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_padded, y_train, epochs=10, validation_data=(test_padded, y_test))

new_text = "this movie is good"
new_text_sequence = tokenizer.texts_to_sequences([new_text])
new_text_padded = pad_sequences(new_text_sequence, maxlen=sequence_length, padding='post', truncating='post')

# Πρόβλεψη του συναισθήματος του νέου κειμένου
prediction = model.predict(new_text_padded)

# Εκτύπωση της πρόβλεψης
print("Predicted Sentiment:", ("Positive" if prediction > 0.5 else "Negative"), "with evaluation:",prediction)


