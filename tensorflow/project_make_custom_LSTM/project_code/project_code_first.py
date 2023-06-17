import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import re
#--------------------------------------------------------------------------------
#1. data preprocessing
data = pd.read_csv('Hotel_reviews.csv', encoding='utf-8')
def remove_emoji(text):
    if isinstance(text, str):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # 이모티콘
                                   u"\U0001F300-\U0001F5FF"  # 기호 및 문장 부호
                                   u"\U0001F680-\U0001F6FF"  # 트랜스포트 및 심볼
                                   u"\U0001F1E0-\U0001F1FF"  # 국기 및 기호
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text
def remove_hanja(text):
    if isinstance(text, str):
        hanja_pattern = re.compile("[\u4E00-\u9FFF]+")
        return hanja_pattern.sub(r'', text)
    else:
        return text
data['review_full'] = data['review_full'].apply(remove_emoji)
data['review_full'] = data['review_full'].apply(remove_hanja)
data.dropna(subset=['review_full', 'rating_review'], inplace=True)
train_data = data[data.index % 2 == 0].reset_index(drop=True)
test_data = data[data.index % 2 == 1].reset_index(drop=True)
train_reviews = train_data['review_full']
train_ratings = train_data['rating_review']
test_reviews = test_data['review_full']
test_ratings = test_data['rating_review']
#--------------------------------------------------------------------------------
#2. data (text) processing for using LSTM
tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_reviews)

vocab_size = len(tokenizer.word_index) + 1

train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

max_sequence_length = max(len(seq) for seq in train_sequences)
train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)
#--------------------------------------------------------------------------------
#3. make LSTM based model
model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=max_sequence_length))
model.add(LSTM(128))
model.add(Dense(1, activation='linear'))
#--------------------------------------------------------------------------------
#4. model compile & train
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(train_data, train_ratings, epochs=1, batch_size=32) #epoch 10
#--------------------------------------------------------------------------------
#5. model predict
predictions = model.predict(test_data)
predictions = predictions.astype(int)
for i in range(len(test_reviews)):
    print("Review:", test_reviews[i])
    print("Predicted Rating:", predictions[i])
    print("-------------")
#--------------------------------------------------------------------------------
#6. visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(test_reviews)), test_ratings, label='Actual')
ax.set_xlabel('Review Index')
ax.set_ylabel('Rating')
ax.set_title('Actual')
ax.legend()
plt.show()
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(test_reviews)), predictions, label='Predicted')
ax.set_xlabel('Review Index')
ax.set_ylabel('Rating')
ax.set_title('Predicted')
ax.legend()
plt.show()


