#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.models import Sequential
import keras.utils as ku 
import pandas as pd
import numpy as np
import string, os 
import warnings
import jieba
import random
import sys
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# import dataset
male = pd.read_csv("AllMalePrizedArtist.csv")

df = male
lyrics = df['lyric_ckip']

lyrics_fulls = lyrics.values.tolist()

for i in range(len(lyrics_fulls)):
    lyrics_fulls[i] = lyrics_fulls[i].split("', '")
    lyrics_fulls[i][0] = lyrics_fulls[i][0][2:]
    lyrics_fulls[i][-1] = lyrics_fulls[i][-1][:-2]

lyrics_full = lyrics_fulls[:1400]


# In[3]:


#split in to segment
seg_len = 20
stride = 5

segments = []
next_chars = []
for i in range(len(lyrics_full)):
    for j in range(0, len(lyrics_full[i]) - seg_len, stride):
        segments.append(lyrics_full[i][j: j + seg_len])
        next_chars.append(lyrics_full[i][j + seg_len])


# In[4]:


#construct one-hot vectors
vocab = []
for i in range(len(lyrics_full)):
    tokens = sorted(lyrics_full[i])
    for token in tokens:
        if token not in vocab:
            vocab.append(token)
vocab_size = len(vocab)

segments_vector = np.zeros((len(segments), seg_len, vocab_size), int)
next_chars_vector = np.zeros((len(segments), vocab_size), int)
for i, segment in enumerate(segments):
    for j, char in enumerate(segment):
        segments_vector[i, j, vocab.index(char)] = 1
    next_chars_vector[i, vocab.index(next_chars[i])] = 1


# In[5]:


#LSTM model
model = Sequential()
model.add(LSTM(16, input_shape=(seg_len, vocab_size)))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()


# In[6]:


#Train model
from keras.optimizers import Adam
optimizer = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(segments_vector, next_chars_vector, batch_size=128, epochs=20)


# In[7]:


def sample(pred, temperature):
    pred = pred ** (1 / temperature)
    pred = pred / np.sum(pred)
    return np.argmax(pred)


# In[10]:


import csv
text_len = 80
temperature = 0.7
full_outcomes = []
for time in range(10):
    full = ""

    start_index_song = random.randint(0, len(lyrics_full) - 1)
    start_index_lyrics = random.randint(0, len(lyrics_full[start_index_song]) - 1.5*seg_len)

    generated_text_list = lyrics_full[start_index_song][start_index_lyrics:start_index_lyrics + seg_len]
    generated_text = ""
    
    
    for i in range(len(generated_text_list)):
        generated_text += str(generated_text_list[i])
    sys.stdout.write(generated_text)
    full += generated_text

    for i in range(text_len):
        sampled = np.zeros((1, seg_len, vocab_size), int)
        for j, char in enumerate(generated_text_list):
            sampled[0, j, vocab.index(char)] = 1
        pred = model.predict(sampled, verbose=0)[0]
        next_index = sample(pred, temperature)
        next_char = vocab[next_index]

        generated_text_list.append(next_char)
        generated_text_list = generated_text_list[1:]

        sys.stdout.write(str(next_char))
        full += str(next_char)
    full_outcomes.append([full])

with open(file='Result.csv', mode='w', newline="", encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for  outcome in full_outcomes:
        writer.writerow(outcome)


# In[ ]:




