#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:42:13 2019

@author: papadaki
"""

import pandas as pd
import json
import glob
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
import re
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


MAX_LENGTH = 300 # number of words to use from article
EMBEDDING_SIZE = 100
vocab_size = 88991


def clean_text(text):

    text = re.sub(r"\n"," ", text)
    text = re.sub(r"“", "\"", text)
    text = re.sub(r"”", "\"", text)
    text = re.sub(r"[‘’]", "'", text)

    # Convert words to lower case and split them
    text = text.lower().split()

    # remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    translator = str.maketrans('', '', string.punctuation)

    # Clean the text
    text = re.sub(r"\-", " ", text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r"\s{2,}", " ", text)

    ## Stemming
    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return(text)


files = glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/*/fake/*/news content.json')
cols = ['id', 'label', 'text']
lst = []

for filename in files:
    with open(filename, 'r') as json_file:
        article_id = os.path.basename(os.path.dirname(filename))
        data = json.load(json_file)
        text = data['text']
        text = clean_text(text)
        lst.append([article_id, 1, text])

files = glob.iglob('FakeNewsNet/code/fakenewsnet_dataset/*/real/*/news content.json')
for filename in files:
    with open(filename, 'r') as json_file:
        article_id = os.path.basename(os.path.dirname(filename))
        data = json.load(json_file)
        text = data['text']
        text = clean_text(text)
        lst.append([article_id, 0, text])

df = pd.DataFrame(lst, columns=cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

### Create sequence
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=MAX_LENGTH)


# load the whole embedding into memory
embeddings_index = dict()
f = open('glove/glove.6B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, EMBEDDING_SIZE))
for word, i in tokenizer.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector

e = Embedding(vocab_size, EMBEDDING_SIZE, weights=[embedding_matrix], 
              input_length=MAX_LENGTH, trainable=False)

# doing it probably the better way
X_train, X_test, y_train, y_test = train_test_split(data, df['label'], 
                                                    random_state=12, test_size=0.2)
model = Sequential()
model.add(e)
#model.add(Flatten())
#model2.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200, batch_size=64,
                     validation_split=0.1,
                     callbacks=[EarlyStopping(monitor='val_loss',
                                              patience=3,
                                              min_delta=0.0001)])


###### EVALUATION #######
pred = model.predict_classes(X_test, batch_size=32, verbose=1)
report = classification_report(np.array(y_test), pred)
print(report)


####### PLOTS #######

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

accr = model.evaluate(X_test, y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))



####### ADDENDUM ########

# get word embeddings
word_embds = model.layers[0].get_weights()
word_list = []
for word, i in tokenizer.word_index.items():
    word_list.append(word)

# count class breakdown in data
df.groupby('label').count()
