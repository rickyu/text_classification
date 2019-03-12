#encoding: utf-8

import keras
from keras.preprocessing.text import Tokenizer
import sklearn
import sklearn.preprocessing
import tensorflow as tf
import pandas as pd
import numpy as np

def clean_chinese_str(s):
    out = ''
    #mystr = string.decode('utf-8')
    mystr = s
    mystr = mystr[:100]
    for letter in mystr:
        out += letter + ' '
    return out

def preprocess_text(texts):
    import jieba
    return texts.map(lambda x:' '.join(list(jieba.cut(x))))

# 加载数据,对文本进行预处理。返回texts, labels.
def load_csv_data(path):
    df_train = pd.read_csv(path)
    data_list = df_train['text']
    class_list = df_train['label']
    y_labels = list(class_list.value_counts().index)
    label_encoder = sklearn.preprocessing.LabelEncoder()
    label_encoder.fit(y_labels)
    num_labels = len(y_labels)
    y_labels = keras.utils.to_categorical(class_list.map(lambda x: label_encoder.transform([x])), num_labels)
    texts = preprocess_text(data_list)
    return texts, y_labels

def tokenize_texts(texts):
    tokenizer = Tokenizer(filters='!"#$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
    tokenizer.fit_on_texts(texts)
    vocab = tokenizer.word_index
    word_ids = tokenizer.texts_to_sequences(texts)
    padded_seqs = keras.preprocessing.sequence.pad_sequences(
        word_ids, padding='post', truncating='post', maxlen=100)
    return padded_seqs, vocab


def split_train_test(X, y):
    train_count = len(X)
    X_train = X[0:train_count]
    Y_train = y[0:train_count]
    X_test = X[train_count:]
    Y_test = y[train_count:]
    return X_train, Y_train, X_test, Y_test


def load_embedding_layer(word_index, input_length, embedding_path):
    import yznlp.wordvec

    embeddings, dim = yznlp.wordvec.load_tencent_word_embedding(embedding_path)
    weights = np.zeros((len(word_index), dim))
    unknown_words = []
    for word, i in word_index.items():
        vec = embeddings.get(word, None)
        if vec is not None:
            weights[i - 1] = vec
        else:
            unknown_words.append((word, i))
    print(len(unknown_words))
    return keras.layers.Embedding(len(word_index), dim, input_length=input_length, trainable=False, weights=[weights])


def get_tencent_lstm_model(word_index, input_length, embedding_path):
    embedding_layer = load_embedding_layer(word_index, input_length, embedding_path)
    model3 = keras.models.Sequential()
    model3.add(embedding_layer)
    model3.add(keras.layers.LSTM(256, dropout=0.5, recurrent_dropout=0.1))
    model3.add(keras.layers.Dense(256, activation='relu'))
    model3.add(keras.layers.Dense(2, activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model3

def get_tencent_lstm_model(word_index, input_length, embedding_path):
    embedding_layer = load_embedding_layer(word_index, input_length, embedding_path)
    model3 = keras.models.Sequential()
    model3.add(embedding_layer)
    model3.add(keras.layers.LSTM(256, dropout=0.5, recurrent_dropout=0.1))
    model3.add(keras.layers.Dense(256, activation='relu'))
    model3.add(keras.layers.Dense(2, activation='softmax'))
    model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model3