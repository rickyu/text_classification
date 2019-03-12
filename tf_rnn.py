import keras
from keras.preprocessing.text import Tokenizer
import sklearn
import sklearn.preprocessing
import tensorflow as tf
import sys

import time
import pandas as pd
from yznlp.rnn import load_csv_data,get_tencent_lstm_model,tokenize_texts,split_train_test


texts, labels = load_csv_data('data/ad1/ch_ad.csv')
seqs, vocab = tokenize_texts(texts)
X_train, Y_train, X_test, Y_test = split_train_test(seqs, labels)
sess = tf.Session(config=tf.ConfigProto(device_count={'gpu':1},log_device_placement=True))
import keras.backend.tensorflow_backend as KTF
KTF.set_session(sess)
model3 = get_tencent_lstm_model(vocab, 100, 'test/Tencent_AILab_ChineseEmbedding_example.txt')
model3.fit(X_train, Y_train, batch_size=32, epochs=12, validation_data=(X_test, Y_test))