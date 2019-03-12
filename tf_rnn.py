import keras
from keras.preprocessing.text import Tokenizer
import sklearn
import sklearn.preprocessing
import tensorflow as tf
import sys

import time
import pandas as pd
df_train = pd.read_csv('data/ad1/ch_ad.csv')
data_list = df_train['text']
class_list = df_train['label']
y_labels = list(class_list.value_counts().index)
label_encoder = sklearn.preprocessing.LabelEncoder()
label_encoder.fit(y_labels)
num_labels = len(y_labels)
y_labels = keras.utils.to_categorical(class_list.map(lambda x: label_encoder.transform([x])), num_labels)
tokenizer = Tokenizer(filters='!"#$%&()*,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True,split=" ")
def preprocess_text(texts):
    import jieba
    return texts.map(lambda x:' '.join(list(jieba.cut(x))))
texts = preprocess_text(data_list)
tokenizer.fit_on_texts(texts)
vocab = tokenizer.word_index
def clean_chinese_str(s):
    out = ''
    #mystr = string.decode('utf-8')
    mystr = s
    mystr = mystr[:100]
    for letter in mystr:
        out += letter + ' '
    return out

train_count = int(len(df_train) * 0.9)

X_train = texts[0:train_count]
Y_train = y_labels[0:train_count]
X_test = texts[train_count:]
Y_test = y_labels[train_count:]

x_train_word_ids = tokenizer.texts_to_sequences(X_train)

x_test_word_ids = tokenizer.texts_to_sequences(X_test)
x_train_padded_seqs = keras.preprocessing.sequence.pad_sequences(x_train_word_ids, padding='post', truncating='post', maxlen=100)
x_test_padded_seqs = keras.preprocessing.sequence.pad_sequences(x_test_word_ids,padding='post', truncating='post', maxlen=100)
import numpy as np
x_train_padded_seqs=np.expand_dims(x_train_padded_seqs,axis=2)
x_test_padded_seqs=np.expand_dims(x_test_padded_seqs,axis=2)


num_cores = 4
CPU=False
GPU=True
if GPU:
    num_GPU = 1
    num_CPU = 1
if CPU:
    num_CPU = 1
    num_GPU = 0

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        log_device_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

import keras.backend.tensorflow_backend as KTF
KTF.set_session(sess)
model = keras.models.Sequential()
model.add(keras.layers.LSTM(256, dropout=0.5, recurrent_dropout=0.1))
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
t1= time.time()
model.fit(x_train_padded_seqs, Y_train, batch_size=64, epochs=12, validation_data=(x_test_padded_seqs, Y_test))
t2 = time.time()
print("finished, time=%.2f" % (t2-t1))
