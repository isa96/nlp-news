# -*- coding: utf-8 -*-
"""

Sumber Data : https://www.kaggle.com/hgultekin/bbcnewsarchive?select=bbc-news-data.csv
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
import timeit

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  print(
      '\n\nThis error most likely means that this notebook is not '
      'configured to use a GPU.  Change this in Notebook Settings via the '
      'command palette (cmd/ctrl-shift-P) or the Edit menu.\n\n')
  raise SystemError('GPU device not found')

def cpu():
  with tf.device('/cpu:0'):
    random_image_cpu = tf.random.normal((100, 100, 100, 3))
    net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
    return tf.math.reduce_sum(net_cpu)

def gpu():
  with tf.device('/device:GPU:0'):
    random_image_gpu = tf.random.normal((100, 100, 100, 3))
    net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
    return tf.math.reduce_sum(net_gpu)
  
# We run each op once to warm up; see: https://stackoverflow.com/a/45067900
cpu()
gpu()

# Run the op several times.
print('Time (s) to convolve 32x7x7x3 filter over random 100x100x100x3 images '
      '(batch x height x width x channel). Sum of ten runs.')
print('CPU (s):')
cpu_time = timeit.timeit('cpu()', number=10, setup="from __main__ import cpu")
print(cpu_time)
print('GPU (s):')
gpu_time = timeit.timeit('gpu()', number=10, setup="from __main__ import gpu")
print(gpu_time)
print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

import pandas as pd
import os, re, string
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Embedding, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/bbc-news-data.csv', sep='\t')
df = df.drop(columns=['filename'])

df.head()

df.category.value_counts()

df.shape

plt.figure(figsize = (10, 6))
sns.countplot(df.category)

remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)

def cleaner(data):
    return(data.translate(str.maketrans('','', string.punctuation)))
    df.title = df.title.apply(lambda x: cleaner(x))
    df.content = df.content.apply(lambda x: lem(x))

def rem_numbers(data):
    return re.sub('[0-9]+','',data)
    df['title'].apply(rem_numbers)
    df['content'].apply(rem_numbers)

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def lem(data):
    pos_dict = {'N': wn.NOUN, 'V': wn.VERB, 'J': wn.ADJ, 'R': wn.ADV}
    return(' '.join([lemmatizer.lemmatize(w,pos_dict.get(t, wn.NOUN)) for w,t in nltk.pos_tag(data.split())]))
    df.title = df.title.apply(lambda x: lem(x))
    df.content = df.content.apply(lambda x: lem(x))

from nltk.corpus import stopwords
nltk.download('stopwords')

stwrds = stopwords.words('english')
def stopword(data):
    return(' '.join([w for w in data.split() if w not in stwrds ]))
    df.title = df.title.apply(lambda x: stopword(x))
    df.content = df.content.apply(lambda x: lem(x))

category = pd.get_dummies(df['category'])
df = pd.concat([df, category], axis=1)
df = df.drop('category', axis=1)
df.head()

a = df['title'].values + '' + df['content'].values

a

b=df.iloc[:,2:]

b

a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)

print(b_train.shape)
print(b_test.shape)

tokenizer = Tokenizer(num_words=5000, oov_token='x')

tokenizer.fit_on_texts(a_train) 
tokenizer.fit_on_texts(a_test)
 
sekuens_train = tokenizer.texts_to_sequences(a_train)
sekuens_test = tokenizer.texts_to_sequences(a_test)
 
padded_train = pad_sequences(sekuens_train) 
padded_test = pad_sequences(sekuens_test)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.97 and logs.get('val_accuracy')>0.90):
      self.model.stop_training = True
      print("\nAkurasi sudah mencapai > 97%, hentikan proses training!")
callbacks = myCallback()

model = tf.keras.Sequential([
     tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.LSTM(64),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(5, activation='softmax')
     ])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'
              ])

model.summary()

history = model.fit(padded_train, b_train, epochs=50, 
                    validation_data=(padded_test, b_test), verbose=2, callbacks=[callbacks], validation_steps=30)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()