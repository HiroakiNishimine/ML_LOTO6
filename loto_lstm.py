# -*- coding: utf-8 -*-
'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
import csv
import pdb
import numpy as np
from sklearn.model_selection import train_test_split


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# 1.データのロード
# 訓練データ,答えの用意
f = open( 'loto4data2.csv', 'r' )
reader = csv.reader( f )

x_train = np.array( [] ) # 訓練データの用意
y_train = np.array( [] ) # 答えの用意

print( reader )
for row in reader:
    print( row )
    x_train = np.append( x_train, int( row[3] ) )
    x_train = np.append( x_train, int( row[4] ) )
    x_train = np.append( x_train, int( row[5] ) )
    x_train = np.append( x_train, int( row[6] ) )

f.close()

f = open( 'loto4data2.csv', 'r' )
reader = csv.reader( f )
for row in reader:
    print( row )
    y_train = np.append( y_train, int( row[2] ) )

f.close()

# x:n行4列, y:n行1列に変更
x_train = x_train.reshape( -1, 4 )
y_train = y_train.reshape( -1, 1 )


# 訓練データの末尾データと答えの最初のデータを削除
x_train = np.delete( x_train, x_train.shape[0] - 1 , 0 )
y_train = np.delete( y_train, 0, 0 )

# テストデータの生成（配列の分割）
x_train, x_test, y_train, y_test = train_test_split( x_train, y_train, test_size=0.33, random_state=42 )

print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
# 2.ネットワークの構築
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 3.コンパイル
#     - 最適化アルゴリズム
#     - 損失関数
#     - 損失関数のリスト（metrics=['accuracy']）

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# 4.訓練
print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
