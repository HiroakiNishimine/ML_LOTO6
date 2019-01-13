'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
ひと桁目を当てるプログラム
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD, Adam
import csv
import sys
import pdb
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
#from keras.utils import multi_gpu_model

batch_size = 1000
num_classes = 10
epochs = 1000
# full_model_path = 'checkpoints/weights.870-16.06.hdf5'

# 訓練データ,答えの用意
f = open( 'loto4data2.csv', 'r' )
reader = csv.reader( f )

x_train = np.array( [] ) # 訓練データの用意
y_train = np.array( [] ) # 答えの用意

# print( reader )
for row in reader:
    # print( row )
    x_train = np.append( x_train, int( row[3] ) )
    x_train = np.append( x_train, int( row[4] ) )
    x_train = np.append( x_train, int( row[5] ) )
    x_train = np.append( x_train, int( row[6] ) )

f.close()

f = open( 'loto4data2.csv', 'r' )
reader = csv.reader( f )
for row in reader:
    # print( row )
    # y_train = np.append( y_train, int( row[2] ) ) # for 10000 classification
    # for 10 classification
    y_train = np.append( y_train, int( row[3] ) )
    # y_train = np.append( y_train, int( row[4] ) )
    # y_train = np.append( y_train, int( row[5] ) )
    # y_train = np.append( y_train, int( row[6] ) )

f.close()

# x:n行4列, y:n行1列に変更
x_train = x_train.reshape( -1, 4 )
# y_train = y_train.reshape( -1, 1 ) # for 10000 classification
y_train = y_train.reshape( -1, 1 ) # for 10 classification
# pdb.set_trace()


# 訓練データの末尾データと答えの最初のデータを削除
x_train = np.delete( x_train, x_train.shape[0] - 1 , 0 )
y_train = np.delete( y_train, 0, 0 )

# テストデータの生成（配列の分割）
x_train, x_test, y_train, y_test = train_test_split( x_train, y_train, test_size=0.13, random_state=42 )
# x_train, x_test = np.split( x_train, [2000] )
# y_train, y_test = np.split( y_train, [2000] )


# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = x_train.reshape(2000, 4)
# x_test = x_test.reshape(495, 4)
x_train = x_train.astype( 'float32' )
x_test = x_test.astype( 'float32' )

# 入力データは[0,1]の範囲に正規化しなければならないので、
# 入力が画像の場合、画素の最大である255で割り算しておく
x_train /= 10
x_test /= 10
print( x_train.shape[0], 'train samples' )
# print( x_train[0] )
# print( y_train.shape )
print( x_test.shape[0], 'test samples' )
# print( x_test[0][0] )

# convert class vectors to binary class matrices
# Kerasはラベルを数値ではなく、0or1を要素に持つベクトル（バイナリベクトル）で扱うため
# ラベルをバイナリベクトルに変換する
y_train = keras.utils.to_categorical( y_train, num_classes )
y_test = keras.utils.to_categorical( y_test, num_classes )
# y_train = y_train.reshape( -1, 10000 )
# y_test = y_train.reshape( -1, 10000 )

# np.ma.mask_or を用いて、on-hotラベルをmulti-hotラベルに変換する必要がある

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(4,)))
# model.add(Dense(512, activation='sigmoid', input_shape=(4,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
# model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

# parallel_model = multi_gpu_model(model, gpus=2)
# parallel_model.compile(loss='categorical_crossentropy', #　多クラス分類の場合
model.compile(loss='categorical_crossentropy', #　多クラス分類かつone-hot labelingの場合
# model.compile(loss='binary_crossentropy', # 1クラス分類、またはmulti-hot labelingの場合
# model.compile(loss='mean_squared_error',
            #   optimizer=RMSprop(),
              optimizer=SGD(),
            #   optimizer=Adam(),
              metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=2)
# print(full_model_path)
# model.load_weights(full_model_path)
# print('Model loaded.')
#
# es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
# tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, histogram_freq=1)

callbacks = [keras.callbacks.ModelCheckpoint('./checkpoints/2/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                             verbose=1,
                                             save_weights_only=False),
            TensorBoard(log_dir='./graph',histogram_freq=1,write_graph=True)]

# fit 固定のエポック数でモデルを訓練する．
# history = parallel_model.fit(x_train, y_train,
history = model.fit( x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_split=0.2,
                     validation_data=(x_test, y_test),
                     shuffle=True,
                     initial_epoch=0,  # 訓練開始時のepoch（前の学習から再開する際に便利です）
                    #  callbacks=[es_cb, tb_cb])
                     callbacks=callbacks)

# validation_split: 浮動小数点数 (0. < x < 1) で， ホールドアウト検証のデータとして使うデータの割合．
# validation_data: ホールドアウト検証用データとして使うデータのタプル (x_val, y_val) か (x_val, y_val, val_sample_weights)． 設定すると validation_split を無視します．
# shuffle: 真理値か文字列 (for 'batch')． 各エポックにおいてサンプルをシャッフルするかどうか． 'batch' は HDF5 データだけに使える特別なオプションです．バッチサイズのチャンクの中においてシャッフルします．

# evaluate バッチごとにある入力データにおける損失値を計算します．
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
lot
