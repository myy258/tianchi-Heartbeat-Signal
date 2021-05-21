# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:00:31 2021

@author: yazce
"""
import pandas as pd 
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 1 # 分配显存
config.gpu_options.allow_growth = True # 按需分配显存
session = InteractiveSession(config=config)
import tensorflow as tf
import warnings
from keras.utils import np_utils
warnings.filterwarnings('ignore')

train_data = pd.read_csv(r'D:/share/tainchi_心跳信号分类预测/train.csv')
test_data = pd.read_csv(r'D:/share/tainchi_心跳信号分类预测/testA.csv')

train_data.heartbeat_signals
test_data.heartbeat_signals

train_get_feature = [k.split(',') for k in train_data.heartbeat_signals]
test_get_feature = [k.split(',') for k in test_data.heartbeat_signals]

train_feature = pd.DataFrame(train_get_feature)

#train_feature['zero'] = ''
#for i in range(len(train_feature)):
#    train_feature['zero'][i] = train_feature.iloc[i,:].value_counts().tolist()[0]
train_feature = train_feature.astype(float)

test_feature = pd.DataFrame(test_get_feature)
test_feature = test_feature.astype(float)

from sklearn.model_selection import train_test_split
X_train, X_validation, Y_train, Y_validation = train_test_split(train_feature,train_data.label,test_size = 0.2)

train_X = np.array(X_train)
train_Y = np.array(Y_train)
X_validation = np.array(X_validation)
#test_X = np.array(test_feature)
#test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], 1)

model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001), input_shape = (train_X.shape[1],1)),
        tf.keras.layers.Conv1D(filters=64, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        tf.keras.layers.Conv1D(filters=128, kernel_size=(5,), padding='same', activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        tf.keras.layers.MaxPool1D(pool_size=(5,), strides=2, padding='same'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
#        tf.keras.layers.Dense(units=1024, activation=tf.keras.layers.LeakyReLU(alpha=0.001)),
        tf.keras.layers.Dense(units=4, activation='softmax')
])

#model = tf.keras.Sequential()
#model.add(tf.keras.layers.Conv1D(64, 15, strides=1, padding='SAME',input_shape=(205, 1), use_bias=False))
#model.add(tf.keras.layers.ReLU())
#model.add(tf.keras.layers.Conv1D(64, 3, strides=2,padding='SAME'))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Conv1D(64, 3, strides=2,padding='SAME'))
#model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.LSTM(64, dropout=0.5, return_sequences=True))
#model.add(tf.keras.layers.LSTM(32))
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(4, activation="softmax"))

model.summary()
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

num_epochs = 55
model.fit(train_X, 
          train_Y,
          batch_size=64,
          epochs=num_epochs,
          verbose=2)

def abs_sum(y_tru,y_pre):
    y_tru = pd.get_dummies(data=y_tru)
    y_pre=np.array(y_pre)
    y_tru=np.array(y_tru)
    loss=sum(sum(abs(y_pre-y_tru)))
    return loss

pre = model.predict(X_validation)
abs_sum(Y_validation,pre)

#pre = model.predict(test_X)
#data_test_price = pd.DataFrame(pre,columns = ['label_0','label_1','label_2','label_3'])
#results = pd.concat([test_data['id'],data_test_price],axis = 1)
#submit_file_z_score = r'D:/share/tainchi_心跳信号分类预测/cnn.csv'
#results.to_csv(submit_file_z_score,encoding='utf8',index=0)