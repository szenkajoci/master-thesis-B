# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:15:11 2023

@author: jozsef.szenka
"""

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import math
import keras
import os
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
import tensorflow as tf
import pandas as pd
import datetime
from time import time, sleep
import keras.backend as K

plt.rcParams['text.usetex'] = True

valami1 = 0
valami2 = 0

modelname = "ujmod_30"

def errorFunction(y_true, y_pred):
    valami11 = tf.math.reduce_sum(tf.matmul(y_pred,tf.constant([1,0,0,0],dtype=float,shape=[2,2])),axis=-1)
    valami12 = tf.math.reduce_sum(tf.matmul(y_pred,tf.constant([0,0,0,1],dtype=float,shape=[2,2])),axis=-1)
    valami21 = tf.math.reduce_sum(tf.matmul(y_true,tf.constant([1,0,0,0],dtype=float,shape=[2,2])),axis=-1)
    valami22 = tf.math.reduce_sum(tf.matmul(y_true,tf.constant([0,0,0,1],dtype=float,shape=[2,2])),axis=-1)
    mean_pred = tf.math.multiply(valami11,valami12)
    variance_pred = tf.math.multiply(valami11, tf.math.pow(valami12,2))
    skewness_pred = tf.math.multiply(tf.math.sign(valami11), tf.math.divide_no_nan(tf.ones_like(valami11) * 2,tf.math.sqrt(tf.math.abs(valami11))))
    kurtosis_pred = tf.math.divide_no_nan(tf.ones_like(valami11) * 6,valami11)
    mean_true = tf.math.multiply(valami21,valami22)
    variance_true = tf.math.multiply(valami21, tf.math.pow(valami22,2))
    skewness_true = tf.math.multiply(tf.math.sign(valami21), tf.math.divide_no_nan(tf.ones_like(valami21) * 2,tf.math.sqrt(tf.math.abs(valami21))))
    kurtosis_true = tf.math.divide_no_nan(tf.ones_like(valami21) * 6,valami21)
    elements_pred = tf.stack([valami11, valami12, mean_pred, variance_pred, skewness_pred, kurtosis_pred],axis=-1)
    elements_true = tf.stack([valami21, valami22, mean_true, variance_true, skewness_true, kurtosis_true],axis=-1)
    loss = (abs(valami21 - valami11) / valami21 + abs(valami22 - valami12) / valami22 + \
            abs(mean_true - mean_pred) / mean_true + abs(variance_true - variance_pred) / variance_true + \
            abs(skewness_true - skewness_pred) / skewness_true + abs(kurtosis_true - kurtosis_pred) / kurtosis_true ) / 6 * 100
    return loss 

def calcError(x,xfit):
    mean = x.mean()
    print([x,xfit,sum((x-xfit) ** 2),sum((x-mean) ** 2)])
    return 1 - sum((x-xfit) ** 2) / sum((x-mean) ** 2)

def fitandplot(x,y,name = ""):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(x, y, alpha=0.2)
    rrrr = calcError(y, x)
    ax.plot(x, x, color='C1')
    plt.xlabel("Known value")
    plt.ylabel("Estimated value")
    eq = (rf"$$y=x$$" r"\vspace{-30pt} \\" 
          rf"$$R^2= {rrrr:.4f}$$")
    ax.text(0.05, 0.95, eq, color="k",
            horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
    if name != "" : 
        plt.title(name)
        if not os.path.exists("./"+modelname+"_img"):
           os.makedirs("./"+modelname+"_img")
        plt.savefig("./"+modelname+"_img/"+name.replace(" ", "")+".png")
    plt.show()


df = pd.read_table("exports30.txt",header=None,names=["k1","th1","k2","th2","kc","thc","muc","err"]) 

df = df.dropna()

#df2 = pd.read_table("exports20.txt") 

#df2 = df2.dropna()

#df = pd.concat([df,df2])

df = df.where(df['muc'] > 0).dropna()
df = df.where(df['k1'] > 0.1).dropna()

train_size, val_size = 0.7, 0.15

num_time_steps = df.shape[0]
num_train, num_val = (
    int(num_time_steps * train_size),
    int(num_time_steps * val_size),
)

X_all = []
y_all = []
for i in range(df.shape[0]):
    X_all.append(df.iloc[i,[2,3,4,5,6]])
    y_all.append(df.iloc[i,[0,1]])
X_all, Y_all = np.array(X_all), np.array(y_all)

X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1]))
Y_all = np.reshape(Y_all, (Y_all.shape[0], Y_all.shape[1]))

train_array = df[:num_train]
val_array = df[num_train : (num_train + num_val)]
test_array = df[(num_train + num_val) :]

X_train = []
y_train = []
for i in range(train_array.shape[0]):
    X_train.append(train_array.iloc[i,[2,3,4,5,6]])
    y_train.append(train_array.iloc[i,[0,1]])
X_train, Y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
Y_train = np.reshape(Y_train, (Y_train.shape[0], Y_train.shape[1]))

X_valid = []
y_valid = []
for i in range(val_array.shape[0]):
    X_valid.append(train_array.iloc[i,[2,3,4,5,6]])
    y_valid.append(train_array.iloc[i,[0,1]])
X_valid, Y_valid = np.array(X_valid), np.array(y_valid)

X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1]))
Y_valid = np.reshape(Y_valid, (Y_valid.shape[0], Y_valid.shape[1])) 

X_test = []
y_test = []
for i in range(test_array.shape[0]):
    X_test.append(train_array.iloc[i,[2,3,4,5,6]])
    y_test.append(train_array.iloc[i,[0,1]])
X_test, Y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1])) 
Y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1]))

print(X_train.shape)
print(X_valid.shape)

hiddenwidth = 100

inputs = keras.Input(shape=(5,))
first = layers.Dense(hiddenwidth, activation='relu')(inputs)
second = layers.Dense(3/2*hiddenwidth, activation='relu')(first)
# third = layers.Dense(2*hiddenwidth, activation='relu')(second)
fourth = layers.Dense(3/2*hiddenwidth, activation='relu')(second)
fifth = layers.Dense(hiddenwidth, activation='relu')(fourth)
output = layers.Dense(2, activation='linear')(fifth)

model = keras.Model(inputs, output)

log_dir = "/tmp/autoencoder/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

optimizer = keras.optimizers.Adam()

#model.compile(optimizer=optimizer, loss=keras.losses.mean_squared_error)
model.compile(optimizer=optimizer, loss=keras.losses.mean_absolute_percentage_error)

history = model.fit(X_train, Y_train,
                epochs=1500,
                batch_size=64,
                shuffle=True,
                validation_data=(X_valid, Y_valid),
                callbacks=[TensorBoard(log_dir=log_dir)])

plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train', linewidth=1)
plt.plot(history.history['val_loss'], label='test', linewidth=1)
plt.legend()
plt.show()

dfhistory = pd.DataFrame(data=history.history)
dfhistory.to_excel(modelname + ".xlsx")

model.save('./' + modelname)
weights = model.get_weights()

all_predicted = model.predict(X_all)

all_maxkerror = -1
all_maxkerrpos = -1

all_maxtherror = -1
all_maxtherrpos = -1

for i, row in enumerate(all_predicted):
    if(abs((row[0] - Y_all[i][0])/Y_all[i][0]) > all_maxkerror):
        all_maxkerror = abs((row[0] - Y_all[i][0])/Y_all[i][0])
        all_maxkerrpos = i
    if(abs((row[1] - Y_all[i][1])/Y_all[i][1]) > all_maxtherror):
        all_maxtherror = abs((row[1] - Y_all[i][1])/Y_all[i][1])
        all_maxtherrpos = i
    
fitandplot(Y_all[:,0], all_predicted[:,0],name="Shape parameter")
fitandplot(Y_all[:,1], all_predicted[:,1],name="Scale parameter")
fitandplot(Y_all[:,0] * Y_all[:,1], all_predicted[:,0] * all_predicted[:,1],name="Mean")
fitandplot(Y_all[:,0] * Y_all[:,1] ** 2, all_predicted[:,0] * all_predicted[:,1] ** 2,name="Variance")
fitandplot(2/np.sqrt(Y_all[:,0]), 2/np.sqrt(all_predicted[:,0]),name="Skewness")
fitandplot(6/Y_all[:,0], 6/all_predicted[:,0],name="Kurtosis")
