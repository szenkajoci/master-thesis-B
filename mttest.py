# -*- coding: utf-8 -*-
"""
Created on Wed May 17 22:24:28 2023

@author: jozsef.szenka
"""

import h5py as h5
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
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
import seaborn as sns

modelname = "ujmod_22"

df = pd.read_excel("MT_BG.xlsx",sheet_name="MT") 

df['cppm_mu'] = df['cppm_mu'].clip(lower=0)

X_all = []
y_all = []
for i in range(df.shape[0]):
    X_all.append(df.iloc[i,[33,34,22,23,24]])
X_all = np.array(X_all,dtype=float)

X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1]))

print(X_all.shape)

model2 = keras.models.load_model('./'+modelname)
weights = model2.get_weights()

all_predicted = model2.predict(X_all)
