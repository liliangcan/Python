import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import data_load

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


x = np.load('data/K562npy/K562_FF_train.npy')
print(x.shape)
y1 = np.ones(int(len(x)/2))
y2 = np.zeros(int(len(x)/2))
y = np.concatenate((y1, y2), axis=0)
print(y.shape)
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=10)

x_test = np.load('data/K562npy/K562_FF_test.npy')
print(x_test.shape)
y1 = np.ones(int(len(x_test)/2))
y2 = np.zeros(int(len(x_test)/2))
y_test = np.concatenate((y1, y2), axis=0)
print(y_test.shape)


INPUT_SHAPE = x_train.shape[1:3]
'''KERNEL_SIZE = 5
LEARNING_RATE = 0.001
LSTM_UNITS = 32'''

LEARNING_RATE = 0.001
KERNEL_NUMBER = 32
KERNEL_SIZE = 5
LSTM_UNITS = 32


print(x_test.shape)