# from keras.layers import Conv1D
import h5py
from numpy.random import random, random_integers
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Conv1D


# x = np.load('K562_RR_train.npy')
# print(x.shape)
# y1 = np.ones(int(len(x)/2))
# y2 = np.zeros(int(len(x)/2))
# y = np.concatenate((y1,y2),axis=0)
# print(y.shape)
# x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=10)
#
# INPUT_SHAPE = x_train.shape[1:3]
#
# model = Sequential()
# model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=INPUT_SHAPE))

# test_h5 = h5py.File("test.h5","w")
# imgname = np.fromstring('img.png',dtype=np.uint8).astype('float64')#str_imgname------>float64
# test_h5 .create_dataset('imgname', data=imgname)#变成f8之后就可以直接往h5中写了
# test_h5.close()
# """
# 6 最后得出来的矩阵长度是字符串的长度。---1个字符串的长度就是对应编码的h5向量的长度
# 7 如果想将多个字符串拼成一个大的numpy矩阵，写到h5文件中，必须先将字符串转换成相同长度。
# 8 通常的做法是在字符串后面补上\x00。
# 9 """
#
# test_h5 = h5py.File("test.h5","r")
# img = test_h5['imgname'][:]
# img = img.astype(np.uint8).tostring().decode('ascii')
# print(img)
# test_h5.close()

# def noramlization(data):
#     minVals = data.min(0)
#     print(minVals)
#     maxVals = data.max(0)
#     print(maxVals)
#     ranges = maxVals - minVals
#     # normData = np.zeros(np.shape(data))
#     m = data.shape[0]
#     print("m", m)
#     normData = data - np.tile(minVals, (m, 1))
#     print("np.tile(minVals, (m, 1))", np.tile(minVals, (m, 1)))
#     print("normData", normData)
#     normData = normData / np.tile(ranges, (m, 1))
#     print("np.tile(ranges, (m, 1)", np.tile(ranges, (m, 1)))
#     print("normData", normData)
#     return normData

def NPSE(k):
    number = len(a)
    length = len(a[0])
    feature_NPSE = np.array([[0.0] * (4 * 4 * (k + 1))] * number)
    for i in range(number):
        for s in range(k + 1):
            for j in range(length - (s + 1)):
                pos = int(a[i][j] * 4 + a[i][j + (s + 1)])
                feature_NPSE[i][pos + (4 * 4 * s)] = feature_NPSE[i][pos + (4 * 4 * s)] + 1 / (length - (s + 1))
    return feature_NPSE



a = random_integers(low=0, high=3, size=(3, 5))
metrics.auc()
metrics.precision_recall_fscore_support
metrics.roc_curve
print(len(a))
print(len(a[0]))
print(a)
print(NPSE(3))
# print(noramlization(a))
