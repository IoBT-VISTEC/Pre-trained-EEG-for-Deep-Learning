'''
This file is used to train the model with four different datasets
execution format
python train.py $X1$ $Y1$ ... $X4$ $Y4$ $model$ $out$
'''

import numpy as np
from model import hybrid_LSTM, baseline
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import itertools
from scipy import stats
from keras.optimizers import RMSprop,Adam,Adadelta,SGD
from keras import models
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard,LearningRateScheduler
from time import time
from loss import mean_squared_error_ignore_0,dummy
import tensorflow as tf
from keras.utils import multi_gpu_model
import keras.backend as K
from keras.losses import binary_crossentropy
from numpy import unique
from numpy import random 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("X1", help="path to X.npy",
                    type=str)
parser.add_argument("Y1", help="path to Y.npy",
                    type=str)
parser.add_argument("X2", help="path to X.npy",
                    type=str)
parser.add_argument("Y2", help="path to Y.npy",
                    type=str)
parser.add_argument("X3", help="path to X.npy",
                    type=str)
parser.add_argument("Y3", help="path to Y.npy",
                    type=str)
parser.add_argument("X4", help="path to X.npy",
                    type=str)
parser.add_argument("Y4", help="path to Y.npy",
                    type=str)
parser.add_argument("X5", help="path to X.npy",
                    type=str)
parser.add_argument("Y5", help="path to Y.npy",
                    type=str)
parser.add_argument("model", help="model name",
                    type=str)
parser.add_argument("out", help="output text path",
                    type=str)
args = parser.parse_args()


epochs=800

#normalization
channel_2d = [grid for grid in itertools.product(range(9), repeat=2)]
def normalize(X_train,X_val):
    mu=[stats.tmean(X_train[:,d[0],d[1]])  for d in channel_2d]
    std=[stats.tstd(X_train[:,d[0],d[1]]) for d in channel_2d]
    for i in range(len(channel_2d)):
        if(std[i]!=0):
            X_train[:,channel_2d[i][0],channel_2d[i][1]]= (X_train[:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i]
            X_val[:,channel_2d[i][0],channel_2d[i][1]]= (X_val[:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i]
    # minn = [np.min(X_train[:,d[0],d[1]]) for d in channel_2d]
    # maxx = [np.max(X_train[:,d[0],d[1]]) for d in channel_2d]
    # for i in range(len(channel_2d)):
    #   if(maxx[i]-minn[i]!=0):
    #       X_train[:,channel_2d[i][0],channel_2d[i][1]]= (X_train[:,channel_2d[i][0],channel_2d[i][1]]-minn[i]) / (maxx[i]-minn[i])
    #       X_val[:,channel_2d[i][0],channel_2d[i][1]]= (X_val[:,channel_2d[i][0],channel_2d[i][1]]-minn[i]) / (maxx[i]-minn[i])
    return X_train,X_val

#train weight
def weightFunction(y_train):
    weight = np.zeros(2) # class
    for i in y_train:
      weight[int(i)] +=1 
    d = np.min(weight)
    weight = [temp/d for temp in weight]
    temp = np.empty_like(y_train)
    temp[y_train==0] = weight[0]
    temp[y_train==1] = weight[1]
    return temp
#load data
X1 = np.load(args.X1)
y1 = np.load(args.Y1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1,test_size=0.2,random_state=0)
X1_train,X1_test = normalize(X1_train,X1_test)
y1_train = np.reshape(y1_train,(-1,1))
y1_test = np.reshape(y1_test,(-1,1))
w1 = weightFunction(y1_train)
X1_w = np.multiply(np.reshape(np.repeat(len(X1_train),len(X1_train)),(-1,1)),w1)
del X1,y1

X2 = np.load(args.X2)
y2 = np.load(args.Y2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,test_size=0.2,random_state=0)
X2_train, X2_test = normalize(X2_train,X2_test)
y2_train = np.reshape(y2_train,(-1,1))
y2_test = np.reshape(y2_test,(-1,1))
w2 = weightFunction(y2_train)
X2_w = np.multiply(np.reshape(np.repeat(len(X2_train),len(X2_train)),(-1,1)),w2)
del X2,y2


X3 = np.load(args.X3)
y3 = np.load(args.Y3)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,test_size=0.2,random_state=0)
X3_train,X3_test = normalize(X3_train,X3_test)
y3_train = np.reshape(y3_train,(-1,1))
y3_test = np.reshape(y3_test,(-1,1))
w3 = weightFunction(y3_train)
X3_w = np.multiply(np.reshape(np.repeat(len(X3_train),len(X3_train)),(-1,1)),w3)
del X3,y3


X4 = np.load(args.X4)
y4 = np.load(args.Y4)
X4_train, X4_test, y4_train, y4_test = train_test_split(X4,y4,test_size=0.2,random_state=0)
X4_train,X4_test = normalize(X4_train,X4_test)
y4_train = np.reshape(y4_train,(-1,1))
y4_test = np.reshape(y4_test,(-1,1))
w4 = weightFunction(y4_train)
X4_w = np.multiply(np.reshape(np.repeat(len(X4_train),len(X4_train)),(-1,1)),w4)
del X4,y4

X5 = np.load(args.X5)
y5 = np.load(args.Y5)
X5_train, X5_test, y5_train, y5_test = train_test_split(X5,y5,test_size=0.2,random_state=0)
X5_train,X5_test = normalize(X5_train,X5_test)
y5_train = np.reshape(y5_train,(-1,1))
y5_test = np.reshape(y5_test,(-1,1))
w5 = weightFunction(y5_train)
X5_w = np.multiply(np.reshape(np.repeat(len(X5_train),len(X5_train)),(-1,1)),w5)
del X5,y5



X_train = np.concatenate((X2_train,X1_train,X3_train,X4_train,X5_train))
X_weight = np.concatenate((X1_w,X2_w,X3_w,X4_w,X5_w)).flatten()
X_weight = 1.0/X_weight*np.max(X_weight)
del X1_train,X3_train,X4_train,X2_train,X5_train
X_test = np.concatenate((X1_test,X2_test,X3_test,X4_test,X5_test))
del X1_test,X2_test,X3_test,X4_test,X5_test

y_train = np.concatenate((y1_train,y2_train,y3_train,y4_train,y5_train))
del y1_train,y2_train,y3_train,y4_train,y5_train
y_test = np.concatenate((y1_test,y2_test,y3_test,y4_test,y5_test))
del y1_test,y2_test,y3_test,y4_test,y5_test

#chop out unnecessary data
X_train = X_train[:,4:9,:,50:150]
X_test = X_test[:,4:9,:,50:150]
with tf.device('/cpu:0'):
    if args.model == 'hybrid_LSTM':
        model = hybrid_LSTM(depth=2,conv_size=8,dense_size=512,input_dim=(5,9,100,),dropoutRate=0.2)
    if args.model == 'baseline':
        model = baseline(input_dim=(4500,))
        #baseline is FC layers not CNN, need to flatten
        X_train = np.reshape(X_train,(X_train.shape[0],-1))
        X_test = np.reshape(X_test,(X_test.shape[0],-1))

model.compile(optimizer=SGD(lr=0.002,decay=1E-5),loss=[mean_squared_error_ignore_0,'binary_crossentropy'],metrics=['accuracy'],loss_weights=[0.4,0.6])
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.__setattr__('callback_model',model)
parallel_model.compile(optimizer=SGD(lr=0.002,decay=1E-5),loss=[mean_squared_error_ignore_0,'binary_crossentropy'],metrics=['accuracy'],loss_weights=[0.4,0.6])


print(model.summary())

#train the model
csv_logger = CSVLogger(out+'.log')
filepath="out"+{epoch:02d}+".hdf5"
tensorboard = TensorBoard(log_dir="../logs/{}".format(time()))
checkpointer = ModelCheckpoint(monitor='val_loss', filepath=filepath, verbose=1, save_best_only=True)
parallel_model.fit(x=X_train,y=[X_train,y_train],batch_size=512,epochs=epochs,validation_data=(X_test,[X_test,y_test]),callbacks=[checkpointer,csv_logger,tensorboard],sample_weight=[X_weight,X_weight]) 
