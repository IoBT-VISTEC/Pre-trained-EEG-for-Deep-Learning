'''
This file is used for benchmark the model
execution format
python baseline.py $X$ $Y$ $model$ $n_freeze$ $out$
'''
import numpy as np
from model import direct_8,direct_8_dist,LSTM_2D,distributed_time,autoencoder_time,autoencoder_CNN,dense_1,hybrid_LSTM
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import itertools
from scipy import stats
from keras.optimizers import RMSprop,Adam,Adadelta,SGD
from keras import models
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from time import time
from loss import mean_squared_error_ignore_0,dummy
from keras.models import load_model
import keras.losses
import sklearn.preprocessing as pre
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from numpy import unique
from numpy import random
import tensorflow as tf 
from keras.utils import multi_gpu_model
import argparse
keras.losses.mean_squared_error_ignore_0 = mean_squared_error_ignore_0
epochs=1000

parser = argparse.ArgumentParser()
parser.add_argument("X", help="path to X.npy",
                    type=str)
parser.add_argument("Y", help="path to Y.npy",
                    type=str)
parser.add_argument("model", help="path to __.hdf5",
                    type=str)
parser.add_argument("n_freeze", help="# of freeze layers(6 for baseline, 63 for the proposed model)",
                    type=int)
parser.add_argument("out", help="output text path",
                    type=str)
args = parser.parse_args()

#normalization
channel_2d = [grid for grid in itertools.product(range(9), repeat=2)]
def normalize(X_train,X_val):
	mu=[stats.tmean(X_train[:,d[0],d[1]])  for d in channel_2d]
	std=[stats.tstd(X_train[:,d[0],d[1]]) for d in channel_2d]
	for i in range(len(channel_2d)):
		if(std[i]!=0):
			X_train[:,channel_2d[i][0],channel_2d[i][1]]= (X_train[:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i]
			X_val[:,channel_2d[i][0],channel_2d[i][1]]= (X_val[:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i]
	return X_train,X_val

#load data
X1 = np.load(args.X)
y1 = np.load(args.Y)
X_train, X_test, y_train, y_test = train_test_split(X1,y1,test_size=0.2,random_state=0)
X_train,X_test = normalize(X_train,X_test)
del X1,y1
with tf.device('/cpu:0'):
    model = load_model(args.model,custom_objects={'mean_squared_error_ignore_0':mean_squared_error_ignore_0})

#chop out unnecessary data
X_train = X_train[:,4:9,:,50:150]
X_test = X_test[:,4:9,:,50:150]

#freeze layer
for i in range(args.n_freeze):
    model.layers[i].trainable = False


csv_logger = CSVLogger(out+'.log')
tensorboard = TensorBoard(log_dir="../logs/{}".format(time()))

model.compile(optimizer=Adadelta(),loss=[mean_squared_error_ignore_0,'binary_crossentropy'],metrics=['accuracy'],loss_weights=[0,1.0])
model.fit(x=X_train,y=[X_train,y_train],batch_size=1024,epochs=300,validation_data=(X_test,[X_test,y_test]))

re,pre  = model.predict(X_test)
print(roc_auc_score(y_test,pre))
