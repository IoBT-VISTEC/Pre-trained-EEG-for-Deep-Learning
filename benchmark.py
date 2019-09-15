'''
This file is used for benchmark the model
execution format
python baseline.py $X$ $Y$ $model$ $n_freeze$ $out$
'''
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import itertools
from scipy import stats
from tensorflow.keras.optimizers import RMSprop,Adam,Adadelta,SGD
from tensorflow.keras import models
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import TensorBoard
from time import time

from tensorflow.keras.models import load_model
import tensorflow.keras.losses
import sklearn.preprocessing as pre
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_auc_score
from numpy import unique
from numpy import random
import tensorflow as tf 
from tensorflow.keras.utils import multi_gpu_model
import argparse

epochs=800

parser = argparse.ArgumentParser()
parser.add_argument("X", help="path to X.npy",
                    type=str)
parser.add_argument("Y", help="path to Y.npy",
                    type=str)
parser.add_argument("model", help="path to __.hdf5",
                    type=str)
parser.add_argument("out", help="output text path",
                    type=str)
args = parser.parse_args()
out = args.out
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

def EEGnetFormat(X):
    X = np.reshape(X,(X.shape[0],1,X.shape[1] * X.shape[2],X.shape[3]))
    notZero = [i for i in range(45) if i not in [44,43,42,38,37,36,35,33,29,27]]
    return X[:,:,notZero,:]
#load data
X1 = np.load(args.X)
y1 = np.load(args.Y)
X_train, X_test, y_train, y_test = train_test_split(X1,y1,test_size=0.2,random_state=0)
X_train,X_test = normalize(X_train,X_test)
del X1,y1
X_train = EEGnetFormat(X_train[:,4:9,:,50:150])
X_test = EEGnetFormat(X_test[:,4:9,:,50:150])

model = load_model(args.model)

csv_logger = CSVLogger(out+'.log')
tensorboard = TensorBoard(log_dir="../logs/{}_{}".format(out,time()))

model.compile(optimizer=Adadelta(),loss=['binary_crossentropy'],metrics=['accuracy'])
model.fit(x=X_train,y=y_train,batch_size=512,epochs=300,validation_data=(X_test,y_test))

pre  = model.predict(X_test)
print(roc_auc_score(y_test,pre))

