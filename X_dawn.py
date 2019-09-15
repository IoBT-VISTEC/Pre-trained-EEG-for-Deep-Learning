'''
xDawn algorithm add name in file variable below
'''
file = ['00','15','18','17','bci_comp_zeropad','erpbci_zeropad','300']
from sklearn.pipeline import make_pipeline
from mne.decoding import Vectorizer
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit,StratifiedKFold
from pyriemann.estimation import ERPCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn
import pandas as pd
from collections import OrderedDict

import numpy as np
import itertools

from numpy import unique
from numpy import random

channel_2d = [grid for grid in itertools.product(range(9), repeat=2)]

def normalize(X_train,X_val):
	mu=[stats.tmean(X_train[:,d[0],d[1]])  for d in channel_2d]
	std=[stats.tstd(X_train[:,d[0],d[1]]) for d in channel_2d]
	for i in range(len(channel_2d)):
		if(std[i]!=0):
			X_train[:,channel_2d[i][0],channel_2d[i][1]]= (X_train[:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i]
			X_val[:,channel_2d[i][0],channel_2d[i][1]]= (X_val[:,channel_2d[i][0],channel_2d[i][1]]-mu[i]) / std[i]
	return X_train,X_val

results=0
if __name__ == "__main__":
    f = []
    auc = []
    acc = []
    methods = []
    clfs = OrderedDict()
    clfs['Xdawn + RegLDA'] = make_pipeline(Xdawn(2), Vectorizer(), LDA(shrinkage='auto', solver='eigen'))

    for name in file:
    	X1 = np.load('data/X_' + name + '.npy')
    	y1 = np.load('data/Y_' + name + '.npy')
    	X = X1[:,4:9,:,50:150]
    	X = np.reshape(X,(-1,9*5,100))
    	y = y1.flatten()
    	zero = np.sum(X,axis=-1)[0]!=0
    	X = X[:,zero,:]
    	cv = StratifiedKFold(n_splits=10, random_state=0)
    	for m in clfs:
    	    print name,m
    	    res1 = cross_val_score(clfs[m], X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    	    print name,m,res1[0]
    	    res2 = cross_val_score(clfs[m], X, y, scoring='roc_auc', cv=cv, n_jobs=-1)
    	    print name,m,res2[0]
    	    acc.extend(res1)
    	    auc.extend(res2)
    	    methods.extend([m]*len(res1))
    	    f.extend([name]*len(res1))

    results = pd.DataFrame(data=auc, columns=['AUC'])
    results['ACC'] = acc
    results['Method'] = methods
    results['file'] = f
np.save('out',results)
