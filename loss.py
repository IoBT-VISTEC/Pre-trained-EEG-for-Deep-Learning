import keras.backend as K
from keras.losses import binary_crossentropy
def mean_squared_error_ignore_0(y_true, y_pred):
	""" loss function computing MSE of non-blank(!=0) in y_true
		Args:
			y_true(tftensor): true label
			y_pred(tftensor): predicted label
		return:
			MSE reconstruction error for loss computing
	"""
	loss = K.switch(K.equal(y_true, K.constant(0)),K.zeros(K.shape(y_true)),K.square(y_pred - y_true))
	return K.mean(loss, axis=-1)

def dummy(y_true, y_pred):
	"""
	return a tensor of zero
	"""
	return K.mean(K.zeros(K.shape(y_true)))
