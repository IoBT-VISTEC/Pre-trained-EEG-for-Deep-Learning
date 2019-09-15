from keras.layers import Input, TimeDistributed, Conv2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Reshape,GRU, LeakyReLU, Reshape,Permute,ConvLSTM2D,Conv1D	,UpSampling2D,Conv2DTranspose,ZeroPadding2D,UpSampling1D,Cropping1D,Lambda,LSTM,RepeatVector
from keras.models import Model
from keras import initializers
import keras.backend as K
import tensorflow as tf

def hybrid_LSTM(depth=2,conv_size=16,dense_size=512,input_dim=(5,9,100,),dropoutRate=0.2):
	"""
	Autoencoder model builder composes of CNNs and a LSTM
	Args:
		depth (int): number of CNN blocks, each has 3 CNN layers with BN and a dropout
		conv_size (int): initial CNN filter size, doubled in each depth level
		dense_size (int): size of latent vector and a number of filters of ConvLSTM2D
		input_dim (tuple): input dimention, should be in (y_spatial,x_spatial,temporal)
		dropoutRate (float): dropout rate used in all nodes
	Return:
		keras model
	"""
	"""Setup"""
	temp_filter = conv_size
	X = Input(shape=input_dim, name = 'input')
	X = Permute((3,1,2))(X)  #move temporal axes to be first dim
	X = Reshape((100,5,9,1))(X) #reshape (,1) to be feature of each spatial

	"""Encoder"""
	for i in range(depth):
		for j in range(3):
			if j == 0: #j==0 is first layer(j) of the CNN block(i); apply stride with double filter size
				X = TimeDistributed(Conv2D(2*temp_filter,(3,3),padding='same' ,strides=(2,2),data_format="channels_last"),name = 'encoder_'+str(i)+str(j)+'_timeConv2D')(X)
			else:
				X = TimeDistributed(Conv2D(temp_filter,(3,3), padding='same', data_format="channels_last"),name = 'encoder_'+str(i)+str(j)+'_timeConv2D')(X)
			X = BatchNormalization(name = 'encoder_'+str(i)+str(j)+'_BN')(X)
			X = LeakyReLU(alpha=0.1,name = 'encoder_'+str(i)+str(j)+'_relu')(X)
			X = Dropout(dropoutRate,name = 'encoder_'+str(i)+str(j)+'_drop')(X)
		temp_filter = int(temp_filter * 2)
	X = TimeDistributed(Flatten())(X)
	X = LSTM(dense_size, recurrent_dropout=dropoutRate ,return_sequences=False, implementation=2)(X)

	"""Latent"""
	latent = X

	"""Setup for decoder""""
	X = RepeatVector(100)(X)
	temp_filter = temp_filter/2

	"""Decoder"""
	X = LSTM(temp_filter*2*3, recurrent_dropout=dropoutRate ,return_sequences=True, implementation=2)(X)
	X = Reshape((100,2,3,temp_filter))(X)
	for i in range(depth):
		for j in range(3):
			if j == 0:
				X = TimeDistributed(UpSampling2D((2,2)),name = 'decoder_'+str(i)+str(j)+'_upsampling')(X)
				X = TimeDistributed(ZeroPadding2D(((1,0),(1,0))),name = 'decoder_'+str(i)+str(j)+'_padding')(X)
				X = TimeDistributed(Conv2D(temp_filter,(3,3),data_format="channels_last"),name = 'decoder_'+str(i)+str(j)+'_timeConv2D')(X)
			else:
				X = TimeDistributed(Conv2D(temp_filter,(3,3), padding='same', data_format="channels_last"),name = 'decoder_'+str(i)+str(j)+'_timeConv2D')(X)
			X = BatchNormalization(name = 'decoder_'+str(i)+str(j)+'_BN')(X)
			X = LeakyReLU(alpha=0.1,name = 'decoder_'+str(i)+str(j)+'_relu')(X)
			X = Dropout(dropoutRate,name = 'decoder_'+str(i)+str(j)+'_drop')(X)
		temp_filter = int(temp_filter / 2)
	X = TimeDistributed(Conv2D(1,(1,1), padding='same', data_format="channels_last"),name = 'decoder__timeConv2D')(X)
	X = Reshape((100,5,9))(X)
	X = Permute((2,3,1))(X)
	decoded = X
	X = latent
	X = Dense(1,name = 'Dense10',activation='sigmoid')(X)
	return Model(inputs = model_input, outputs = [decoded,X])




def baseline(input_dim):
	"""
	Baseline mode(AE-SLIC) model
	Args:
		input_dim(tuple): input dimention; need to be in flatten format (samples,feature)
	Return:
		keras model
	"""

	model_input = Input(shape=input_dim, name = 'input')
	X=model_input
	X = Dense(500,name = 'Encoder1')(X)
	X = Dense(250,name = 'Encoder2')(X)
	latent = X
	X = Dense(500,name = 'Decoder1')(X)
	X = Dense(250,name = 'Decoder2')(X)
	X = Dense(4500,name = 'Decoder3')(X)
	Output = X
	X = Dense(1,activation = 'sigmoid')(latent)
	return Model(inputs = model_input, outputs = [Output,X])
