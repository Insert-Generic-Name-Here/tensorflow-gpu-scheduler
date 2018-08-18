"""
@authors: TheodoropoulosGiorgos, TritsarolisAndreas
"""
import math, keras
import numpy as np
import tensorflow as tf
from keras import optimizers
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard
from custom_callbacks import SignalReconstructionRatio
from keras.backend.tensorflow_backend import set_session
import sys
import os


class AutoEncoder:
	def __init__(self, initial_dim, target_dim, 
				 lr_=0.3, momentum_=0.001, decay_=0.0001, nesterov_=False):
		#set environment
		#Initializing Neural Network
		self.training = False
		self.autoEncoder = None
		self.initial_dim = initial_dim
		self.target_dim = target_dim
		self.log_run = 0

		layers = self.__CreateDecodingLayers(initial_dim, target_dim)
		
		with tf.device("/cpu:0"):
			# initialize the model
			self.autoEncoder = Sequential()	
			
			self.autoEncoder.add(Dense(input_dim = initial_dim, units = layers[0], kernel_initializer = 'uniform', activation = 'tanh', name='AutoEncoder_Input'))
			# Adding the rest of the hidden layers
			for output_dim in layers[1:-1]:
				self.autoEncoder.add(Dense(units = output_dim, kernel_initializer = 'uniform', activation = 'tanh'))

			# Adding the output layer
			self.autoEncoder.add(Dense(units = layers[-1], kernel_initializer = 'uniform', activation = 'linear', name='Encoded_Output'))

			# Adding the rest of the hidden layers
			for output_dim in layers[::-1][1:-1]:
				self.autoEncoder.add(Dense(units = output_dim, kernel_initializer = 'uniform', activation = 'tanh'))

			# Adding the output layer
			self.autoEncoder.add(Dense(units = layers[::-1][-1], kernel_initializer = 'uniform', activation = 'linear', name='Decoded_Output'))


		# Compiling Neural Network
		sgd = optimizers.SGD(lr=lr_, momentum=momentum_, decay=decay_, nesterov=nesterov_)
		self.autoEncoder.compile(optimizer = sgd, loss = 'mean_squared_error', metrics = None)


	def __CreateDecodingLayers(self, initiald, targetd):
		return [initiald, math.floor((initiald + targetd)/2), targetd]

	# def __CreateDecodingLayers(self, initiald, targetd):
	# 	neurons = [initiald]
	# 	nxt = 2 ** (math.floor(math.log2(initiald)))
	# 	neurons.append(nxt)
	# 	nxt //= 2
	# 	while (nxt > targetd):
	# 		neurons.append(nxt)
	# 		nxt //= 2
	# 	neurons.append(targetd)
	# 	return neurons

	def fit(self, data, gpuid='0', comp_StRr=True, tensorbrd = False, 
			batch_size=200, epochs=10, verbose=1, validation_split=0.2 ):
		
		self.log_run += 1
		calbck_list = []
		if (tensorbrd):
			tensorboard = TensorBoard(log_dir="logs/Run #{}".format(self.log_run), histogram_freq=0, write_graph=True, write_images=False)
			calbck_list.append(tensorboard)
		if (comp_StRr):
			data_norm = np.sum(np.linalg.norm(data, axis=1))
			StRR = SignalReconstructionRatio(data_norm, data.shape[0])
			calbck_list.append(StRR)
		if calbck_list is [] :
			calbck_list = None
		
		gpu_num = gpuid.split(':')[-1]
		with tf.name_scope(gpu_num) as scope:
			# with tf.device('/device:GPU:'+str(gpuid)):
			with tf.device(gpuid):
				config = tf.ConfigProto()
				config.gpu_options.allow_growth = True
				
				set_session(tf.Session(config=config))
				self.autoEncoder.fit(data, data, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split, callbacks=calbck_list)


	def transform(self, data):
		# with a Sequential model
		try:
			getEncodedOutput = K.function([self.autoEncoder.layers[0].input],
										[self.autoEncoder.get_layer('Encoded_Output').output])
			layer_output = getEncodedOutput([data])[0]
		except Exception:
			return None
		return layer_output


	def decode(self, data):
		# with a Sequential model
		encoded_index = -math.floor(len(self.autoEncoder.layers)/2)
		try:
			encoded_input = keras.Input(shape=(self.target_dim,))
			deco = self.autoEncoder.layers[encoded_index](encoded_input)
			encoded_index += 1
			while(encoded_index <= -1):
				deco = self.autoEncoder.layers[encoded_index](deco)
				encoded_index += 1
			decoder = keras.Model(encoded_input, deco)
			layer_output = decoder.predict(data)
		except Exception:
			return None
		return layer_output


	def fit_transform(self, data, batch_size_=200, epochs_=10, verbose_=1, validation_split_=0.2):
		self.fit(data, batch_size_, epochs_, verbose_, validation_split_)
		return self.transform(data)


	def summary(self):
		return self.autoEncoder.summary()
