"""
@authors: TheodoropoulosGiorgos, TritsarolisAndreas
"""

import math
import keras
import numpy as np

class SignalReconstructionRatio(keras.callbacks.Callback):
	def __init__(self, data_norm, data_num):
		self.data_norm = data_norm
		self.data_num  = data_num

	# def on_train_begin(self, logs={}):		
	# 	return
 
	# def on_train_end(self, logs={}):
	# 	return
 
	# def on_epoch_begin(self, epoch, logs={}):
	# 	return

	def on_epoch_end(self, epoch, logs={}):
		epoch_loss = logs.get('loss')
		mean_data_norm = self.data_norm/self.data_num
		print (f'\n[train_data] Signal-to-Reconstruction Ratio: {10 * np.log10(mean_data_norm/epoch_loss):.5} dB')
		
		val_epoch_loss = logs.get('val_loss')
		val_data_norm = np.sum(np.linalg.norm(self.validation_data[1], axis=1))
		val_mean_data_norm = val_data_norm/self.validation_data[1].shape[0]
		print (f'[val_data] Signal-to-Reconstruction Ratio: {10 * np.log10(val_mean_data_norm/val_epoch_loss):.5} dB\n')
		return
 
	# def on_batch_begin(self, batch, logs={}):
	# 	return
 
	# def on_batch_end(self, batch, logs={}):
	# 	return