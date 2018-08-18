"""
@authors: TheodoropoulosGiorgos, TritsarolisAndreas
"""


import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import autoencoder as autoenc
from gpu_scheduler import Scheduler



# Runs the Autoencoder module in parallel using all the available GPUs

def autodec(data, gpu_id):
	#print (data.shape)
	scaler = StandardScaler()
	X_std  = scaler.fit_transform(data)

	initiald = X_std.shape[1]       # Feature Dimensions
	targetd = 500                   # TO-DO Change to Desired Output with sys.argv[*]

	autoEncoder = autoenc.AutoEncoder(initiald, targetd, lr_=1, momentum_=0.001, decay_=0.0001, nesterov_=False)
	#lr_=0.5

	autoEncoder.fit(X_std, gpuid=gpu_id, batch_size=100, epochs=10, verbose=1, validation_split=0.2)
	# end monitoring

if __name__ == "__main__" :
	queue = [autodec, autodec, autodec]
	queue_args = [{'data': make_blobs(n_samples=10000, centers=20, n_features=1916, random_state=0)[0]},
            	  {'data': make_blobs(n_samples=20000, centers=20, n_features=1916, random_state=10)[0]},
                  {'data': make_blobs(n_samples=30000, centers=20, n_features=1916, random_state=100)[0]}]

	_scheduler = Scheduler()
	_scheduler.Enqueue(queue, queue_args)
	_scheduler.Start()