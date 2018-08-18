"""
@authors: TheodoropoulosGiorgos, TritsarolisAndreas
"""
import time
import subprocess
from multiprocessing import Process

class Scheduler:
	def __init__(self):
		# self.AVAILABLE_GPUS = self.__mp_get_available_gpus()	# for getting the gpu ids using the subprocess library
		self.AVAILABLE_GPUS = self.__tf_get_available_gpus()	# for getting the gpu ids using the tensorflow library
		self.PENDING_MODELS = []
		self.PENDING_MODELS_ARGS = []
        
	def __mp_get_available_gpus(self):
		proc = subprocess.Popen(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
		output = proc.stdout.read().decode('utf-8').split('\n')
		# print ('Raw Info:', output)
		gpu_ids = []
		for line in output:
			if not line:
				continue
			gpu_ids.append(line.split()[1].split(':')[0])
		return gpu_ids

	def __tf_get_available_gpus(self):
		import tensorflow
		from tensorflow.python.client import device_lib
		from keras.backend.tensorflow_backend import set_session
		
		config = tensorflow.ConfigProto()
		config.gpu_options.allow_growth = True		
		with tensorflow.Session(config=config) as sess:
			local_device_protos = device_lib.list_local_devices()
		return [x.name for x in local_device_protos if x.device_type == 'GPU']

	def __init_process_per_gpu(self):
		processes = []
		for i in self.AVAILABLE_GPUS:
			try:
				target_func = self.PENDING_MODELS.pop()
				target_kwargs = self.PENDING_MODELS_ARGS.pop()
				target_kwargs['gpu_id'] = str(i)

				proc = Process(target=target_func, name=str(i), kwargs=target_kwargs)
				proc.start()
				processes.append(proc)
			except IndexError:
				break
		return processes

	def __clean_processes(self, processes):
		for proc in processes:
			if not proc.is_alive():
				try:
					procname = proc.name
					proc.join()
					processes.pop(processes.index(proc))

					target_func = self.PENDING_MODELS.pop()
					target_kwargs = self.PENDING_MODELS_ARGS.pop()
					target_kwargs['gpu_id'] = procname
					
					proc = Process(target=target_func, name=procname, kwargs=target_kwargs)
					proc.start()
					processes.append(proc)
				except IndexError:
					pass

	def __alive_threads(self, processes):
		allAlive = False
		for proc in processes:
			allAlive |= proc.is_alive()
		return allAlive

	def Enqueue(self, PENDING_MODELS = [], PENDING_MODELS_ARGS = []):
		self.PENDING_MODELS = PENDING_MODELS
		self.PENDING_MODELS_ARGS = PENDING_MODELS_ARGS
        
	def Start(self):
		procs = self.__init_process_per_gpu()
		while True:
			if not self.PENDING_MODELS and not self.__alive_threads(procs):
				print('[+] All Models Trained.')
				break
			self.__clean_processes(procs)
			time.sleep(1)
