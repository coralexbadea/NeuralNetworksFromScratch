import numpy as np

class Dense: 

	def __init__(self, number_inputs, number_neurons, rand_init=True):
		if rand_init:
			self.weights = 0.01 * np.random.randn(number_inputs, number_neurons)
		else:
			self.weights = 0.01 * np.ones((number_inputs, number_neurons))
		self.biases =  0.01 * np.ones((1,number_neurons))


	def forward(self, inputs):
		self.inputs = inputs.copy()
		self.outputs = np.dot(self.inputs, self.weights) + self.biases

	def backward(self, doutputs):
		self.dweights = np.dot(self.inputs.T, doutputs)
		self.dbiases = np.sum(doutputs, axis=0, keepdims=True)
		self.dinputs = np.dot(doutputs, self.weights.T)
			
class ReLU:
	def forward(self, inputs):
		self.inputs = inputs.copy()
		self.outputs = np.maximum(0, inputs)
		
	def backward(self, doutputs):
		self.dinputs = doutputs.copy()
		self.dinputs[self.inputs <= 0] = 0 
		
		
class Softmax_CELoss:
	@classmethod
	def calculate_softmax(self, inputs):
		exponents = np.exp(inputs)
		return exponents/np.sum(exponents, axis=1, keepdims=True) #aka probabilities

	def forward(self,inputs, y_true):   
		self.softmax_outputs = self.calculate_softmax(inputs)
		self.celoss_outputs = -np.log(self.softmax_outputs[range(len(inputs)), y_true])

	def backward(self,y_true):
		self.dinputs = self.softmax_outputs.copy()
		self.dinputs[range(len(y_true)), y_true] -= 1
		self.dinputs /= len(y_true)
		
class SGD:
	def __init__(self, learning_rate = 0.01):
		self.learning_rate = learning_rate
		
	def update_params(self, layer):
		layer.weights -= self.learning_rate * layer.dweights
		layer.biases -= self.learning_rate * layer.dbiases
