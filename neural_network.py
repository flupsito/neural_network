import numpy as np
from typing import Literal
from activation import relu, tanh, sigmoid, softmax, mse, selectDerivative

class neural_network:
	def __init__(self, hidden_layers, output_layer, loss, learning_rate=1e-4):
		self.hidden_layers = hidden_layers
		self.output_layer = output_layer
		self.loss = loss
		self.dloss = selectDerivative(loss)
		self.learning_rate = learning_rate

	def fit(self, x_train, y_train, batch=1):
		pred = self.forward_path(x_train)
		print(f"costs of prediction: {self.loss(pred, np.array(y_train))}")
		self.backward_path(pred, y_train)
		self.updates()
	
	def forward_path(self, input):
		for hidden_layer in self.hidden_layers:
			input = hidden_layer.forward_path(input)
		self.output_layer.forward_path(input)
		return self.output_layer.x

	def backward_path(self, y_pred, y):
		output = self.dloss(y_pred, y)
		output = self.output_layer.backward_path(output)
		for layer in reversed(self.hidden_layers):
			output = layer.backward_path(output)
		return output

	def predict(self, input):
		print(input)
		predictions = []
		for i in range(len(input)):
			predictions.append(self.forward_path(input[i]).copy())
		return predictions
	
	def predict_proba(self, input):
		predictions = []
		for i in range(len(input)):
			predictions.append(self.forward_path(input[i]))
		return predictions
	
	def updates(self):
		for layer in self.hidden_layers:
			if isinstance(layer, dense_layer):
				layer.update(self.learning_rate)
				


class dense_layer:
	def __init__(self, n_inputs, n_outputs, learning_rate, activation ,weights=None, bias=0):
		if weights:
			self.weights = weights
		else:
			self.weights = np.random.uniform(-1,1,(n_outputs,n_inputs))
		self.bias = np.full(shape=(n_outputs,),fill_value=bias)
		self.learning_rate = learning_rate
		self.activation = activation
		self.dActivation = selectDerivative(activation)

	def forward_path(self, input):
		self.input = input
		self.z = np.dot(input, self.weights.T) + self.bias
		self.x = self.activation(self.z)
		return self.x

	def backward_path(self, input):
		N = input.shape[0]
		dz_ = input * self.dActivation(self.z)
		self.gow = np.dot(self.input.T, dz_) / N
		self.gob = np.sum(dz_, axis=0, keepdims=True) / N
		return np.dot(dz_, self.weights)
	
	def update(self, learning_rate):
		self.weights = self.weights - learning_rate * self.gow
		self.bias = self.bias - learning_rate * self.gob
	
def cross_entropy(prediction, true_label):
	true_idx = np.argmax(true_label)
	q = prediction[true_idx]
	return -np.log(q)

def binary_crossentropy(y, y_pred):
	N = 1
	return - 1/N * np.sum(y_pred * np.log2(y) + (1 - y_pred) * np.log2(1 - y))

if __name__ == '__main__':
	layer1 = dense_layer(2, 3, 0.1, activation=sigmoid, bias=1)
	layer2 = dense_layer(3, 2, 0.1, activation=sigmoid, bias=1)
	layer3 = dense_layer(2, 1, 0.1, activation=sigmoid, bias=0)

	nn = neural_network([layer1, layer2], layer3, mse)

	nn.fit(np.array([[0,0],[1,0],[0,1],[1,1]]), np.array([[0],[1],[1],[0]]))