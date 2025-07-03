import numpy as np

class neural_network:
	def __init__(self, hidden_layers, output_layer, loss):
		self.hidden_layers = hidden_layers
		self.output_layer = output_layer
		self.loss = loss

	def fit(self, x_train, y_train, batch=1):
		for i in range(len(x_train)):
			pred = self.forward_path(input=x_train[i])
			print(f"costs of prediction: {self.loss(pred, y_train[i]):.3f}\tprediciton of {x_train[i]}\tis {pred} true label {y_train[i]}")
			self.backward_path(y_true=y_train[i])
			if (i + 1) % batch == 0:
				print("update")
				self.updates()
				print(f"weights layer 1 after updatea\n{self.hidden_layers[0].weights}")
				print(f"weights layer 2 after update\n{self.hidden_layers[2].weights}")

	def forward_path(self, input):
		for hidden_layer in self.hidden_layers:
			hidden_layer.forward_path(input)
			input = hidden_layer.output
		self.output_layer.forward_path(input)
		return self.output_layer.prediction

	def backward_path(self, y_true):
		output = self.output_layer.backward_path(y_true)
		for layer in reversed(self.hidden_layers):
			output = layer.backward_path(output)
		return output

	def predict(self, input):
		predictions = []
		for i in range(len(input)):
			predictions.append(mayority_vote(self.forward_path(input[i])))
		return predictions
	
	def predict_proba(self, input):
		predictions = []
		for i in range(len(input)):
			predictions.append(self.forward_path(input[i]))
		return predictions
	
	def updates(self):
		for layer in self.hidden_layers:
			if isinstance(layer, dense_layer):
				layer.update_weights()
				layer.update_bias()

class dense_layer:
	def __init__(self, n_inputs, n_outputs, learning_rate, weights=None, bias=0):
		if weights:
			self.weights = weights
		else:
			self.weights = np.random.uniform(-1,1,(n_outputs,n_inputs))
		self.bias = np.full(shape=(n_outputs,1),fill_value=bias)
		self.learning_rate = learning_rate
		self.gow = list()
		self.gob = list()
		
	def forward_path(self, input):
		self.__pre_activation_output(input)

	def __pre_activation_output(self, input):
		self.input = input
		self.output = np.sum(np.dot(self.weights, input) + self.bias, axis=0)
	
	def backward_path(self, delta):
		delta = np.array(delta)
		self.dCda = np.dot(np.transpose(self.weights), delta)
		self.gow.append(self.gradient_of_weights(delta))
		self.gob.append(self.gradient_of_bias(delta))
		return self.dCda
	
	def update_weights(self):
		self.weights = self.weights - self.learning_rate * np.mean(self.gow ,axis=0)
		self.gow.clear()

	def update_bias(self):
		self.bias = self.bias - self.learning_rate * np.mean(self.gob, axis=0)
		self.gob.clear()

	def gradient_of_weights(self, delta):
		d = np.expand_dims(delta, axis=1)
		i = np.expand_dims(self.input, axis=1)
		return d * np.transpose(i)

	def gradient_of_bias(self, delta):
		return delta

class activationLayer:
	def __init__(self, func, dt):
		self.activate = func
		self.dtActivate = dt

	def forward_path(self, input):
		self.z = input
		self.__output()
		return self.output
	
	def __output(self):
		activate_arr = np.vectorize(self.activate)
		self.output = activate_arr(self.z)

	def backward_path(self, dCda):
		self.delta = np.multiply(dCda, self.dtActivate(self.z))
		return self.delta

class softmax_layer:
	def __init__(self):
		self.prediction: np.ndarray
		self.delta: np.ndarray

	def forward_path(self, input):
		self.prediction = np.zeros(shape=input.shape)
		input = input - np.max(input)
		sum = np.sum(np.e ** input)
		self.prediction = np.e**input / sum
		return self.prediction

	def backward_path(self, y_true):
		self.__soft_max_error(y_true)
		return self.delta

	def __soft_max_error(self, y_true):
		self.delta = self.prediction - y_true

class sigmoid_layer:
	def __init__(self):
		self.prediction = np.zeros(shape=(1,))
		self.delta: np.ndarray

	def forward_path(self, input):
		self.prediction[0] = sigmoid(np.sum(input))
		return self.prediction

	def backward_path(self, y_true):
		self.__soft_max_error(y_true)
		return self.delta

	def __soft_max_error(self, y_true):
		self.delta = self.prediction - y_true


def relu(value):
	return value if value > 0 else 0

def dtrelu(z):
	dt = [1 if value > 0 else 0 for value in z]
	return dt

def sigmoid(value):
	return 1 / (1 + np.e ** -value) 

def dtsigmoid(value):
	return sigmoid(value) * (1 - sigmoid(value))

def cross_entropy(prediction, true_label):
	true_idx = np.argmax(true_label)
	q = prediction[true_idx]
	return -np.log(q)

def binary_crossentropy(y, y_pred):
	N = 1
	return - 1/N * np.sum(y_pred * np.log2(y) + (1 - y_pred) * np.log2(1 - y))

def mayority_vote(prediction):
	return [1 if p == max(prediction) else 0 for p in prediction]
