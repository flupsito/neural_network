import numpy as np

def sigmoid(x):
	return 1 / (1 + np.e ** -x)

def dSigmoid(x):
	s = sigmoid(x)
	return s * (1 - s)

def relu(x):
	return 0 if x < 0 else x

def drelu(x):
	return 1 if x > 0 else 0

def tanh(x): 
	a = np.e ** -x
	b = np.e ** x
	return (a - b / (a + b))

def dtanh(x):
	t = tanh(x)
	return 1 - t ** 2

def softmax(x):
	return x / np.sum(x)

def mse(y, y_pred):
	return np.mean((y_pred - y) ** 2)

def dmse(y, y_pred):
	return -2 * (y_pred - y)

def selectDerivative(act):
	if act == sigmoid:
		return dSigmoid
	elif act == tanh:
		return dtanh
	elif act == relu:
		return drelu
	elif act == mse:
		return dmse
	else:
		raise ValueError ('activation function not in scope tanh, sigmoid, relu')