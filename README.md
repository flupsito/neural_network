In this repository you find the neural_network.py file
This file contains the simple implementation of a neural network made to solve classification problems:
It is split in a number of helper functions and some classes
The class neural network represents the overall nn and saves the included layer
The other classes are:
-  dense_layer
-  activation_layer implementing ReLu activation function
-  softmax_layer implementing softmax function to generate class_predictions

How to:
  define the size and amounts of hidden_layer needed for your classification problem
  you can create a dense_layer object with a custom initial weight function or a prefilled weight matrix
  biasas are initialized as zeros
  the learning rate has to be set by the user
  activation and softmax objects dont require arguments
  One has to create a neural network object:
  needed arguments are list of hidden layer starting with the hidden_layer closest to the input
  one output_layer object

  after the initialization training the 
