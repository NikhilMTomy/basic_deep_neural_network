#!/usr/bin/env python

import pandas as pd
import numpy as np
import model

TRAIN = True
TEST = True
LOAD = True
SAVE = True

def output_reshape(y, output_size):
	y_out = np.zeros((output_size, y.shape[1]))
	for j in range(y.shape[1]):
		y_out[y[0][j]][j] = 1
	return y_out

def label_for_one(label):
	if label == 1:
		return 1
	else:
		return 0

def normalize_value(value):
	return value/255.

vectorized_normalize_value = np.vectorize(normalize_value)
vectorized_label_for_one = np.vectorize(label_for_one)

network_model = model.Model()

if __name__ == '__main__':
	train_data = pd.read_csv('./dataset/mnist_train.csv', delimiter=',')
	train_y = output_reshape(np.array(train_data.iloc[:,0]).reshape((1, -1)), 10)
	train_x = vectorized_normalize_value(np.array(train_data.iloc[:, 1:]).T)

	test_data = pd.read_csv('./dataset/mnist_test.csv', delimiter=',')
	test_y = output_reshape(np.array(test_data.iloc[:,0]).reshape((1, -1)), 10)
	test_x = vectorized_normalize_value(np.array(test_data.iloc[:, 1:]).T)

	network_model.initialize(train_x, train_y, test_x, test_y, batch_size=100, iterations=50, learning_rate=0.01)
	if LOAD:
		network_model.load()
	if TRAIN:
		test_costs = network_model.train()
	if TEST:
		network_model.test(test_x, test_y)
	if SAVE:
		network_model.save()
