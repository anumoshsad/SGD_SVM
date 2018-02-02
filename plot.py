#!/usr/bin/python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt


def read_file_to_matrix(file_path):
	with open(file_path) as fp:
		lines = fp.read().splitlines()
	X = np.zeros((len(lines), 123))
	Y = np.zeros(len(lines))
	for i,line in enumerate(lines):
		L = line.split()
		Y[i]=int(L[0])
		for j in range(1,len(L)):
			col = int(L[j].split(":")[0])-1   # there are 123 features so my index is 0..122
			X[i,col] = 1
	return X, Y


train_X, train_Y = read_file_to_matrix("/u/cs246/data/adult/a7a.train")
dev_X, dev_Y = read_file_to_matrix("/u/cs246/data/adult/a7a.dev")
test_X, test_Y = read_file_to_matrix("/u/cs246/data/adult/a7a.test")


def gradient(x_n, y_n, weights, bias, capacity, N):
	if 1-y_n*(np.dot(weights,x_n) + bias)>=0:
		dw = (1/N)*weights - capacity*y_n*x_n
		db = - capacity*y_n
	else:
		dw = weights/N
		db = 0.
	return dw,db

def SVM(X, Y, epochs, capacity, l_rate = 0.1):
	w = np.zeros(123)
	b = 0
	N = X.shape[0]
	for i in range(epochs):
		for n in range(N):
			dw, db = gradient(X[n], Y[n], w, b, capacity, N)
			w -= l_rate * dw
			b -= l_rate * db
	return w,b


def accuracy(X, Y, w, b):
	tot = len(X)
	correct=0
	for i in range(tot):
		if (np.dot(X[i],w)+b)*Y[i]>0:
			correct+=1
	return correct/tot

if __name__ == "__main__":
	capacities = np.logspace(-3,4, 50) # we will have 50 points for the plot
	dev_accuracy = []
	test_accuracy = []
	for c in capacities:
		w,b = SVM(train_X, train_Y, epochs = 5, capacity = c, l_rate = 0.1)
		dev_acc = accuracy(dev_X, dev_Y, w, b)
		test_acc = accuracy(test_X, test_Y, w, b)
		dev_accuracy.append(dev_acc)
		test_accuracy.append(test_acc)
	

	plt.plot(capacities, dev_accuracy, label = "Dev")
	plt.plot(capacities, test_accuracy, label = "Test")
	plt.xscale('log')	
	plt.xlabel("Capacity, C")
	plt.ylabel("Accuracy")
	plt.title("Accuracy vs Capacity")
	plt.legend()
	plt.savefig("plot.png")
	plt.show()
	

















