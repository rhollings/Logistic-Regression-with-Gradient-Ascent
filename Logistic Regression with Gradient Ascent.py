import csv
import pandas as pd
import numpy as np

trainFile = "FILE NAME HERE.txt"
testFile = "FILE NAME HERE.txt"
learningRate = 0.0001
steps = 3000

def reformat(file):
	for row in file:
		row = '1 ' + row 			#Add bias
		row = row.replace(':','')
		yield row

#Sigmoid function equation. Thetas dot product is passed in.
def sigmoid(scores):
	return 1/(1+np.exp(-scores))


with open(trainFile) as tf, open(testFile) as ttf:

	##### files #####

	train = np.loadtxt(reformat(tf), skiprows=2)
	test = np.loadtxt(reformat(ttf), skiprows=2)

	##### train #####

	numrows = train.shape[0]
	numcols = train.shape[1]-1
	features = train[:, 0:numcols]  #Matrix of all features
	label = train[:, numcols]       #List of all labels
	thetas = np.zeros(numcols)      #Initialize thetas. Num thetas = num features + label
	n = 0
	for step in range(steps):
		grad = np.zeros(numcols)    #Initilize blank thetas for current step
		for row in range(numrows):
			scores = np.dot(thetas, features[row])    #Dot overall thetas with all features of curr row
			prediction = label[row] - sigmoid(scores) #Log likelihood function
			xi = features[row]
			grad += xi * prediction   #Multiply each feature with the prediction part of the likelihood equation
		for col in range(numcols):
			thetas[n] += learningRate*grad[n]   #Multiply learningRate and found thetas with overall theteas
			n+=1
		n=0										#One training step completed

	##### test #####

	prob = 0.0
	success = 0.0
	testFeatures = test[:, 0:numcols]			#Matrix of all features in test file
	for row in range(test.shape[0]):
		scores = np.dot(thetas, testFeatures[row])   #Dot thetas found in training and features in curr row
		prob = sigmoid(scores)						 #Plug in scores in sigmoid to find overall prob
		if prob > 0.5 and test[row][test.shape[1]-1] == 1:  #If prob > 0.5 = label should be 1
			success += 1
		elif prob < 0.5 and test[row][test.shape[1]-1] == 0:
			success += 1
	print(success/test.shape[0])
