import numpy as np
try:
  import cPickle as pickle
except ImportError:
  import pickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
from statistics import median, mean
from collections import Counter
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

def CreateEnviroment():
	enviroment = gym.make('CartPole-v0')
	enviroment.reset()
	return enviroment;

learnRate = 1e-3
enviroment = CreateEnviroment()
goalSteps = 500
scoreRequirement = 50
initialGames = 10000 
randomEpisodes = 0
rewardSum = 0

def RunRandomAttempts():
	while randomEpisodes < 100:
		observation = enviroment.reset() 
		for step in range(goalSteps):
			enviroment.render() # render current enviroment.
			action = enviroment.action_space.sample() # get a random action.. will go to right side, or left, whatever..
			observation, reward, done, info = enviroment.step(action) # do some random action ..
			if done:
				print("Episode finished after {} timesteps".format(step+1))
				break

def InitialPopulation():
	trainingData = []
	scores = []
	acceptedScores = []
	for _ in range(initialGames):
		score = 0
		gameMemory = []
		previousObservation = [] 
	
		for _ in range(goalSteps):
			action = enviroment.action_space.sample()
			observation, reward, done, info = enviroment.step(action)

			if len(previousObservation) > 0 :  #is not the first iteration
				gameMemory.append([previousObservation, action]) # I've applied this action at this scenario.
				
			previousObservation = observation
			score+=reward # how many iterations to done? 
			if done : 
				break

		if score >= scoreRequirement : 
			acceptedScores.append(score)
			for data in gameMemory:
				if data[1] == 1: # if car go to the right side...
					output = [0,1]
				elif data[1] == 0: # if car go to the left side..
					output = [1,0]		
				trainingData.append([data[0], output])	
			
		enviroment.reset()
		scores.append(score)

	trainingDataSave = np.array(trainingData)
	np.save('saved.npy', trainingDataSave)

	print('Average Accpeted score: ', mean(acceptedScores))
	print('Median accpeted score: ', median(acceptedScores))
	print(Counter(acceptedScores))
	return trainingData

def BuildNeuralNetworkModel(inputSize):
	network = input_data(shape=[None, inputSize, 1], name = 'input')

	network = fully_connected(network, 128, activation = 'relu')		
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation = 'relu')		
	network = dropout(network, 0.8)

	network = fully_connected(network, 512, activation = 'relu')		
	network = dropout(network, 0.8)

	network = fully_connected(network, 254, activation = 'relu')		
	network = dropout(network, 0.8)

	network = fully_connected(network, 128, activation = 'relu')		
	network = dropout(network, 0.8)

	network = fully_connected(network, 2, activation='softmax')
	network = regression(network, optimizer='adam', learning_rate=learnRate,
						loss='categorical_crossentropy', name='targets')
	model = tflearn.DNN(network, tensorboard_dir='log')

	return model

def BuildKerasNeuralNetworkModel(inputSize):
	model = Sequential()
	model.add(Dense(128, activation='relu', input_shape=[None, inputSize, 1]))
	model.add(Dropout(0.8))

	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.8))

	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.8))

	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.8))

	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.8))

	model.add(Dense(2, activation='softmax'))
	model.compile(optimizer='adam', learning_rate=learnRate, loss='categorical_crossentropy', name='targets')
	return model


BuildNeuralNetworkModel(50)