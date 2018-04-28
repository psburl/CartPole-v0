#Romulo Marchioro Rossi - Wesley Burlani
#Federal University of Fronteira Sul - 2018.1 - Artificial Inteligence

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 
import numpy as np
import tensorflow as tf
try:
    import matplotlib as mpl
    mpl.use('TkAgg')
    import mpl.pyplot as pl
except:
    import matplotlib.pyplot as pl
import math
import gym
from statistics import median, mean
from collections import Counter
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os.path

class Cartpole:
    def __init__(self):
        self.enviroment = gym.make('CartPole-v0')
        self.enviroment.reset()
        self.scoreRequirement = 100
        self.initialGames = 50 
        self.trainingDataFileName = 'saved.npy'
        self.logActionsFileName = 'actionsLog.txt'
    
    def BuildTrainingData(self):
        if os.path.isfile(self.trainingDataFileName):
            print('Using previously data trained')
            return np.load(self.trainingDataFileName)
        
        print('Training data...')
        trainingData = []
        acceptedScores = []
        while(len(acceptedScores) < self.initialGames):
            score = 0
            gameMemory = []
            previousObservation = [] 

            while True: 
                action = self.enviroment.action_space.sample()
                observation, reward, done, info = self.enviroment.step(action)

                if len(previousObservation) > 0 :  #is not the first iteration
                    gameMemory.append([previousObservation, action]) # I've applied this action at this scenario.
                    
                previousObservation = observation
                score+=reward # how many iterations to done? 
                if done : 
                    break

            if score >= self.scoreRequirement: 
                acceptedScores.append(score)
                for data in gameMemory:
                    if data[1] == 1: # if car go to the right side...
                        output = [0,1]
                    elif data[1] == 0: # if car go to the left side..
                        output = [1,0]		
                    trainingData.append([data[0], output])	
                
            self.enviroment.reset()

        np.save(self.trainingDataFileName, np.array(trainingData))
        print('Average Accpeted score: ', mean(acceptedScores))
        print('Median accpeted score: ', median(acceptedScores))
        print(Counter(acceptedScores))
        return trainingData

    def BuildNeuralNetworkModel(self,inputSize):
        network = input_data(shape=[None, inputSize, 1], name = 'input')
        network = fully_connected(network, 8, activation = 'relu')		
        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')
        return model

    def TrainModel(self, trainingData):
        X = np.array([i[0] for i in trainingData]).reshape(-1,len(trainingData[0][0]),1)
        y = [i[1] for i in trainingData]
        model = self.BuildNeuralNetworkModel(inputSize = len(X[0]))
        model.fit({'input': X}, {'targets': y}, n_epoch=4, snapshot_step=500, run_id='cartpole')
        return model   

    def Solve(self):     
        trainingData = self.BuildTrainingData()
        model = self.TrainModel(trainingData)
        scores = []
        choices = []
        allMemory = []
        logActions = ""

        for trial in range(100):
            score = 0
            gameMemory = []
            previousObservation = []
            self.enviroment.reset()
            logActions+= "trial " + str(trial) + "\n"
            while True:
                self.enviroment.render()

                if len(previousObservation)==0:
                    action =  self.enviroment.action_space.sample()
                else:
                    action = np.argmax(model.predict(previousObservation.reshape(-1,len(previousObservation),1))[0])

                choices.append(action)      
                observation, reward, done, info = self.enviroment.step(action)
                gameMemory.append([previousObservation, action])
                logActions += "\t\t".join(("%8.5f" % e) for e in previousObservation) + "\t\t" + str(action) + "\n"
                previousObservation = observation
                score+=reward
                if done: 
                    break
            
            print('trial: {} score: {}'.format(trial,score))
            scores.append(score)

        logActionsFile = open(self.logActionsFileName, "w")
        logActionsFile.write(logActions)
        logActionsFile.close()
        print('Average Score:', mean(scores))
        print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
        print('used score requirement: {}'.format(self.scoreRequirement))

print('Starting program ....')
cartpole = Cartpole()
cartpole.Solve()