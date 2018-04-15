import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym
from statistics import median, mean
from collections import Counter
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class Cartpole:
    def __init__(self):
        self.learnRate = 1e-3
        self.enviroment = gym.make('CartPole-v0')
        self.enviroment.reset()
        self.scoreRequirement = 100
        self.initialGames = 50 
    
    def InitialPopulation(self):
        trainingData = []
        scores = []
        acceptedScores = []
        while(len(acceptedScores) < self.initialGames):
            score = 0
            gameMemory = []
            previousObservation = [] 

            done = False
            while not done: 
                action = self.enviroment.action_space.sample()
                observation, reward, done, info = self.enviroment.step(action)

                if len(previousObservation) > 0 :  #is not the first iteration
                    gameMemory.append([previousObservation, action]) # I've applied this action at this scenario.
                    
                previousObservation = observation
                score+=reward # how many iterations to done? 
                if done : 
                    break

            if score >= self.scoreRequirement : 
                acceptedScores.append(score)
                for data in gameMemory:
                    if data[1] == 1: # if car go to the right side...
                        output = [0,1]
                    elif data[1] == 0: # if car go to the left side..
                        output = [1,0]		
                    trainingData.append([data[0], output])	
                
            self.enviroment.reset()
            scores.append(score)

        trainingDataSave = np.array(trainingData)
        np.save('saved.npy', trainingDataSave)

        print('Average Accpeted score: ', mean(acceptedScores))
        print('Median accpeted score: ', median(acceptedScores))
        print(Counter(acceptedScores))
        return trainingData

    def BuildNeuralNetworkModel(self,inputSize):
        network = input_data(shape=[None, inputSize, 1], name = 'input')

        network = fully_connected(network, 8, activation = 'relu')		

        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=self.learnRate,
                            loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')

        return model

    def TrainModel(self,training_data, model=False):

        X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
        y = [i[1] for i in training_data]

        if not model:
            model = self.BuildNeuralNetworkModel(inputSize = len(X[0]))

        model.fit({'input': X}, {'targets': y}, n_epoch=4, snapshot_step=500, run_id='cartpole')
        return model   

    def Solve(self):     
        trainingData = self.InitialPopulation()
        model = self.TrainModel(trainingData)

        scores = []
        choices = []
        for each_game in range(100):
            score = 0
            game_memory = []
            prev_obs = []
            self.enviroment.reset()
            done = False
            while not done:
                self.enviroment.render()

                if len(prev_obs)==0:
                    action =  self.enviroment.action_space.sample()
                else:
                    action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

                choices.append(action)
                        
                new_observation, reward, done, info = self.enviroment.step(action)
                prev_obs = new_observation
                game_memory.append([new_observation, action])
                score+=reward
                if done: break
            
            print('trial: {} score: {}'.format(each_game,score))
            scores.append(score)

        print('Average Score:',sum(scores)/len(scores))
        print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
        print(self.scoreRequirement)

cartpole = Cartpole()
cartpole.Solve()