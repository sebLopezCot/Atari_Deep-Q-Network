#!/usr/bin/env python2

import gym

import keras
from keras.layers import Input, Conv2D, Dense
from keras.models import Model
from keras.optimizers import RMSprop, SGD

import matplotlib.pyplot as plt
import numpy as np

from collections import deque

class CircularQueue(object):

    def __init__(self, max_n):
        self.data = deque()
        self.max_n = max_n

    def push(self, value):
        if len(self.data) == self.max_n:
            self.data.popleft()

        self.data.append(value)

    def clear(self):
        self.data = deque()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ReplayMemory(CircularQueue):

    def sample(self):
        if len(self) == 0:
            return None
        else:
            return self[np.random.randint(len(self))]

class DQN(object):

    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.model = self.create_model(hyperparams)

    def create_model(self, hyperparams):
        print "CREATING MODEL...\n"
        
        input_layer = Input(shape = (84,84,4))
        conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', padding='same')(input_layer)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
        fc = Dense(256, activation='relu')(conv2)
        output = Dense(hyperparams['num_actions'], activation='linear')(fc)
        
        print "COMPILING MODEL...\n"
        
        model = Model(input_layer, output)
        model.compile(loss='mean_squared_error', optimizer=SGD())
        
        print "DONE: MODEL SUMMARY:"
        model.summary()
        
        return model

    def fit(self):
        pass

class DQNTrainer(object):

    def __init__(self, hyperparams):
        print "LOADING GYM..."
        self.env = gym.make(hyperparams['gym_name'])
        print "DONE.\n"
        self.env.reset()
        hyperparams['num_actions'] = self.env.action_space.n
        self.dqn = DQN(hyperparams)
        self.hyperparams = hyperparams
        self.replay_memory = ReplayMemory(hyperparams['replay_memory_size'])

    def preprocess_sequence(self):
        pass

    def train(self):
        for epoch in range(self.hyperparams['num_episodes']):
            for t_step in range(self.hyperparams['num_time_steps']):
                pass

    def evaluate(self):
        pass

if __name__ == "__main__":

    hyperparams = {
        'gym_name':             'Breakout-v0',
        'num_episodes':         100,
        'num_time_steps':       1000,
        'replay_memory_size':   100,
        'batch_size':           32
    }

    print "\nUSING THE FOLLOWING HYPERPARAMS:\n"
    for k,v in hyperparams.items():
        print "\'", k, "\': ", v
    print "\n"

    trainer = DQNTrainer(hyperparams)




