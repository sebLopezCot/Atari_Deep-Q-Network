#!/usr/bin/env python2

import gym

import keras
from keras.layers import Input, Conv2D, Dense
from keras.models import Model
from keras.optimizers import RMSprop, SGD

import matplotlib.pyplot as plt
import numpy as np
import scipy

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
        hyperparams['num_actions'] = self.env.action_space.n
        self.dqn = DQN(hyperparams)
        self.hyperparams = hyperparams
        self.replay_memory = ReplayMemory(hyperparams['replay_memory_size'])
        self.obs_seq = CircularQueue(hyperparams['last_k_frames'])
        self.action_seq = CircularQueue(hyperparams['last_k_frames'])
        
        print "LOADING GYM..."
        self.env = gym.make(hyperparams['gym_name'])
        print "DONE.\n"

    @staticmethod
    def preprocess_frame(frame):
        gray = np.mean(frame, axis=2)
        downsampled = scipy.misc.imresize(gray, (110, 84), interp='bilinear')
        return downsampled[13:97,0:84]

    def calc_epsilon(self, episode, t_step):
        iteration = episode * self.hyperparams['num_time_steps'] + t_step
        cutoff_point = self.hyperparams['exploration_period'] * 1.0
        initial_eps = self.hyperparams['exploration_initial']
        final_eps = self.hyperparams['exploration_final']

        if iteration < cutoff_point:
            slope = (final_eps - initial_eps) / cutoff_point
            return initial_eps + slope * iteration
        else:
            return final_eps

    def train(self):
        for episode in range(self.hyperparams['num_episodes']):
            
            # Reset the emulator and initialize the sequence
            initial_obs = self.env.reset()
            self.obs_seq.clear()
            self.action_seq.clear()

            self.obs_seq.push(DQNTrainer.preprocess_frame(initial_obs))
            self.action_seq.push(self.env.action_space.sample()) # just push a random action
            self.env.render()

            for t_step in range(self.hyperparams['num_time_steps']):
                # With probability epsilon, select a random action, otherwise choose
                # action which maximizes the optimal value function
                chosen_action = None
                if np.random.random() < self.calc_epsilon(episode, t_step):
                    chosen_action = np.random.randint(self.hyperparams['num_actions'])
                else:
                    chosen_action = ... # TODO

                # Execute action and observe reward and image
                obs, reward, done, info = self.env.step(chosen_action)
                #self.env.render()

                # Update the sequence
                self.obs_seq.push(DQNTrainer.preprocess_frame(obs))
                self.action_seq.push(chosen_action)

                # Store the transition in the replay memory
                self.replay_memory((self.obs_seq[-2], chosen_action, reward, self.obs_seq[-1]))

                # Sample random minibatch of transitions from replay memory
                minibatch = [self.replay_memory.sample() for j in range(self.hyperparams['batch_size'])]

                # Calculate target values for loss
                y_js = np.zeros(self.hyperparams['batch_size'])
                for obs_j, action_j, reward_j, obs_j_plus_one in minibatch:
                    pass # TODO

                hist = self.dqn.fit(x=TODO, y=y_js, batch_size=self.hyperparams['batch_size'])




    def evaluate(self):
        pass

if __name__ == "__main__":

    hyperparams = {
        'gym_name':             'Breakout-v0',
        'num_episodes':         100,
        'num_time_steps':       1000,
        'replay_memory_size':   100,
        'last_k_frames':        4,
        'batch_size':           32,
        'exploration_initial':  1.0,
        'exploration_final':    0.1,
        'exploration_period':   10000
    }

    print "\nUSING THE FOLLOWING HYPERPARAMS:\n"
    for k,v in hyperparams.items():
        print "\'", k, "\': ", v
    print "\n"

    trainer = DQNTrainer(hyperparams)




