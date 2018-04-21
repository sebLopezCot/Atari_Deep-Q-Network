#!/usr/bin/env python2

import gym

import keras
from keras.layers import Input, Conv2D, Dense, Flatten
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import TensorBoard

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

    def __iter__(self):
        for item in self.data:
            yield item

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class ReplayMemory(CircularQueue):

    def sample(self):
        if len(self) == 0:
            raise ValueError 
        else:
            return self[np.random.randint(len(self))]

class DQN(object):

    def __init__(self, hyperparams, tbClbk):
        self.hyperparams = hyperparams
        self.model = self.create_model(hyperparams)
        self.tbClbk = tbClbk

    def create_model(self, hyperparams):
        print "CREATING MODEL...\n"
        
        input_layer = Input(shape = (84,84,self.hyperparams['last_k_frames']))
        conv1 = Conv2D(16, (8, 8), strides=(4, 4), activation='relu', padding='same')(input_layer)
        conv2 = Conv2D(32, (4, 4), strides=(2, 2), activation='relu', padding='same')(conv1)
        flat = Flatten()(conv2)
        fc = Dense(256, activation='relu')(flat)
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

    def __init__(self, hyperparams, tbClbk): 
        print "LOADING GYM..."
        self.env = gym.make(hyperparams['gym_name'])
        print "DONE.\n"
       
        self.tbClbk = tbClbk
        hyperparams['num_actions'] = self.env.action_space.n
        self.dqn = DQN(hyperparams, self.tbClbk)
        self.hyperparams = hyperparams
        self.replay_memory = ReplayMemory(hyperparams['replay_memory_size'])
        self.obs_seq = CircularQueue(hyperparams['last_k_frames'] + 1) # + 1 to be able to keep the last sequence as well as the current
        self.action_seq = CircularQueue(hyperparams['last_k_frames'])

    @staticmethod
    def preprocess_frame(frame):
        gray = np.mean(frame, axis=2)
        downsampled = scipy.misc.imresize(gray, (110, 84), interp='bilinear')
        return downsampled[13:97,0:84]

    def get_last_obs_seq(self):
        k = self.hyperparams['last_k_frames']
        if len(self.obs_seq) < k + 1:
            raise ValueError 

        return list(self.obs_seq)[0:k]

    def get_curr_obs_seq(self):
        k = self.hyperparams['last_k_frames']
        if len(self.obs_seq) < k:
            raise ValueError
        elif len(self.obs_seq) == k:
            return list(self.obs_seq)
        else:
            return list(self.obs_seq)[1:(k+1)]

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
            print "\n\nEPISODE: ", episode
            print "================================================="

            # Reset the emulator and initialize the sequence
            initial_obs = self.env.reset()
            self.obs_seq.clear()
            self.action_seq.clear()

            self.obs_seq.push(DQNTrainer.preprocess_frame(initial_obs))
            self.action_seq.push(self.env.action_space.sample()) # just push a random action

            for t_step in range(self.hyperparams['num_time_steps']):
                if t_step % self.hyperparams['print_period'] == 0:
                    print "T STEP: ", t_step

                # With probability epsilon, select a random action, otherwise choose
                # action which maximizes the optimal value function
                chosen_action = None
                if t_step % (self.hyperparams['frame_skip_amount']+1) != 0:
                    # Just perform the last action every few timesteps for efficiency
                    if len(self.action_seq) > 0:
                        chosen_action = self.action_seq[-1]
                    else:
                        chosen_action = self.env.action_space.sample()
                elif np.random.random() < self.calc_epsilon(episode, t_step):
                    chosen_action = self.env.action_space.sample() 
                elif len(self.obs_seq) >= self.hyperparams['last_k_frames']:
                    obs_stack = np.dstack(self.get_curr_obs_seq())
                    obs_stack = np.reshape(obs_stack, (1,) + obs_stack.shape)
                    assert(obs_stack.shape == (1,84,84,4))
                    q_vals = self.dqn.model.predict(obs_stack, batch_size=1)
                    chosen_action = np.argmax(q_vals) # returns the action index
                else:
                    chosen_action = self.env.action_space.sample()

                # Execute action and observe reward and image
                obs, reward, done, info = self.env.step(chosen_action)
                if t_step % 4 == 0:
                    self.env.render()

                # Update the sequence
                self.obs_seq.push(DQNTrainer.preprocess_frame(obs))
                self.action_seq.push(chosen_action)

                # Store the transition in the replay memory
                if len(self.obs_seq) >= self.hyperparams['last_k_frames'] + 1:

                    last_seq = np.dstack(self.get_last_obs_seq())
                    curr_seq = np.dstack(self.get_curr_obs_seq())
                    self.replay_memory.push((last_seq, chosen_action, reward, curr_seq))

                    if t_step % (self.hyperparams['frame_skip_amount']+1) == 0:
                        # Sample random minibatch of transitions from replay memory
                        minibatch = [self.replay_memory.sample() for j in range(self.hyperparams['batch_size'])]

                        # Calculate target values for loss
                        y_js = np.zeros(self.hyperparams['batch_size'])
                        psi_js = np.array(map(lambda (psi, a, r, psi_next): psi, minibatch))
                        gamma = self.hyperparams['discount_factor']
                        for j in range(len(minibatch)):
                            seq_j, action_j, reward_j, seq_j_next = minibatch[j]
                            seq_j_next_shaped = np.reshape(seq_j_next, (1,) + seq_j_next.shape) # one sample to load into DQN
                            q_max_val = np.max(self.dqn.model.predict(seq_j_next_shaped, batch_size=1))
                            bellman_term = 0.0 if done else gamma*q_max_val
                            y_js[j] = reward_j + bellman_term 

                        y_js = np.reshape(np.repeat(y_js, self.hyperparams['num_actions']), (self.hyperparams['batch_size'], self.hyperparams['num_actions']))

                        #hist = self.dqn.model.fit(x=psi_js, y=y_js, batch_size=self.hyperparams['batch_size'], callbacks=[self.tbClbk])
                        hist = self.dqn.model.fit(x=psi_js, y=y_js, batch_size=self.hyperparams['batch_size'], verbose=1)

                # Finish this episode if it is our last
                if done:
                    break

    def evaluate(self):
        pass

if __name__ == "__main__":

    hyperparams = {
        'gym_name':             'Breakout-v0',
        'num_episodes':         10000,
        'num_time_steps':       10000,
        'replay_memory_size':   100000,
        'last_k_frames':        4,
        'batch_size':           64,
        'exploration_initial':  1.0,
        'exploration_final':    0.1,
        'exploration_period':   1000000,
        'frame_skip_amount':    3,
        'discount_factor':      0.99,
        'print_period':         10
    }

    print "\nUSING THE FOLLOWING HYPERPARAMS:\n"
    for k,v in hyperparams.items():
        print "\'", k, "\': ", v
    print "\n"

    tbClbk = TensorBoard(log_dir='./tensorboard', histogram_freq=0)

    trainer = DQNTrainer(hyperparams, tbClbk)
    print "\n\n---------- BEGINNING TRAINING! ----------\n\n"
    trainer.train()




