'''
Credits: 
https://github.com/mswang12/minDQN/blob/main/minDQN.py
https://github.com/tychovdo/PacmanDQN


Possible:
https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial#training_the_agent

'''

import tensorflow as tf
import numpy as np
from tensorflow import keras
from collections import deque
import time
import random

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay, batch_size, epsilon, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10_000)
        self.model = self._build_model()
        self.epsilon = epsilon #1.0
        self.epsilon_min = epsilon_min #0.01
        self.epsilon_decay = epsilon_decay # 0.995
        self.batch_size = batch_size # 128

    def _build_model(self):
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(30, input_shape=self.state_size, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(30, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model

    def act(self, observation):
        return np.argmax(self.act_percentages(observation))

    def act_percentages(self, observation):  
        if np.random.rand() <= self.epsilon:
            result = [0.1 for _ in range(9)]
            result[random.randrange(self.action_size)] = 1
            return result
        return self.model.predict(observation)[0]

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, done):
        learning_rate = 0.7
        discount_factor = 0.618

        MIN_REPLAY_SIZE = 2000
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        old_values = self.model.predict(current_states)
        next_state = np.array([transition[3] for transition in mini_batch])
        future_values = self.model.predict(next_state)

        X = []
        Y = []
        for index, (state, action, reward, _, done) in enumerate(mini_batch):
            if not done:
                optimal_future_value = np.max(future_values[index])
                learned_value = reward + discount_factor * optimal_future_value
            else:
                learned_value = reward

            # Q(s_t, a_t)
            old_value = old_values[index]
            new_value = old_value
            # Q^new(s_t, a_t)
            new_value[action] = (1 - learning_rate) * old_value[action] + learning_rate * learned_value

            X.append(state)
            Y.append(new_value)
        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)


    def decayEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
