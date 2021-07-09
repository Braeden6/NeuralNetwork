"""
A Minimal Deep Q-Learning Implementation (minDQN)
Running this code will render the agent solving the CartPole environment using OpenAI gym. Our Minimal Deep Q-Network is approximately 150 lines of code. In addition, this implementation uses Tensorflow and Keras and should generally run in less than 15 minutes.
Usage: python3 minDQN.py
"""

import gym
import tensorflow as tf
import numpy as np
from tensorflow import keras

from collections import deque
import time
import random

RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v1')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 300
test_episodes = 100

class DQNAgent:
    def __init__(self, state_size, action_size, epsilon_decay, batch_size, epsilon, epsilon_min):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=50_000)
        self.model = self._build_model()
        self.targetModel = self._build_model()
        self.targetModel.set_weights(self.model.get_weights())
        self.epsilon = epsilon #1.0
        self.epsilon_min = epsilon_min #0.01
        self.epsilon_decay = epsilon_decay # 0.995
        self.batch_size = batch_size # 128

    def _build_model(self):
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = keras.Sequential()
        model.add(keras.layers.Dense(24, input_shape=self.state_size, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(12, activation='relu', kernel_initializer=init))
        model.add(keras.layers.Dense(self.action_size, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model

    def updateTarget(self):
        self.targetModel.set_weights(self.model.get_weights())

    def act(self, observation):
        if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
        predicted = self.model.predict(observation)
        return np.argmax(predicted[0])

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, done):
        learning_rate = 0.7 # Learning rate
        discount_factor = 0.618

        MIN_REPLAY_SIZE = 2000
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        mini_batch = random.sample(self.memory, self.batch_size)
        current_states = np.array([transition[0] for transition in mini_batch])
        current_qs_list = self.model.predict(current_states)
        new_current_states = np.array([transition[3] for transition in mini_batch])
        future_qs_list = self.targetModel.predict(new_current_states)

        X = []
        Y = []
        for index, (observation, action, reward, new_observation, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(observation)
            Y.append(current_qs)
        self.model.fit(np.array(X), np.array(Y), batch_size=self.batch_size, verbose=0, shuffle=True)

    def decayEpsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)






def get_qs(model, state, step):
    return model.predict(state.reshape([1, state.shape[0]]))[0]
