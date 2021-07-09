import gym
from DQNAgent import DQNAgent
import tensorflow as tf
import numpy as np


RANDOM_SEED = 5
tf.random.set_seed(RANDOM_SEED)

env = gym.make('CartPole-v1')
env.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Action Space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# An episode a full game
train_episodes = 300

model = DQNAgent(env.observation_space.shape, env.action_space.n)

steps_to_update_target_model = 0

for episode in range(train_episodes):
    total_training_rewards = 0
    observation = env.reset()
    done = False
    while not done:
        steps_to_update_target_model += 1
        if True:
            env.render()

        action = model.act(np.reshape(observation, [1,env.observation_space.shape[0]]))
            
        new_observation, reward, done, _ = env.step(action)
        model.memorize(observation, action, reward, new_observation, done)

        if steps_to_update_target_model % 4 == 0 or done:
            model.train(done)

        observation = new_observation
        total_training_rewards += reward

        if done:
            print('Total training rewards: {} after n steps = {} '.format(total_training_rewards, episode))
            total_training_rewards += 1

            if steps_to_update_target_model >= 100:
                print('Copying main network weights to the target network weights')
                steps_to_update_target_model = 0
                model.updateTarget()
            break
    model.decayEpsilon()
env.close()

