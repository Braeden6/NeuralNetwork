import gym
import sys
sys.path.insert(1, "C://Users//bnorm//Desktop//repos//NeuralNetwork")
from DQNAgent import DQNAgent
import tensorflow as tf
import numpy as np

def trainGym(render = True):
    RANDOM_SEED = 5
    tf.random.set_seed(RANDOM_SEED)

    env = gym.make('CartPole-v1')
    env.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Action Space: {}".format(env.action_space))
    print("State space: {}".format(env.observation_space))

    # An episode a full game
    train_episodes = 1000
    REWARDS = []
    model = DQNAgent(env.observation_space.shape, env.action_space.n, 0.995, 128, 1, 0.01)

    steps_to_update_target_model = 0

    for episode in range(train_episodes):
        total_training_rewards = 0
        observation = env.reset()
        done = False
        while not done:
            steps_to_update_target_model += 1
            if render:
                env.render()
            action = model.act(np.reshape(observation, [1,env.observation_space.shape[0]]))
            new_observation, reward, done, _ = env.step(action)
            model.memorize(observation, action, reward, new_observation, done)
            if steps_to_update_target_model % 4 == 0 or done:
                model.train(done)
            observation = new_observation
            total_training_rewards += reward

            if done:
                print('Total training rewards: {} after n steps = {} epsilon = {} '.format(total_training_rewards, episode, model.epsilon))
                REWARDS.append(total_training_rewards)
                total_training_rewards += 1
                break
        model.decayEpsilon()


   
    env.close()
    return model

if __name__ == "__main__":
    model = trainGym(False)
    '''
    memory = model.memory
    model.save('saved/gym1/model.hz')
    envfile = open("saved/gym1/env.pkl","wb")
    pickle.dump(memory, envfile)
    pickle.dump(REWARDS, envfile)
    envfile.close()
    plt.plot(REWARDS)
    plt.show()'''

    '''
    train_episodes = 0
    env = 0


    for episode in range(train_episodes):
        # reset game state
        state = env.reset()
        done = False
        while not done:
            # make prediction
            action = model.act(state)
            # update game state
            new_state, reward, done = env.step(action)
            # remember <s,a,r,s'> for training 
            model.memorize(state, action, reward, new_state)
            # train on subset of memorized data
            model.train()
            state = new_state
        model.decayEpsilon()


mini_batch = 0
α = 0
λ = 0
new_value = 0
def Q(x,y):
    return
X = 0
Y = 0

for (state, action, reward, new_state, done) in mini_batch:
    if not done:
        new_value[action] = (1 - α) * Q(state, action) + α * (reward + λ * max(Q(new_state, action)))
    else:
        new_value[action] = (1 - α) * Q(state, action) + α * (reward)
    X.append(state)
    Y.append(new_value)
    '''