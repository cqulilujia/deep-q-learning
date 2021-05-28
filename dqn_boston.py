# -*- coding: utf-8 -*-
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from matplotlib.animation import FuncAnimation
import pandas as pd
import matplotlib.pyplot as plt


EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            print("1", target_f)
            target_f[0][action] = target
            print("2", target_f)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

ORIGIN_POINT = 61358091
DESTINATION_POINT = 61339761


class ENV:
    def __init__(self):
        self.action_size = 8
        self.observation_size = 256
        self.origin_point = ORIGIN_POINT
        self.destination_point = DESTINATION_POINT
        self.current_point = self.origin_point

    def reset(self):
        self.origin_point = ORIGIN_POINT
        self.current_point = self.origin_point
        state = 1
        return state

    def step(self, action):
        next_state = 2
        reward = 100
        done = 0
        _ = 0
        return next_state, reward, done, _


    def animate(i):
        # test_animation.csv这个表每1秒写入一个数字。
        data = pd.read_csv("test_animation.csv", names=['val'])
        idx = data.index
        val = data['val']
        plt.cla()
        plt.plot(idx, val, marker='o')


    def render(self):
        ani = FuncAnimation(plt.gcf(), self.animate, interval=2)
        plt.show()


if __name__ == "__main__":
    env = ENV()
    state_size = env.observation_size
    action_size = env.action_size
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 32


    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")