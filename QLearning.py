# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:40:00 2022

@author: Rogue
"""

import numpy as np
import pickle


class QLearning:
    def __init__(self, state_size, action_size, lr=0.03, df=0.9):
        self.lr = lr
        self.df = df
        self.q_table = np.zeros([state_size, action_size])

    # Q(st, at) = Q(st,at) + lr*(Rt1 + df * max Q(St1 , a) - Q(st,at))
    def update(self, state, action_index, reward, next_state):
        state_index = self.get_state_index(state)
        next_state_index = self.get_state_index(next_state)
        self.q_table[state_index, action_index] += self.lr * self._get_temporal_difference(state_index, action_index,
                                                                                     reward, next_state_index)

    def _get_temporal_difference(self, state_index, action_index, reward, next_state_index):
        return reward + self.df * np.max(self.q_table[next_state_index]) - self.q_table[state_index, action_index]

    def get_state_index(self, state):
        return state[0]*5 + state[1]

    def get_best_action(self, state):
        return np.argmax(self.q_table[self.get_state_index(state)])

    def save_state(self, file_name):
        with open(file_name+'_qtable.pkl', 'wb') as file:
            pickle.dump(self.q_table, file)

    def load_state(self, file_name):
        with open(file_name+'_qtable.pkl', 'rb') as file:
            self.q_table = pickle.load(file)