# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:40:00 2022

@author: Rogue
"""

import json
import logging.handlers
import os
import random

from QLearning import QLearning
from maze import MazeEnv
import numpy as np

handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "logs/training_logs.log"))
handler.setFormatter(logging.Formatter("%(message)s"))
log = logging.getLogger("main")
log.setLevel(os.environ.get("LOGLEVEL", "INFO"))
log.addHandler(handler)


class Trainer:
    def __init__(self, environment, algorithm, hyperparams, save_states=False):
        self.hyperparams = hyperparams
        self.env = environment
        self.algorithm = algorithm
        self.save_states = save_states

    def sample_action(self, state):
        exploration_type = self.hyperparams['exploration_type']
        action = self.env.sample_action()
        if exploration_type == 'e-greedy':
            epsilon = self.hyperparams['e_greedy_value']
            if random.uniform(0, 1) >= epsilon:
                action_index = self.algorithm.get_best_action(state)
                action = self.env.action_space[action_index]
        return action

    def training_loop(self):
        max_episodes = self.hyperparams['max_episodes']
        max_steps_per_episode = self.hyperparams['max_steps_per_episode']
        for episode in range(1, max_episodes):
            state = self.env.reset()
            steps, total_reward = 0.0, 0.0
            done = False
            terminated = False
            while not done and not terminated and steps < max_steps_per_episode:
                action = self.sample_action(state)
                next_state, reward, done, terminated = self.env.step(action)
                self.algorithm.update(state, action[1], reward, next_state)
                steps += 1
                # episode=%f,steps=%f,reward=%f,action=%s,state=%s,done=%s,qvalue=%s
                log.info("%f;%f;%f;%s;%s;%s;%s", episode, steps, reward, action, state, done,
                         self.algorithm.q_table[self.algorithm.get_state_index(state)][action[1]])
                state = next_state
                total_reward += reward

            print("episode=%f,steps=%f,total_reward=%f,done=%s,terminated=%s" % (
                episode, steps, total_reward, done, terminated))
            if done and self.save_states:
                self.env.show_path(self.env.path, name="resources/episode-" + str(episode))
        print("Training complete")

    def evaluation(self):
        max_eval_episodes = 10
        max_steps_per_episode = self.hyperparams['max_steps_per_episode']
        for episode in range(1, max_eval_episodes):
            state = self.env.reset()
            steps, total_reward = 0.0, 0.0
            done = False
            terminated = False
            while not done and not terminated and steps < max_steps_per_episode:
                action_index = self.algorithm.get_best_action(state)
                action = self.env.action_space[action_index]
                next_state, reward, done, terminated = self.env.step(action)
                steps += 1
                state = next_state
                total_reward += reward
            print("episode=%f,steps=%f,total_reward=%f,done=%s,terminated=%s" % (
                episode, steps, total_reward, done, terminated))
            if done and self.save_states:
                self.env.show_path(self.env.path, name="resources/eval-" + str(episode))
        print("Evaluation complete")


if __name__ == '__main__':
    with open('config/env.json') as json_file:
        envparams = json.load(json_file)

    with open('config/hyperparameters.json') as json_file:
        hyperparams = json.load(json_file)

    with open('config/action_space.json') as json_file:
        action_space = list(json.load(json_file).items())

    env = MazeEnv(action_space, envparams['height'], envparams['width'], envparams['complexity'], envparams['density'],
                  envparams['start_position'])
    qlearning = QLearning((envparams['height'] * envparams['width']), len(action_space), hyperparams['learning_rate'],
                          hyperparams['discount_factor'])

    if envparams['load_model']:
        file_name = "models/" + envparams['model_name']
        print("Loading model from file: " + file_name)
        env.load_state(file_name)
        qlearning.load_state(file_name)
    else:
        env.show_path(points=[[0, 0], [0, 1]])

    trainer = Trainer(env, qlearning, hyperparams, envparams['save_states'])
    trainer.training_loop()
    trainer.evaluation()

    if envparams['save_model']:
        file_name = "models/" + envparams['model_name']
        print("Saving model to file: " + file_name)
        env.save_state(file_name)
        qlearning.save_state(file_name)
