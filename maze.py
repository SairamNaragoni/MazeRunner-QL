# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 14:40:00 2022

@author: Rogue
"""
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import random_integers as rnd
import ast


class MazeEnv(object):
    def __init__(self, action_space, height=21, width=21, complexity=0.05, density=0.01, start_position="(0,0)"):
        self.width = width
        self.height = height
        self.action_space = action_space

        self.maze = self.make_env(self.width, self.height, complexity, density)
        self.start_loc = ast.literal_eval(start_position)
        self.end_loc = [self.height - 2, self.width - 2]

        self.path = []
        self.path.append(self.start_loc)

    def make_env(self, width=21, height=21, complexity=.05, density=.1):
        # Only odd shapes
        shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
        # Adjust complexity and density relative to maze size
        complexity = int(complexity * (5 * (shape[0] + shape[1])))
        density = int(density * (shape[0] // 2 * shape[1] // 2))
        # Build actual maze
        Z = np.zeros(shape, dtype=bool)
        # Fill borders
        Z[0, :] = Z[-1, :] = 1
        Z[:, 0] = Z[:, -1] = 1
        # Make isles
        for i in range(density):
            x, y = rnd(0, shape[1] // 2) * 2, rnd(0, shape[0] // 2) * 2
            Z[y, x] = 1
            for j in range(complexity):
                neighbours = []
                if x > 1:
                    neighbours.append((y, x - 2))
                if x < shape[1] - 2:
                    neighbours.append((y, x + 2))
                if y > 1:
                    neighbours.append((y - 2, x))
                if y < shape[0] - 2:
                    neighbours.append((y + 2, x))
                if len(neighbours):
                    y_, x_ = neighbours[rnd(0, len(neighbours) - 1)]
                    if Z[y_, x_] == 0:
                        Z[y_, x_] = 1
                        Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                        x, y = x_, y_
        return Z

    def sample_action(self):
        return random.choice(self.action_space)

    def show_path(self, points, color='r', name='resources/curr', stay_awake=False):
        points = np.array(points)
        plt.imshow(self.maze, cmap=plt.cm.binary, interpolation='nearest')
        plt.scatter(points[:, 1], points[:, 0], s=7, c=color)
        plt.savefig(name + '_maze_path.png')
        if not stay_awake:
            plt.close()

    def reset(self):
        self.path = []
        self.path.append(self.start_loc)
        return self.start_loc

    def take_action(self, current_state, action):
        x, y = current_state
        if action[0] == 'left':
            nx, ny = x, y - 1
        elif action[0] == 'right':
            nx, ny = x, y + 1
        elif action[0] == 'up':
            nx, ny = x + 1, y
        else:
            nx, ny = x - 1, y

        return [nx, ny] if 0 < nx < self.height and 0 < ny < self.width and not self.maze[nx][ny] else [x, y]

    def step(self, action):
        current_state = self.path[len(self.path) - 1]
        next_state = self.take_action(current_state, action)
        self.path.append(next_state)
        done = True if next_state == self.end_loc else False
        # terminated = True if next_state == current_state and next_state != self.end_loc else False
        terminated = False
        reward = -1
        reward = -10 if current_state == next_state else reward
        reward = 20 if done else reward
        return next_state, reward, done, terminated

    def save_state(self, file_name):
        with open(file_name+'_maze.pkl', 'wb') as file:
            pickle.dump(self.maze, file)

    def load_state(self, file_name):
        with open(file_name+'_maze.pkl', 'rb') as file:
            self.maze = pickle.load(file)

