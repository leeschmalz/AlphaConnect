import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
from stable_baselines3 import PPO
from aux_functions import *
import random

class Env(gym.Env):
    def __init__(self, verbose=0, opponent='random',veritical_win_reward=-1,exploration_reward=1,block_vertical_win_reward=5,flatten=True):
        super(Env, self).__init__()

        self.verbose = verbose
        self.opponent = opponent
        self.veritical_win_reward = veritical_win_reward
        self.exploration_reward = exploration_reward
        self.block_vertical_win_reward = block_vertical_win_reward
        self.flatten = flatten
        self.n_moves = 0

        #  actions: play in any of 7 columns
        self.action_space = spaces.Discrete(7)

        if self.flatten:
            # observation space
            self.observation_space = spaces.Box(low=0, 
                                                high=1, 
                                                shape=(7*6,), # flatten the 7x6 board
                                                dtype=np.float32)
        else:
            # observation space
            self.observation_space = spaces.Box(low=0,
                                                high=1,
                                                shape=(7,6),
                                                dtype=np.float32)
            

    def step(self, action):
        valid = check_valid_action(self.board,action)
        self.n_moves += 1
        # if not valid action, penalize and end game immediately
        if not valid:
            if self.flatten:
                observation = self.board.flatten()
            else:
                observation = self.board
            reward = -50
            done = True
            self.info['done_reason'] = 'invalid action'
            
            if self.verbose >= 1:
                print('reward',reward)
                print('done',done)
                print('info',self.info)
                print(self.board)
                print('\n')

            return observation, reward, done, self.info

        reward = 0 # no reward if game is not over

        if blocked_vertical_win(self.board,action):
            self.info['blocked_vertical_win'] = True
            reward += self.block_vertical_win_reward

        # update board
        self.board = place_piece(board=self.board,action=action,player=1)
        done, winner, win_method = check_game_over(self.board)

        if played_in_new_column(self.board,action):
            reward += self.exploration_reward
        elif played_in_new_row(self.board,action):
            reward += self.exploration_reward

        if self.verbose >= 2:
            print(f"player plays column: {action}")
            print(self.board)
            print('\n')

        if done:
            if winner == 1:
                reward = 20
                reward -= self.n_moves*0.2 # quick wins get rewarded higher
            elif winner == -1:
                reward = -20
                reward += self.n_moves*0.2 # small reward for lasting longer
            else:
                reward = 0

            observation = self.board.flatten()
            self.info['winner'] = winner
            self.info['done_reason'] = f'player {winner} wins {win_method}'

            if self.verbose >= 1:
                print('end game board:')
                print(self.board)
                print('reward',reward)
                print('done',done)
                print('info',self.info)
                print('\n')

            return observation, reward, done, self.info

        # get opponent move, opponent is player -1
        if self.opponent == 'random':
            opponent_action = get_random_valid_action(self.board)
        elif self.opponent == 'latest':
            board_opponent = invert_board(self.board)
            opponent_observation = board_opponent.flatten()
            opponent_action, _ = self.opponent_model.predict(opponent_observation)
        elif self.opponent == 'greedy':
            board_opponent = invert_board(self.board)
            opponent_action = get_greedy_action(board_opponent)
        else:
            raise Exception(f"opponent {self.opponent} not supported")

        self.board = place_piece(board=self.board,action=opponent_action,player=-1)
        done, winner, win_method = check_game_over(self.board)

        if self.flatten:
            observation = self.board.flatten()
        else:
            observation = self.board

        if self.verbose >= 2:
            print(f"opponent plays column: {opponent_action}")
            print(self.board)
            print('\n')

        if done:
            if winner == 1:
                reward = 20
                reward -= self.n_moves*0.2 # quick wins get rewarded higher
            elif winner == -1:
                reward = -20
                reward += self.n_moves*0.2 # small reward for lasting longer
            else:
                reward = 0

            self.info['winner'] = winner
            self.info['done_reason'] = f'player {winner} wins {win_method}'

            if self.verbose >= 1:
                print('end game board:')
                print(self.board)
                print('reward',reward)
                print('done',done)
                print('info',self.info)
                print('\n')

        return observation, reward, done, self.info

    def reset(self):
        if self.verbose >= 2:
            print('\n')
            print('new game')

        # initialize environment vars
        self.info = {}
        self.board = np.zeros((6,7))
        self.n_moves = 0

        # 50% chance player 1 starts
        if random.random() < 0.5:
            # randomly place a piece in column 2,3, or 4 for player -1
            action = random.choice([2,3,4])
            self.board = place_piece(board=self.board,action=action,player=-1)

            if self.verbose >= 2:
                print(f"opponent plays column: {action}")
                print(self.board)
                print('\n')

        if self.flatten:
            observation = self.board.flatten()
        else:
            observation = self.board

        if self.opponent == 'latest':
            self.opponent_model = PPO.load("./models/latest_model.zip")

        return observation