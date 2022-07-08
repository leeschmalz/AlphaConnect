import numpy as np
from stable_baselines3 import A2C
from aux_functions import place_piece, check_game_over, get_greedy_action
import time

board = np.zeros((6,7))
print(board)
done = False
while not done:
    # get user move from 0 to 6
    print('\n')
    user_action = int(input("Enter your move: "))
    print('\n')
    board = place_piece(board,user_action,-1)
    time.sleep(1)
    print(board)
    done, winner, win_method = check_game_over(board)
    if done:
        break
    print('\n')
    print('opponent is thinking...')
    print('\n')
    time.sleep(1)
    # get opponent move
    greedy_action = get_greedy_action(board)
    board = place_piece(board,greedy_action,1)
    time.sleep(1)
    done, winner, win_method = check_game_over(board)
    print(board)
    print('\n')

print(f'Player {winner} wins {win_method}.')