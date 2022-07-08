
from tabnanny import verbose
import numpy as np
import pandas as pd
import random

def invert_board(board):
    return -1*board

def place_piece(board,action,player):
    for i in range(6):
        if board[i,action] != 0: # if a chip is there, play above it
            board[i-1,action] = player
            break
    else:
        board[5,action] = player

    return board

def check_valid_action(board,action):    
    # check if column is full
    if all(board[:,action] != 0):
        valid = False
    else:
        valid = True

    return valid

def get_random_valid_action(board):
    valid_action = False
    while not valid_action:
        action = random.randint(0,6)
        valid_action = check_valid_action(board,action)
    return action

def win_possible(board):
    # check if win is possible this looks at the board from player 1's perspective
    # horizontal
    for i in range(6):
        for j in range(4):
            segment = board[i,j:j+4]
            # check if segment has a win spot available
            if sum(segment == 1) == 3 and sum(segment == 0) == 1:
                rel_winning_col_index = np.where(segment == 0)[0][0]

                # check if the winning spot is in the bottom row
                if i == 5:
                    print('horiz')
                    return True, j+np.where(segment == 0)[0][0]

                # check if the space below the win is not empty
                if board[i+1,j+rel_winning_col_index] != 0:
                    # return True, winning action
                    print('horiz')
                    return True, j+np.where(segment == 0)[0][0]

    # vertical
    for j in range(7):
        for i in range(3):
            segment = board[i:i+4,j]
            if sum(segment == 1) == 3 and segment[0] == 0:
                # return True, winning action
                print('vert')
                return True, j

    # diagonal top left to bottom right
    for row in range(3):
        for col in range(4):
            segment = board[row:row+4,col:col+4].diagonal()
            # check if there is an empty space that would make a win
            if sum(segment == 1) == 3 and sum(segment == 0) == 1:
                rel_winning_col_index = np.where(segment == 0)[0][0]
                # check if the winning spot is the bottom row
                if row + rel_winning_col_index == 5:
                    print('diag1')
                    return True, col+rel_winning_col_index

                # check if there is a piece below the winning spot
                if board[row+rel_winning_col_index+1,col+rel_winning_col_index] != 0:
                    print('diag1')
                    return True, col+rel_winning_col_index
    
    # diagonal bottom left to top right
    for row in range(5, 2, -1):
        for col in range(3):
            segment = np.flip(np.diag(np.fliplr(board[row-3:row+1,col:col+4])))
            # check if there is an empty space that would make a win
            if sum(segment == 1) == 3 and sum(segment == 0) == 1:
                rel_winning_col_index = np.where(segment == 0)[0][0]
                # check if the winning spot is the bottom row
                if row == 5 and rel_winning_col_index == 0:
                    print('diag2')
                    return True, col
                # check if there is a piece below the winning spot
                if board[row-rel_winning_col_index+1,col+rel_winning_col_index] != 0:
                    print('diag2')
                    return True, col+rel_winning_col_index
    
    # if no wins were found
    return False, None
                
def get_greedy_action(board):
    '''
    take available wins, elif block opponent wins, else take random action
    '''
    win_is_possible, winning_move = win_possible(board)
    if win_is_possible: # check if I can win
        if verbose >= 2:
            print('win')
        return winning_move
    else:
        opponent_win_possible, opponent_winning_move = win_possible(invert_board(board)) # check if opponent can win
        if opponent_win_possible:
            if verbose >= 2:
                print('block')
            return opponent_winning_move
        else:
            if verbose >= 2:
                print('random')
            return get_random_valid_action(board)

def check_game_over(board):
    # return -1 if player -1 wins, 1 if player 1 wins, 0 if neither
    win_method = ''

    # horizontal
    winner = 0
    done = False
    for i in range(6):
        for j in range(4):
            if all(board[i,j:j+4] == 1):
                winner = 1
                done = True
                win_method = 'horizontal'
                break
            if all(board[i,j:j+4] == -1):
                winner = -1
                done = True
                win_method = 'horizontal'
                break

        if winner != 0:
            break
            
    # vertical
    for j in range(7):
        for i in range(3):
            if all(board[i:i+4,j] == 1):
                winner = 1
                done = True
                win_method = 'vertical'
                break
            if all(board[i:i+4,j] == -1):
                winner = -1
                done = True
                win_method = 'vertical'
                break

        if winner != 0:
            break
    
    # diagonal top left to bottom right
    for row in range(3):
        for col in range(4):
            if board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == board[row + 3][col + 3] == 1:
                winner = 1
                done = True
                win_method = 'diagonal'
                break
            if board[row][col] == board[row + 1][col + 1] == board[row + 2][col + 2] == board[row + 3][col + 3] == -1:
                winner = -1
                done = True
                win_method = 'diagonal'
                break
    
    # diagonal bottom left to top right
    for row in range(5, 2, -1):
        for col in range(3):
            if board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] == board[row - 3][col + 3] == 1:
                winner = 1
                done = True
                win_method = 'diagonal'
                break
            if board[row][col] == board[row - 1][col + 1] == board[row - 2][col + 2] == board[row - 3][col + 3] == -1:
                winner = -1
                done = True
                win_method = 'diagonal'
                break


        if winner != 0:
            break
    
    # tie
    if all(board.flatten() != 0):
        done = True

    return done, winner, win_method

def played_in_new_column(board,action):
    return sum(board[:,action] == 1) == 1

def played_in_new_row(board,action):
    for i in range(6):
        if board[i,action] == 1:
            row = i
            break

    return sum(board[row,:] == 1) == 1

def blocked_vertical_win(board,action):
    column = board[:,action]
    for i in range(6):
        if column[i] != 0:
            return (all(column[i:i+3] == -1) and len(column[i:i+3])>2)
            
    return False

    