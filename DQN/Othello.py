import copy, time
import numpy as np



class Othello:
    def __init__(self, player=1):
        self.state = [[0 for i in range(8)] for j in range(8)]
        self.player = player
        self.reset()

    def reset(self):
        state = [[0 for i in range(8)] for j in range(8)]
        state[3][3] = -1
        state[3][4] = 1
        state[4][4] = -1
        state[4][3] = 1
        self.state = state
        
        return copy.deepcopy(self.state)
        
    def step(self, state, action, player=1):
        '''Returns new_state, reward, done, info.
        Only use for training the player=1 with itself.'''
        self.state = state
        reward = 0
        done = False
        
        if action is None: # This player has no valid action
            if Othello.get_valid_actions(self.state, -player) == []: # ... and so does the opponent
                done = True
                reward = self.get_reward()
                return copy.deepcopy(self.state), reward, done, {}
            else:
                return copy.deepcopy(self.state), reward, done, {}

        row = action[0]; col = action[1]
        if not Othello.is_valid_action(self.state, action, player):
            print("Invalid move: (",row ,col ,")") 
            done = True
            reward = 0
            return None, reward, done, {}
        else:
            self.state = Othello.get_new_state(self.state, action, player)
            return copy.deepcopy(self.state), reward, done, {}
                
    def get_reward(self, player=1):
        # if player == 1:
        #     return np.sum(self.state)
        # else:
        #     return -np.sum(self.state)        
        S = np.sum(self.state)
        if S == 0:
            return 0.5
        return float(not ((player == 1) ^ (S > 0)))
        
        
# Utilities        
    @staticmethod
    def is_valid_action(state, action, player):
        if action is None:
            return False
        
        row = action[0]; col = action[1]
        if state[row][col] != 0:
            return False
        
        # iterate over 8 directions
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                r = row + i
                c = col + j
                found_opponent = False
                while r >= 0 and r < 8 and c >= 0 and c < 8:
                    if state[r][c] == 0: break
                    
                    elif state[r][c] == player:
                        if found_opponent: return True
                        break
                    
                    else: # state[r][c] = opponent
                        found_opponent = True
                        r += i
                        c += j
                        
        return False        
                        
    @staticmethod
    def get_valid_actions(state, player=1):
        '''Returns list of actions (i,j).'''
        valid_actions = []
        for i in range(8):
            for j in range(8):
                if Othello.is_valid_action(state, action=(i,j), player=player):
                    valid_actions.append((i, j))
        return valid_actions
    
    @staticmethod
    def get_new_state(state, action, player):
        '''Only use this method if action is valid.'''
        new_state = copy.deepcopy(state)
        row = action[0]; col = action[1]
        
        new_state[row][col] = player
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                r = row + i
                c = col + j
                flipped = False
                to_flip = []
                
                while r >= 0 and r < 8 and c >= 0 and c < 8:
                    if new_state[r][c] == 0: break
                    
                    elif new_state[r][c] == player:
                        flipped = True
                        break
                    
                    else:
                        to_flip.append((r, c))
                        r += i
                        c += j
        
                if flipped:
                    for (r, c) in to_flip:
                        new_state[r][c] = player
                        
        return new_state
        
    def get_winner(self):
        '''Returns 1, -1, 0.'''
        diff = np.sum(self.state) 
        if diff > 0: return 1
        if diff < 0: return -1  
        return 0    
    
    
    
    
    
    
    
    
    
    
    
         