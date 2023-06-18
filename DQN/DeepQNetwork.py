import random, math, copy, time
import numpy as np
import tensorflow as tf
from keras import backend as K
from DQN.Othello import Othello
from DQN.SumTree import SumTree 

import DQN.Minimax as Minimax

HUBER_LOSS_DELTA = 2.0

BOARD_SIZE = 8
BOARD_STACK = 2
ACTIONS = 64

def reverse(state):
    '''Reverse the opponent's state to player's state'''
    new_state = copy.deepcopy(state)
    d = {1:-1, 0:0, -1:1}
    for i in range(len(state)):
        for j in range(len(state[0])):
            new_state[i][j] = d[state[i][j]]
    return new_state

def process(state):
    '''Process a state to 2 channels.'''
    new_array = []
    pieces = [1, -1]
    
    for i in range(BOARD_STACK):
        board = []
        for j in range(BOARD_SIZE):
            row = []
            for k in range(BOARD_SIZE):
                row.append(int(state[j][k] == pieces[i]))
            board.append(row)
        new_array.append(board)
    return np.moveaxis(np.array(new_array), 0, 2)

def deprocess(s):
    '''Reverse function of process(state).'''
    array = [[0 for i in range(8)] for j in range(8)]

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            array[i][j] = s[i][j][0] - s[i][j][1]
    return array

def huber_loss(y_true, y_pred):
    err = y_true - y_pred
    
    cond = K.abs(err) <= HUBER_LOSS_DELTA
    loss2 = 0.5 * K.square(err)
    loss1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    
    loss = tf.where(cond, loss2, loss1)
    return K.mean(loss)   

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

LEARNING_RATE = 0.00025
TAU = 1e-3
    
class Brain:
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count
        
        self.model = self._createModel()
        self.model_ = self._createModel()   # target network
        
    def _createModel(self):
        model = Sequential()
        
        model.add(Conv2D(64, 3, padding="same", activation='relu', input_shape=self.state_count, data_format='channels_last'))
        model.add(Conv2D(64, 3, padding="same", activation='relu'))
        model.add(Conv2D(128, 3, padding="same", activation='relu'))
        model.add(Conv2D(128, 3, padding="same", activation='relu'))
        model.add(Conv2D(256, 3, padding="same", activation='relu'))
        model.add(Conv2D(256, 3, padding="same", activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_count, activation='linear'))
        
        opt = RMSprop(learning_rate=LEARNING_RATE)
        model.compile(loss=huber_loss, optimizer=opt)
        return model
    
    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)
    
    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)
        
    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, BOARD_SIZE, BOARD_SIZE, BOARD_STACK), target).flatten()
    
    def updateTargetModel(self):
        for target_weights, q_net_weights in zip(self.model_.weights, self.model.weights):
            target_weights.assign(TAU * q_net_weights + (1.0 - TAU) * target_weights)
            
        # self.model_.set_weights(self.model.get_weights())
        
    def save(self, s):
        self.model.save_weights(s)
        
    def load(self, s):
        self.model.load_weights(s)
        #self.default_graph.finalize()


class Memory:
    '''Store samples (s, a, r, s').'''
    e = 0.01
    a = 0.06
    
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        
    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        '''Insert sample into Sumtree.data.'''
        # p is priority of sample
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):    
        batch = []  
        segment = self.tree.total() / n
        
        for i in range(n):
            # batch is list of Tuple(idx, data)
            # data is sample
            # idx is index of p of that sample
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))
            
        return batch
    
    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


MEMORY_CAPACITY = 100000
    
BATCH_SIZE = 64
        
GAMMA = 0.99

MAX_EPSILON = 0.995
MIN_EPSILON = 0.1

EXPLORATION_STOP = 20000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000 
# START_EPISODE = 180
WEIGHTS_PATH = 'D:\\Othello\\DQN\\weights\\'

class Agent:
    steps = 0
    epsilon = MAX_EPSILON
    
    def __init__(self, state_count, action_count):
        self.state_count = state_count
        self.action_count = action_count
        
        self.brain = Brain(state_count, action_count)
        self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        '''Return the best action.'''
        state = deprocess(s)
        valid_actions = Othello.get_valid_actions(state)
        if len(valid_actions) == 0:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
            
        else:
            rewards = self.brain.predictOne(s)
            valid_rewards = [rewards[i*8 + j] for (i,j) in valid_actions]
            return valid_actions[np.argmax(valid_rewards)]
            
    def observe(self, sample):
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)
        
        # if self.steps % UPDATE_TARGET_FREQUENCY == 0:
        #     self.brain.updateTargetModel()
        
        self.brain.updateTargetModel()
            
        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def _getTargets(self, batch):
        no_state = np.zeros(self.state_count)
        
        states = np.array([ o[1][0] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])
        
        p = self.brain.predict(states)                          #Q(s, a)
        p_ = self.brain.predict(states_, target=False)          #Q(s', a)
        pTarget_ = self.brain.predict(states_, target=True)     #Q'(s', argmax_a_Q(s', a))
        
        x = np.zeros((len(batch), BOARD_SIZE,BOARD_SIZE, BOARD_STACK))
        y = np.zeros((len(batch), self.action_count))
        errors = np.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            if a == None:
                a = 0
            else: 
                a = a[0] * 8 + a[1]

            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ np.argmax(p_[i]) ]
            
            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])
        return (x, y, errors)
            
    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)
        
        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])
        
        self.brain.train(x, y)
    
    def select_move(self, cur_state, player_to_move, remain_time):
        # print('DQN is choosing action...')
        if player_to_move == -1:
            cur_state = reverse(cur_state)
        
        valid_actions = Othello.get_valid_actions(cur_state)
        if len(valid_actions) == 0:
            return None
        
        if random.random() < 0.01:
            return random.choice(valid_actions)
            
        else:
            s = process(cur_state)
            rewards = self.brain.predictOne(s)
            valid_rewards = [rewards[i*8 + j] for (i,j) in valid_actions]
            return valid_actions[np.argmax(valid_rewards)]
        
        
class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0
    
    def __init__(self, action_count):
        self.action_count = action_count
        
    def act(self, s):
        state = deprocess(s)
        valid_actions = Othello.get_valid_actions(state)
        if len(valid_actions) == 0:
            return None
        return random.choice(valid_actions)
    
    def observe(self, sample):  # in (s, a, r, s') format
        error = abs(sample[2])
        self.memory.add(error, sample)
        self.exp += 1
        
    def replay(self):
        pass
    
    # @staticmethod
    def select_move(self, cur_state, player_to_move, remain_time):
        if player_to_move == -1:
            cur_state = reverse(cur_state)
        valid_actions = Othello.get_valid_actions(cur_state)
        if len(valid_actions) == 0:
            return None
        return random.choice(valid_actions)
        
            
class Environment:
    wins = 0; ties = 0; losses = 0      # for debugging
    
    def __init__(self, problem):
        self.env = Othello()
        
    def run(self, agent:Agent):
        state = self.env.reset()
        s = process(state)
        total_reward = 0
        step = 0
        
        while True:
            a = agent.act(s)
            new_state, r, done, info = self.env.step(state, a)
            
            if done:
                s_ = None
            else:
                s_ = process(new_state)
                
            agent.observe( (s, a, r, s_) )
            agent.replay
            
            # Because the player=1 is training with itself,
            # reverse the state before the next turn.
            state = reverse(new_state)
            s = process(state)
            total_reward += r
            
            if done:
                break
            step += 1
            # print('step:', step)
        print("Total reward:", total_reward)
        if total_reward == 1: self.wins += 1
        elif total_reward == 0: self.losses += 1
        else: self.ties += 1
        
    def runAgainstMinimax(self, agent:Agent):
        state = self.env.reset()
        s = process(state)
        total_reward = 0
        step = 0
        
        DQNturn = False
        
        while True:
            if DQNturn:
                a = agent.act(s)
                # print('step', step + 1, ': DQN choose', a)
            else:
                a = Minimax.select_move(state, 1, 60)
                # print('step', step + 1, ': MNM choose', a)
                
            new_state, r, done, info = self.env.step(state, a)
            
            if done:
                s_ = None
            else:
                s_ = process(new_state)
                
            agent.observe( (s, a, r, s_) )
            agent.replay
            
            # Because the player=1 is training with itself,
            # reverse the state before the next turn.
            state = reverse(new_state)
            s = process(state)
            total_reward += r
            # print('r:', r)
            DQNturn = not DQNturn
            
            if done:
                break
            step += 1
            # print('step:', step)
        print("Total reward:", total_reward)
        if total_reward == 1: self.wins += 1
        elif total_reward == 0: self.losses += 1
        else: self.ties += 1



