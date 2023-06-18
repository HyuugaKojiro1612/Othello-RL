import random, math, copy
import numpy as np
import pandas as pd
import tensorflow as tf

from DQN.DeepQNetwork import *
from os.path import exists
from sqlitedict import SqliteDict
from matplotlib import pyplot as plt

START_EPISODE = 1500
'''Set to -1 if training the model anew'''
TOTAL_EPISODE = 200
SAVE_FREQUENCY = 20

WEIGHTS_PATH = 'D:\\Othello\\DQN\\weights\\'
'''Path to the folder storing weights file.'''

MEMORY_PATH = 'D:\\Othello\\DQN\\memory\\'
'''Path to the folder storing memory file.'''

def saveWeights(agent: Agent, path, episode_number):
    agent.brain.save(path + "Othello_%d" % (episode_number))
    
def loadWeights(agent: Agent, path, episode_number):
    agent.brain.load(path + "Othello_%d" % (episode_number))
    

def saveMemory(key, value, cache_file=MEMORY_PATH + "memory.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value # Using dict[key] to store
            mydict.commit() # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)
 
def loadMemory(key, cache_file=MEMORY_PATH + "memory.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key] # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)
        
def saveRatio(key, value, cache_file=MEMORY_PATH + "ratio.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value # Using dict[key] to store
            mydict.commit() # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)
        
def loadRatio(key, cache_file=MEMORY_PATH + "ratio.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key] # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)

def saveSteps(key, value, cache_file=MEMORY_PATH + "steps.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            mydict[key] = value # Using dict[key] to store
            mydict.commit() # Need to commit() to actually flush the data
    except Exception as ex:
        print("Error during storing data (Possibly unsupported):", ex)
        
def loadSteps(key, cache_file=MEMORY_PATH + "steps.sqlite3"):
    try:
        with SqliteDict(cache_file) as mydict:
            value = mydict[key] # No need to use commit(), since we are only loading data!
        return value
    except Exception as ex:
        print("Error during loading data:", ex)


def memoryInitialize():
    env = Environment(Othello())
    randomAgent = RandomAgent(ACTIONS)
    
    if not exists(MEMORY_PATH + "memory.sqlite3"):
        print("Memory initialization with random agent...")
        while randomAgent.exp < MEMORY_CAPACITY:
            env.runAgainstMinimax(randomAgent)
            print(randomAgent.exp, "/", MEMORY_CAPACITY)
            
        saveMemory("memory_key", randomAgent.memory)
    
    randomAgent.memory = loadMemory("memory_key")
    print((len(randomAgent.memory.tree.data)))
    data = randomAgent.memory.tree.data
    total_rewards = pd.Series([data[i][2] for i in range(len(data)) if np.array_equal(data[i][3], None)])
    
    print(len(total_rewards))
    total_rewards.hist()
    plt.show()
    
def train():
    env = Environment(Othello())

    state_count = (BOARD_SIZE, BOARD_SIZE, BOARD_STACK)
    action_count = ACTIONS

    agent = Agent(state_count, action_count)
    randomAgent = RandomAgent(action_count)

    if not exists(MEMORY_PATH + "memory.sqlite3"):
        print("Memory initialization with random agent...")
        while randomAgent.exp < MEMORY_CAPACITY:
            env.runAgainstMinimax(randomAgent)
            print(randomAgent.exp, "/", MEMORY_CAPACITY)
            
        saveMemory("memory_key", randomAgent.memory)
        
    agent.memory = loadMemory("memory_key")
    randomAgent = None
    
    if exists(MEMORY_PATH + "ratio.sqlite3"):
        env.wins, env.ties, env.losses = loadRatio("ratio_key")
        
    if exists(MEMORY_PATH + "steps.sqlite3"):
        agent.steps = loadSteps("steps_key")
    print("Agent continues training from step:", agent.steps)
    agent.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * agent.steps)
    print("Epsilon is now: ", agent.epsilon)
    
    if START_EPISODE != -1:
        loadWeights(agent, WEIGHTS_PATH, START_EPISODE)
        
    for i in range(START_EPISODE, START_EPISODE + TOTAL_EPISODE):
        env.runAgainstMinimax(agent)
        if i % (5) == 0:
            total_episodes = env.wins + env.ties + env.losses
            print('Wins / Ties / Losses ratio:', 
                  '{:.1%}'.format(env.wins/total_episodes), "/", 
                  '{:.1%}'.format(env.ties/total_episodes), "/", 
                  '{:.1%}'.format(env.losses/total_episodes), "/", 
                )
            print("Agent continues training from step:", agent.steps)
            agent.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * agent.steps)
            print("Epsilon is now: ", agent.epsilon)
            
        if i % SAVE_FREQUENCY == 0:
            saveWeights(agent, WEIGHTS_PATH, i)
            saveMemory("memory_key", agent.memory)
            saveRatio("ratio_key", (env.wins, env.ties, env.losses))
            saveSteps("steps_key", agent.steps)
            
    total_episodes = env.wins + env.ties + env.losses
    print('Wins / Ties / Losses ratio:', 
            '{:.1%}'.format(env.wins/total_episodes), "/", 
            '{:.1%}'.format(env.ties/total_episodes), "/", 
            '{:.1%}'.format(env.losses/total_episodes), "/", 
        )
    print("Agent stops training from step:", agent.steps)
    agent.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * agent.steps)
    print("Epsilon is now: ", agent.epsilon)
    
    saveWeights(agent, WEIGHTS_PATH, i)
    saveMemory("memory_key", agent.memory)
    saveRatio("ratio_key", (env.wins, env.ties, env.losses))
    saveSteps("steps_key", agent.steps)
    

if __name__ == "__main__":
    
    # memoryInitialize()
    
    train()
            
            
            
            
    
        
    


        
        
        
        
        
        
        
        