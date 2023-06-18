import DQN.Minimax as Minimax
from DQN.Othello import Othello
from DQN.DeepQNetwork import *
import Agents as ag

VERSION = 1500

def runAgainstMinimax(agent:Agent, DQNfirst:bool):
    env = Environment(Othello())
    
    state = env.env.reset()
    # s = process(state)
    total_reward = 0
    step = 0
    
    DQNturn = DQNfirst
    
    while True:
        if DQNturn:
            a = agent.select_move(state, 1, 60)
            # print('step', step + 1, ': DQN choose', a)
        else:
            a = Minimax.select_move(state, 1, 60)
            # print('step', step + 1, ': MNM choose', a)
            
        new_state, r, done, info = env.env.step(state, a)
        
        
        total_reward += r
        # print('r:', r)
        
        if done:
            break
        state = reverse(new_state)
        s = process(state)
        DQNturn = not DQNturn
        step += 1
        # print('step:', step)
    # print("Total reward:", total_reward)
    if DQNturn:
        result = total_reward
    else:
        result = 1 - total_reward
    
    if result == 1: return 'Win'
    if result == 0.5: return 'Draw'
    else: return 'Lose'

def runAgainstRandom(agent:Agent, DQNfirst:bool):
    env = Environment(Othello())
    
    state = env.env.reset()
    # s = process(state)
    total_reward = 0
    step = 0
    
    DQNturn = DQNfirst
    
    while True:
        if DQNturn:
            a = agent.select_move(state, 1, 60)
            # a = ag.random_agent(state, 1, 60)
            # print('step', step + 1, ': DQN choose', a)
        else:
            a = ag.random_agent(state, 1, 60)
            # print('step', step + 1, ': RDM choose', a)
            
        new_state, r, done, info = env.env.step(state, a)
        
        
        total_reward += r
        # print('r:', r)
        
        if done:
            break
        state = reverse(new_state)
        s = process(state)
        DQNturn = not DQNturn
        step += 1
        # print('step:', step)
    # print("Total reward:", total_reward)
    if DQNturn:
        result = total_reward
    else:
        result = 1 - total_reward
    
    if result == 0: return 'Lose'
    if result == 0.5: return 'Draw'
    else: return 'Win'




if __name__ == "__main__":
    state_count = (BOARD_SIZE, BOARD_SIZE, BOARD_STACK)
    action_count = ACTIONS

    agent = Agent(state_count, action_count)
    agent.brain.load(WEIGHTS_PATH + "Othello_%d" % (VERSION))
    
    win = 0
    draw = 0
    lose = 0
    total_episodes = 100
    for i in range(total_episodes):
        # result = runAgainstMinimax(agent, True)
        result = runAgainstRandom(agent, False)
        print('Game', i + 1, ':', result)
        if result == 'Win': win += 1
        elif result == 'Draw': draw += 1
        else: lose += 1
        
    print('Wins / Ties / Losses ratio:', 
                  '{:.1%}'.format(win/total_episodes), "/", 
                  '{:.1%}'.format(draw/total_episodes), "/", 
                  '{:.1%}'.format(lose/total_episodes), "/", 
                )










