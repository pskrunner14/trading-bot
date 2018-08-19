import numpy as np
from tqdm import tqdm
from time import clock

from utils import get_state, format_currency, format_position



"""
Train the Agent on one step
"""

def train_model(agent, episode, data, episode_count=100, batch_size=32, window_size=10):

    data_length = len(data) - 1
    total_profit = 0
    agent.inventory = []
    avg_loss = []
    start = clock()
    state = get_state(data, 0, window_size + 1)

    for t in tqdm(range(data_length), total=data_length, leave=True, ascii=True, desc='Episode {}/{}'.format(episode, episode_count)):
        action = agent.act(state)

        # SIT
        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0

        # BUY
        if action == 1:
            agent.inventory.append(data[t])

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price

        done = True if t == data_length - 1 else False
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        if done:
            end = clock() - start
            
    if episode % 10 == 0 or episode < 10:
        agent.save(episode)
    return (episode, episode_count, total_profit, np.mean(np.array(avg_loss)), end)


def evaluate_model(agent, data, window_size=10, debug=False):

    data_length = len(data) - 1

    state = get_state(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []

    for t in range(data_length):

        action = agent.act(state, is_eval=True)

        # SIT
        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0

        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            if debug:
                print('Buy at: {}'.format(format_currency(data[t])))
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            if debug:
                print('Sell at: {} | Position: {}'.format(format_currency(data[t]), format_position(data[t] - bought_price)))

        done = True if t == data_length - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            return total_profit
