import sys
import os

# Faster computation on CPU (only if using tensorflow-gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from tqdm import tqdm
from time import clock
import numpy as np

from IPython.display import clear_output

from agent.agent import Agent
from utils import get_stock_data, get_state, format_currency, format_position

JUPYTER = False

if JUPYTER:
    stock_name = input('Enter stock: ')
    window_size = int(input('Enter window size: '))
    episode_count = int(input('Enter episode count: '))
    model_name = input('Enter model name if any (otherwise leave blank): ')
    pretrained = bool(int(input('Pretrained model? (0/1)')))
else:    
    if len(sys.argv) < 6:
        print('Usage: python train.py [stock] [window] [episodes] [model] [pretrained (0/1)]')
        exit(0)
    stock_name, window_size, episode_count, model_name, pretrained = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], bool(int(sys.argv[5]))

agent = Agent(window_size, pretrained=pretrained, model_name=model_name)
data = get_stock_data(stock_name)
data_length = len(data) - 1
batch_size = 50

results = []

def print_result(episode, episode_count, position, avg_mse, time):
    print('Episode {}/{} - Position: {}  MSE: {:.4f}  (~{:.4f} secs)'.format(episode, episode_count, format_position(position), avg_mse, time))

for episode in range(1, episode_count + 1):
    
    if JUPYTER:
        clear_output()
    start = clock()

    state = get_state(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    avg_mse = []

    for t in tqdm(range(data_length), total=data_length, leave=False, desc='Episode {}/{}'.format(episode, episode_count)):

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
            mse = agent.train_experience_replay(batch_size)
            avg_mse.append(mse)

        if done:
            end = clock() - start
            print_result(episode, episode_count, total_profit, np.mean(np.array(avg_mse)), end)

    if episode % 10 == 0:
        agent.save(episode)