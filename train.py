import sys
import os

from tqdm import tqdm
from time import process_time
import numpy as np

from IPython.display import clear_output

from agent.agent import Agent
from utils import get_stock_data, get_state, format_price

JUPYTER = False

if JUPYTER:
    stock_name = input('Enter stock: ')
    window_size = int(input('Enter window size: '))
    episode_count = int(input('Enter episode count: '))
    model_name = input('Enter model name if any (otherwise leave blank): ')
    pretrained = int(input('Pretrained model? (0/1)'))
else:    
    if len(sys.argv) < 4:
        print('Usage: python train.py [stock] [window] [episodes] [model] [pretrained (0/1)]')
        exit(0)
    stock_name, window_size, episode_count, model_name, pretrained = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], int(sys.argv[5])

agent = Agent(window_size, pretrained=bool(pretrained), model_name=model_name)
data = get_stock_data(stock_name)
data_length = len(data) - 1
batch_size = 32

results = []

def print_results(results, episode_count):
    for episode, position, avg_loss, time in results:
        print('Episode {}/{} - Position: {}  Loss: {:.4f}  (~{:.4f} secs)'.format(episode, episode_count, format_price(position), avg_loss, time))

for episode in range(1, episode_count + 1):

    if JUPYTER:
        clear_output()
    else:
        os.system('cls')
    print_results(results, episode_count)
    
    start = process_time()
    state = get_state(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    avg_loss = []

    for t in tqdm(range(data_length), total=data_length, leave=False, desc='Episode {}/{}'.format(episode, episode_count)):

        action = agent.act(state)

        # SIT
        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0

        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            # print('Buy at: {}'.format(format_price(data[t])))

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            # print('Sell at: {} | Position: {}'.format(format_price(data[t]), format_price(data[t] - bought_price)))

        done = True if t == data_length - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if len(agent.memory) > batch_size:
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        if done:
            end = process_time() - start
            results.append((episode, total_profit, np.mean(np.array(avg_loss)), end))

    if episode % 20 == 0:
        agent.model.save('models/{}_{}'.format(model_name, episode))
        print('\n{} model saved!\n'.format(model_name))

