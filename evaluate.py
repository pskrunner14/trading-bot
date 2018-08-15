import sys

from agent.agent import Agent
from utils import get_stock_data, get_state, format_price

JUPYTER = False

if JUPYTER:
    stock_name = input('Enter stock: ')
    window_size = int(input('Enter window size: '))
    model_name = input('Enter model name if any (otherwise leave blank): ')
else:    
    if len(sys.argv) < 3:
        print('Usage: python train.py [stock] [window] [model]')
        exit(0)
    stock_name, window_size, model_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]

agent = Agent(window_size, is_eval=True, pretrained=True, model_name=model_name)
data = get_stock_data(stock_name)
data_length = len(data) - 1
batch_size = 32

state = get_state(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

for t in range(data_length):

	action = agent.act(state)

	# SIT
	next_state = get_state(data, t + 1, window_size + 1)
	reward = 0

	# BUY
	if action == 1:
		agent.inventory.append(data[t])
		print('Buy at: {}'.format(format_price(data[t])))

	# SELL
	elif action == 2 and len(agent.inventory) > 0:
		bought_price = agent.inventory.pop(0)
		reward = max(data[t] - bought_price, 0)
		total_profit += data[t] - bought_price
		print('Sell at: {} | Position: {}'.format(format_price(data[t]), format_price(data[t] - bought_price)))

	done = True if t == data_length - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print('\n--------------------------------')
		print('Final Position: {}'.format(format_price(total_profit)))
		print('--------------------------------\n')