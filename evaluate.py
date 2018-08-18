import sys
import os

# Faster computation on CPU (only if using tensorflow-gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from agent.agent import Agent
from utils import get_stock_data, get_state, format_position, format_currency

def evaluate_model(model_name):

	agent = Agent(window_size, is_eval=True, pretrained=True, model_name=model_name)
	data = get_stock_data(stock_name)
	data_length = len(data) - 1

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
			# print('Buy at: {}'.format(format_currency(data[t])))
		# SELL
		elif action == 2 and len(agent.inventory) > 0:
			bought_price = agent.inventory.pop(0)
			reward = max(data[t] - bought_price, 0)
			total_profit += data[t] - bought_price
			# print('Sell at: {} | Position: {}'.format(format_currency(data[t]), format_position(data[t] - bought_price)))
		# else:
			# print('Sit at: {}'.format(format_currency(data[t])))

		done = True if t == data_length - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			return total_profit

if __name__ == '__main__':

	if len(sys.argv) < 3:
		print('Usage: python train.py [stock] [window]')
		exit(0)
	stock_name, window_size = sys.argv[1], int(sys.argv[2])

	for model in os.listdir('models'):
		if not os.path.isdir('models/{}'.format(model)) and 'memory' not in model:
			profit = evaluate_model(model)
			if profit == 6.5 or profit == 0.0:
				print('USELESS: {}'.format(model))
			else: 
				print('{}: {}'.format(model, format_position(profit)))