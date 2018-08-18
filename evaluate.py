import sys
import os

# Faster computation on CPU (only if using tensorflow-gpu)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from agent.agent import Agent
from model import evaluate_model
from utils import get_stock_data, format_position


"""
Show Evaluation result
"""

def show_eval_result(model_name, profit, initial_offset):
	if profit == initial_offset or profit == 0.0:
		print('{}: USELESS\n'.format(model_name))
	else: 
		print('{}: {}\n'.format(model_name, format_position(profit)))


if __name__ == '__main__':

	model_name = None

	if len(sys.argv) == 3:
		stock_name, window_size = sys.argv[1], int(sys.argv[2])
	elif len(sys.argv) == 4:
		stock_name, window_size, model_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
	else:
		print('Usage: python evaluate.py [stock] [window] [model (optional)]')
		exit(0)

	data = get_stock_data(stock_name)
	initial_offset = data[1] - data[0]

	if model_name is not None:
		agent = Agent(window_size, pretrained=True, model_name=model_name)
		profit = evaluate_model(agent, data, window_size=window_size)
		show_eval_result(model_name, profit, initial_offset)
		del agent
	else:
		for model in os.listdir('models'):
			if not os.path.isdir('models/{}'.format(model)) and 'memory' not in model:
				agent = Agent(window_size, pretrained=True, model_name=model)
				profit = evaluate_model(agent, data, window_size=window_size)
				show_eval_result(model, profit, initial_offset)
				del agent