import os
import sys
import click
import logging
import coloredlogs

import numpy as np
import keras.backend as K

from tqdm import tqdm
from time import clock

from agent import Agent
from utils import get_state, get_stock_data, format_currency, format_position


@click.command()
@click.option(
    '-es',
    '--eval-stock',
    type=click.Path(exists=True),
    default='data/GOOGL_2018.csv',
    help='Evaluation stock data csv file path'
)
@click.option(
    '-ws',
    '--window-size',
    default=10,
    help='n-day window size of previous states to normalize over'
)
@click.option(
    '-mn',
    '--model-name',
    default='model_GOOGL',
    help='Name of the model for saving/checkpointing etc. [script looks in `models/`]'
)
@click.option(
    '-d',
    '--debug',
    is_flag=True,
    help='Flag for debug mode (prints position on each step)'
)
def main(eval_stock, window_size, model_name, debug):
    """ Evaluates the stock trading bot.

Please see https://arxiv.org/abs/1312.5602 for more details.

Args: optional arguments [python evaluate.py --help]
"""
    switch_k_backend_device()
    data = get_stock_data(eval_stock)
    initial_offset = data[1] - data[0]

    if model_name is not None:
        '''Single Model Evaluation'''
        agent = Agent(window_size, pretrained=True, model_name=model_name)
        profit, _ = evaluate_model(agent, data, window_size, debug)
        show_eval_result(model_name, profit, initial_offset)
        del agent
    else:
        '''Multiple Model Evaluation'''
        for model in os.listdir('models'):
            if not os.path.isdir('models/{}'.format(model)):
                agent = Agent(window_size, pretrained=True, model_name=model)
                profit = evaluate_model(agent, data, window_size, debug)
                show_eval_result(model, profit, initial_offset)
                del agent


def evaluate_model(agent, data, window_size, debug):
    data_length = len(data) - 1
    state = get_state(data, 0, window_size + 1)
    total_profit = 0
    agent.inventory = []
    history = []
    for t in range(data_length):
        action = agent.act(state, is_eval=True)
        # SIT
        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0
        # BUY
        if action == 1:
            agent.inventory.append(data[t])
            history.append((data[t], 'BUY'))
            if debug:
                logging.debug('Buy at: {}'.format(format_currency(data[t])))
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            history.append((data[t], 'SELL'))
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            if debug:
                logging.debug('Sell at: {} | Position: {}'.format(
                    format_currency(data[t]), format_position(data[t] - bought_price)))
        else:
            history.append((data[t], 'HOLD'))

        done = True if t == data_length - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            return total_profit, history


def show_eval_result(model_name, profit, initial_offset):
    if profit == initial_offset or profit == 0.0:
        logging.info('{}: USELESS\n'.format(model_name))
    else:
        logging.info('{}: {}\n'.format(model_name, format_position(profit)))


def switch_k_backend_device():
    """ Switches `keras` backend from GPU to CPU if required.

    Faster computation on CPU (if using tensorflow-gpu).
    """
    if K.backend() == 'tensorflow':
        logging.debug('switching to TensorFlow for CPU')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':
    coloredlogs.install(level='DEBUG')
    try:
        main()
    except KeyboardInterrupt:
        print('Aborted')
