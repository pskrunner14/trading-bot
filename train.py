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
from evaluate import evaluate_model
from utils import (
    get_state,
    get_stock_data,
    format_currency,
    format_position
)


@click.command()
@click.option(
    '-ts',
    '--train-stock',
    type=click.Path(exists=True),
    default='data/GOOGL.csv',
    help='Training stock data csv file path'
)
@click.option(
    '-vs',
    '--val-stock',
    type=click.Path(exists=True),
    default='data/GOOGL_2018.csv',
    help='Validation stock data csv file path'
)
@click.option(
    '-ws',
    '--window-size',
    default=10,
    help='n-day window size of previous states to normalize over'
)
@click.option(
    '-bz',
    '--batch-size',
    default=16,
    help='Mini-batch size to use when training the agent'
)
@click.option(
    '-ep',
    '--ep-count',
    default=50,
    help='Number of episodes to train the agent'
)
@click.option(
    '-mn',
    '--model-name',
    default='model_GOOGL',
    help='Name of the model for saving/checkpointing etc.'
)
@click.option(
    '-pre',
    '--pretrained',
    is_flag=True,
    help='If model is pre-trained'
)
@click.option(
    '-d',
    '--debug',
    is_flag=True,
    help='Flag for debug mode (prints position on each step during evaluation)'
)
def main(train_stock, val_stock, window_size, batch_size, ep_count, model_name, pretrained, debug):
    """ Trains the stock trading bot using Deep Q-Learning.

    Please see https://arxiv.org/abs/1312.5602 for more details.

    Args: optional arguments [python train.py --help]
    """
    switch_k_backend_device()

    agent = Agent(window_size, pretrained=pretrained, model_name=model_name)
    train_data = get_stock_data(train_stock)
    val_data = get_stock_data(val_stock)

    initial_offset = val_data[1] - val_data[0]

    for episode in range(1, ep_count + 1):
        train_result = train_model(agent, episode, train_data, ep_count=ep_count,
                                   batch_size=batch_size, window_size=window_size)
        val_result, _ = evaluate_model(agent, val_data, window_size, debug)
        show_train_result(train_result, val_result, initial_offset)


def train_model(agent, episode, data, ep_count=100, batch_size=32, window_size=10):
    data_length = len(data) - 1
    total_profit = 0
    agent.inventory = []
    avg_loss = []
    start = clock()
    state = get_state(data, 0, window_size + 1)
    for t in tqdm(range(data_length), total=data_length, leave=True, desc='Episode {}/{}'.format(episode, ep_count)):
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

    if episode % 10 == 0:
        agent.save(episode)
    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)), end)


def show_train_result(result, val_position, initial_offset):
    """ Displays training results. """
    if val_position == initial_offset or val_position == 0.0:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: USELESS  Train Loss: {:.4f}  (~{:.4f} secs)'
                     .format(result[0], result[1], format_position(result[2]), result[3], result[4]))
    else:
        logging.info('Episode {}/{} - Train Position: {}  Val Position: {}  Train Loss: {:.4f}  (~{:.4f} secs)'
                     .format(result[0], result[1], format_position(result[2]), format_position(val_position), result[3], result[4]))


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
        print('Aborted!')
