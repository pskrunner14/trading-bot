import sys
import os

import tensorflow as tf

from agent.agent import Agent
from model import train_model, evaluate_model
from utils import get_stock_data, format_position

TENSORFLOW_BACKEND = True

"""
Print result
"""

def show_train_result(result, val_position, initial_offset):
    if val_position == initial_offset or val_position == 0.0:
        print('Episode {}/{} - Train Position: {}  Val Position: USELESS  Loss: {:.4f}  (~{:.4f} secs)'.format(result[0], 
                result[1], format_position(result[2]), result[3], result[4]))
    else:
        print('Episode {}/{} - Train Position: {}  Val Position: {}  Loss: {:.4f}  (~{:.4f} secs)'.format(result[0], 
                result[1], format_position(result[2]), format_position(val_position), result[3], result[4]))

if __name__ == '__main__':

    """Faster computation on CPU (only if using tensorflow-gpu)"""
    if TENSORFLOW_BACKEND:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if len(sys.argv) < 7:
        print('Usage: python train.py [train stock] [val stock] [window] [episodes] [model] [pretrained (0/1)]')
        exit(0)
    train_stock_name, val_stock_name, window_size, episode_count, model_name, pretrained = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], bool(int(sys.argv[6]))

    agent = Agent(window_size, pretrained=pretrained, model_name=model_name)
    train_data = get_stock_data(train_stock_name)
    test_data = get_stock_data(val_stock_name)

    batch_size = 32
    initial_offset = test_data[1] - test_data[0]

    for episode in range(1, episode_count + 1):
        train_result = train_model(agent, episode, train_data, episode_count=episode_count,
                    batch_size=batch_size, window_size=window_size)
        val_result = evaluate_model(agent, test_data, 
                    window_size=window_size)
        show_train_result(train_result, val_result, initial_offset)
    
    print('Done Training!')