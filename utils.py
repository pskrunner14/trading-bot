import math

import pandas as pd
import numpy as np

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))


def sigmoid(x):
    """ Computes sigmoid activation.

    Args:
            x (float): input value to sigmoid function.
    Returns:
            float: sigmoid function output.
    """
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except OverflowError as err:
        print("Overflow err: {0} - Val of x: {1}".format(err, x))
    except ZeroDivisionError:
        print("division by zero!")
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_stock_data(stock_file):
    """ Reads stock data from csv file. """
    df = pd.read_csv(stock_file)
    return list(df['Adj Close'])


def get_state(data, t, n_days):
    """ Returns an n-day state representation ending at time t. """
    d = t - n_days + 1
    block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1]  # pad with t0
    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])
