import math

import numpy as np

# Formats Position
format_position = lambda price: ('-$' if price < 0 else '+$') + '{0:.2f}'.format(abs(price))

# Formats Currency
format_currency = lambda price: '${0:.2f}'.format(abs(price))

# Computes sigmoid activation
def sigmoid(x):
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

# Returns the list containing stock data from historical financial data csv file
def get_stock_data(stock):
	stock_prices = []
	
	lines = open('data/{}.csv'.format(stock), 'r').read().splitlines()
	
	for line in lines[1:]:
		stock_prices.append(float(line.split(',')[4]))
	return stock_prices

# Returns an n-day state representation ending at time t
def get_state(data, t, n_days):
	d = t - n_days + 1
	block = data[d: t + 1] if d >= 0 else -d * [data[0]] + data[0: t + 1] # pad with t0
	
	res = []
	
	for i in range(n_days - 1):
		res.append(sigmoid(block[i + 1] - block[i]))

	return np.array([res])
