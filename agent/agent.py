import random
import pickle
from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop, Adam
from keras.initializers import VarianceScaling

from utils import save_pickle, load_pickle


"""
Huber loss - Custom Loss Function for Q Learning

Links: 	https://en.wikipedia.org/wiki/Huber_loss
		https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
"""

def huber_loss(y_true, y_pred, clip_delta=1.0):
	error = y_true - y_pred
	cond  = K.abs(error) < clip_delta

	squared_loss = 0.5 * K.square(error)
	linear_loss  = clip_delta * (K.abs(error) - 0.5 * clip_delta)

	return K.mean(tf.where(cond, squared_loss, linear_loss))


"""
AGENT - Stock Trading Bot
"""

class Agent:

	"""
    Initialization of agent
	"""

	def __init__(self, state_size, pretrained=False, model_name=None):
		# agent config
		self.state_size = state_size    # normalized previous days
		self.action_size = 3            # [sit, buy, sell]
		self.model_name = model_name
		self.inventory = []
		self.memory = deque(maxlen=1000)
		self.first_iter = True

        # model config
		self.model_name = model_name
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.loss = huber_loss
		self.custom_objects = {'huber_loss': huber_loss}
		self.optimizer = RMSprop(lr=self.learning_rate)
		self.initializer = VarianceScaling()

		# load pretrained model or instantiate a new one
		if pretrained and self.model_name is not None:
			self.model = self.load()
			print('Loaded {} model!\n'.format(self.model_name))
			print(self.model.name)
			print(str(self.model.optimizer))
			print(str(self.model.loss))
		else:
			self.model = self._model()


	"""
    Create the model
	"""

	def _model(self):
		model = Sequential()
		
		model.add(Dense(units=24, input_dim=self.state_size, kernel_initializer=self.initializer))
		model.add(Activation('relu'))
		
		model.add(Dense(units=64, kernel_initializer=self.initializer))
		model.add(Activation('relu'))

		model.add(Dense(units=64, kernel_initializer=self.initializer))
		model.add(Activation('relu'))
		
		model.add(Dense(units=24, kernel_initializer=self.initializer))
		model.add(Activation('relu'))
		
		model.add(Dense(units=self.action_size, kernel_initializer=self.initializer))

		model.compile(loss=self.loss, optimizer=self.optimizer)
		return model


	"""
	Remember the action on a certain step
	"""

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))


	"""
    Take action from given possible actions
	"""

	def act(self, state, is_eval=False):
		if not is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		if self.first_iter:
			self.first_iter = False
			return 1
		options = self.model.predict(state)
		return np.argmax(options[0])


	"""
    Train on previous experience in memory
	"""

	def train_experience_replay(self, batch_size):
		
		mini_batch = random.sample(self.memory, batch_size)
		
		X_train, y_train = [], []

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			X_train.append(state[0])
			y_train.append(target_f[0])

		history = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0)
		loss = history.history['loss'][0]

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return loss


	"""
	Save agent model on disk
	"""

	def save(self, episode):
		self.model.save('models/{}_{}'.format(self.model_name, episode))


	"""
	Load the agent model saved on disk
	"""

	def load(self):
		return load_model('models/' + self.model_name, custom_objects=self.custom_objects)