import random
from collections import deque

import keras
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l1, l2

# AGENT - Stock Trading Bot
class Agent:

    # Initialization of agent
	def __init__(self, state_size, is_eval=False, pretrained=False, model_name=None):
		
		# agent config
		self.state_size = state_size    # normalized previous days
		self.action_size = 3            # sit, buy, sell
		self.model_name = model_name
		self.is_eval = is_eval
		self.memory = deque(maxlen=1000)
		self.inventory = []

        # model config
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.loss = 'mse'
		self.optimizer = Adam(lr=self.learning_rate)

		self.first_iter = True

		# load pretrained model or instantiate a new one
		if pretrained and model_name is not None:
			self.model = load_model('models/' + model_name)
			print('Loaded {} model!\n'.format(model_name))
		else:
			self.model = self._model()

    # Create the model
	def _model(self):
		model = Sequential()
		# Input Layer
		model.add(Dense(units=64, input_dim=self.state_size, activation="relu", 
						kernel_initializer='glorot_normal',  kernel_regularizer=l2(0.01), 
						activity_regularizer=l1(0.01)))
		# 2 Hidden Layers
		model.add(Dense(units=32, activation="relu", 
						kernel_initializer='glorot_normal',  kernel_regularizer=l2(0.01), 
						activity_regularizer=l1(0.01)))
		model.add(Dense(units=8, activation="relu", 
						kernel_initializer='glorot_normal',  kernel_regularizer=l2(0.01), 
						activity_regularizer=l1(0.01)))
		# Output Layer
		model.add(Dense(self.action_size, 
						kernel_initializer='glorot_normal',  kernel_regularizer=l2(0.01), 
						activity_regularizer=l1(0.01)))
		model.compile(loss=self.loss, optimizer=self.optimizer)
		return model

	# Remember the action on a certain step
	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

    # Take action from given possible actions
	def act(self, state):
		if not self.is_eval and random.random() <= self.epsilon:
			return random.randrange(self.action_size)
		if self.first_iter:
			self.first_iter = False
			return 1
		options = self.model.predict(state)
		return np.argmax(options[0])

    # Train on previous experience in memory
	def train_experience_replay(self, batch_size):
		
		mini_batch = random.sample(self.memory, batch_size)

		avg_loss = []

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			history = self.model.fit(state, target_f, epochs=1, verbose=0)
			loss = history.history['loss'][0]
			avg_loss.append(loss)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return np.mean(np.array(avg_loss))