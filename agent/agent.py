import random
import pickle
from collections import deque

import keras
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, BatchNormalization, Dense, Dropout
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
		self.inventory = []
		self.model_type = 'REV_1'

        # model config
		self.model_name = model_name
		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.001
		self.loss = 'mse'
		self.optimizer = Adam(lr=self.learning_rate)

		self.first_iter = True

		# load pretrained model or instantiate a new one
		if pretrained and self.model_name is not None:
			self.model, self.memory = self.load()
			if not self.is_eval:
				print('Loaded {} model and memory!\n'.format(self.model_name))
			# self.model.summary()
		else:
			self.model, self.memory = self._model(), deque(maxlen=1000)

    # Create the model
	def _model(self):
		model = Sequential(name=self.model_type)
		
		model.add(Dense(units=24, input_dim=self.state_size,
						kernel_initializer='glorot_normal', 
						kernel_regularizer=l2(0.01)))
		model.add(Activation('relu'))
		
		model.add(Dense(units=64,
						kernel_initializer='glorot_normal', 
						kernel_regularizer=l2(0.01)))
		model.add(Activation('relu'))

		model.add(Dense(units=64,
						kernel_initializer='glorot_normal',
						kernel_regularizer=l2(0.01)))
		model.add(Activation('relu'))
		
		model.add(Dense(units=24, 
						kernel_initializer='glorot_normal',
						kernel_regularizer=l2(0.01)))
		model.add(Activation('relu'))
		
		model.add(Dense(units=self.action_size, 
						kernel_initializer='glorot_normal', 
						kernel_regularizer=l2(0.01)))

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

		del mini_batch
		del X_train
		del y_train

		mse = history.history['loss'][0]

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return mse

	# Save agent model and memory on disk
	def save(self, episode):
		self.model.save('models/{}_{}'.format(self.model_name, episode))
		with open('models/{}_{}.memory.pkl'.format(self.model_name, episode), 'wb') as file:
			pickle.dump(self.memory, file)

	# Load the agent model and memory saved on disk
	def load(self):
		model = load_model('models/' + self.model_name)
		if not self.is_eval:
			with open('models/{}.memory.pkl'.format(self.model_name), 'rb') as file:
				memory = pickle.load(file)
		memory = deque(maxlen=1000)
		return model, memory