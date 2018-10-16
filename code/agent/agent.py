import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from helper import *

from keras.callbacks import TensorBoard, EarlyStopping

import numpy as np
import random
import os
# from collections import dematque
from keras import backend as K

def huber_loss(a, b, in_keras=True):
	error = a - b
	quadratic_term = error*error / 2
	linear_term = abs(error) - 1/2
	use_linear_term = (abs(error) > 1.0)
	if in_keras:
		# Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
		use_linear_term = K.cast(use_linear_term, 'float32')
	return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term
  
class Agent:
	def __init__(self, state_size, memory, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		# self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		# self.epsilon = 1.0 
		self.epsilon = 0.7
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.495
		self.firstIter = True

		self.memory = memory
		self.bigMemory = []

		# self.model = load_model("models/" + model_name) if is_eval else self._model()
		self.model = self._model()
		self.target_model = clone_model(self.model)
		self.target_model.set_weights(self.model.get_weights())

	def _model(self):
		model = Sequential()
		# model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		# model.add(Dense(units=32, activation="relu"))
		# model.add(Dense(units=8, activation="relu"))
		# model.add(Dense(self.action_size, activation="linear"))
		model.add(LSTM(64, input_shape=(self.state_size, 1)))
		model.add(Dense(units=32, activation="relu"))
		model.add(Dense(units=8, activation="relu"))
		model.add(Dense(self.action_size, activation="softmax"))
		# model.compile(loss="mse", optimizer=Adam(lr=0.001))
		model.compile(loss='mse', optimizer=Adam(lr=0.001))
		# model.compile(loss='mse', optimizer=Adam(lr=0.001))

		return model

	#Action
	def act(self, state):
		if not self.is_eval and rand_val <= self.epsilon:
			return random.randrange(self.action_size)
		if(self.firstIter):
			self.firstIter = False
			return 1
		options = self.model.predict(state)
		#print("Using prediction")
		
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		# mini_batch = []
		# l = len(self.memory)
		# for i in range(l - batch_size + 1, l):
		# 	mini_batch.append(self.memory.popleft())
		mini_batch = self.memory.sample(batch_size)

		for state, action, reward, next_state, done in mini_batch:
			if not done:
				# target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			else:
				target = reward

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

	def getTDError(self):
		error = 0
		for state, action, reward, next_state, done in self.bigMemory:
			# target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])
			target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
			predicted_target = self.model.predict(state)[0][action]
			error += (target - predicted_target) ** 2
		
		error /= len(self.bigMemory)
		return error

