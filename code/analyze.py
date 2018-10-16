from agent.agent import Agent
from agent.memory import Memory
from functions import *
from preprocess_price import preprocess_price
from keras.models import clone_model
import sys
import numpy as np 
import matplotlib.pyplot as plt
from keras.models import load_model
from random import *

iA
    print("Usage: python3 train.py [stock] [window] [episodes]")
    exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

max_queue_size = 100
memory = Memory(max_queue_size)
agent = Agent(window_size, memory, True)

data = preprocess_price(stock_name)
data = data[int(0.8*len(data)): ]

batch_size = 32
budget = 4000
fee = 0.2/100

plt.plot(data, label='Prediction')
inventory = []
sell_owe = []
profit = 0
budget = 4000
l = len(data) - 1

agent.model.load_weights('modelC.hdf5')
for t in range(l):
  state = getState(data, t, window_size + 1)
  # state = state.reshape((state.shape[0], state.shape[1], 1))
  action = agent.act(state)
  if(t == 115):
    action = 1
  
  # print(action)
  if(action == 1 and budget > 0):
    print(t)
    if len(sell_owe) > 0 :
      sell_owe.pop(0)
    else:
      inventory.append(data[t] * (1+fee))

    budget -= data[t] * (1 + fee)
    plt.scatter([t], [data[t]], label='signal', c='b')
    print("BUY")
  elif(action == 2):
    print(t)

    if(len(inventory) > 0):
      bought_price = inventory.pop(0)
    else:
      bought_price = 0
      sell_owe.append(1)
    budget += data[t] * (1 - fee)
    plt.scatter([t], [data[t]], label='signal', c='r')
    print("SELL")

if(len(sell_owe) > 0):
  budget -= data[t] * (1+fee) * len(sell_owe)

if(len(inventory) > 0):
  budget += data[t] * (1-fee) * len(inventory)

print("TOAL PROF")
print(budget)
plt.show()
