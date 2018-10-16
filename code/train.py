from agent.agent import Agent
from agent.memory import Memory
from functions import *
from preprocess_price import preprocess_price
from keras.models import clone_model
import sys
import numpy as np 

if len(sys.argv) != 4:
    print("Usage: python3 train.py [stock] [window] [episodes]")
    exit()

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

max_queue_size = 100
memory = Memory(max_queue_size)
agent = Agent(window_size, memory)

data = preprocess_price(stock_name)
# data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32
budget = 10000
errors = []
profits = []
fee = 0.2 / 100

for e in range(episode_count + 1):
    print ("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)
    state = state.reshape((state.shape[0], state.shape[1], 1))

    if ( (e % 20) == 0 ):
        agent.target_model = clone_model(agent.model)
        agent.target_model.set_weights(agent.model.get_weights())

    total_profit = 0
    agent.inventory = []

    num_buy = 0
    num_sell = 0
    for t in range(l):
        action = agent.act(state)

        # sit
        next_state = getState(data, t + 1, window_size + 1)
        next_state = next_state.reshape((next_state.shape[0], next_state.shape[1], 1))
        reward = 0

        if action == 1 and budget > 0: # buy
            num_buy += 1
            agent.inventory.append(data[t]*(1 + fee))
            budget -= data[t] * (1 + fee)
            print("Buy: " + formatPrice(data[t]))

        elif action == 2 and len(agent.inventory) > 0: # sell
            num_sell += 1
            bought_price = agent.inventory.pop(0)
            reward = max(data[t] * (1 - fee) - bought_price, 0)
            total_profit += data[t] * (1 - fee) - bought_price
            budget += data[t]
            print(t)
            print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] * (1 - fee) - bought_price))

        done = True if t == l - 1 else False
        agent.memory.add((state, action, reward, next_state, done))
        agent.bigMemory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("BUY " + str(num_buy))
            print("SELL " + str(num_sell))
            print(len(agent.inventory))
            print(data[t])
            reward = data[t] * len(agent.inventory) - np.array(agent.inventory).sum()
            error = agent.getTDError()
            errors.append(error)
            print("TOTAL TD ERROR " + str(error))
            print("Last Big sell: " + formatPrice(reward))
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            profits.append(total_profit)
            print("--------------------------------")

        if len(agent.memory.buffer) > batch_size:
            agent.expReplay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))

np.save('error_log', np.array(errors))
np.save('profit_log', np.array(profits))