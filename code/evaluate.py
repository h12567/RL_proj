import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
from preprocess_price import preprocess_price
from keras.models import clone_model
import sys

if len(sys.argv) != 3:
    print ("Usage: python evaluate.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]
model = load_model("models/" + model_name)
window_size = model.layers[0].input.shape.as_list()[1]

agent = Agent(window_size, True, model_name)
# data = getStockDataVec(stock_name)
data = preprocess_price(stock_name)

l = len(data) - 1
batch_size = 32

state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

# agent.inventory.append(data[0])
# next_state = getState(data, 0 + 1, window_size + 1)
# reward = 0
# agent.memory.append((state, action, reward, next_state, done))

for t in range(l):
    if t == 0:
        action = 1
    else: 
        action = agent.act(state)


    if ( (t % 20) == 0 ) :
        agent.target_model = clone_model(agent.model)
        agent.target_model.set_weights(agent.model.get_weights())

    # sit
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    if action == 1: # buy
        agent.inventory.append(data[t])
        print ("Buy: " + formatPrice(data[t]))

    elif action == 2 and len(agent.inventory) > 0: # sell
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    done = True if t == l - 1 else False
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    if len(agent.memory) > batch_size:
        agent.expReplay(batch_size) 

    if done:
        print ("--------------------------------")
        print (stock_name + " Total Profit: " + formatPrice(total_profit))
        print ("--------------------------------")
