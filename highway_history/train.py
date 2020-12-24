import os
import torch
import numpy as np
from tensorboardX import SummaryWriter

from historyobs import historyObs
from dqnagent import DQNAgent

EPISODES = 200000
STORE_PATH = "/home/dawei/highway_project/highway_history/model"

tensorboard_saved_dir = os.path.join(STORE_PATH,"tensorboard")
writer = SummaryWriter(tensorboard_saved_dir)

def write_tenboard(writer,episode,reward,step):
    writer.add_scalar('episode/reward',np.sum(reward),episode)
    writer.add_scalar('episode/step',step,episode)

def train():
    ho = historyObs()
    agent = DQNAgent(ho.env)
    agent.set_writer(writer)
    agent.load("/home/dawei/highway_project/highway_history/model/35600")
    for episde in range(EPISODES):
        reward = []
        step = 0
        state = ho.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done, info = ho.step(action)
            #state, action, reward, next_state, done, info
            agent.record(state,action,reward,next_state,done,info)
            step += 1
            
            if done:
                if episde % 100 ==0:
                    filename = os.path.join(STORE_PATH,str(episde))
                    agent.save(filename)
                write_tenboard(writer,episde,reward,step)
                break

            state = next_state

if __name__ == "__main__":
    train()
                


    