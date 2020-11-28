import os
import torch
import gym
import numpy as np
from tensorboardX import SummaryWriter

import highway_env
from dqnagent import DQNAgent

ENV = gym.make('highway-v0')
EPISODES = 200000
STORE_PATH = "/home/dawei/highway_project/highway_multitask_action/model"

tensorboard_saved_dir = os.path.join(STORE_PATH,"tensorboard")
writer = SummaryWriter(tensorboard_saved_dir)

def write_tenboard(writer,episode,reward,step):
    writer.add_scalar('episode/reward',np.sum(reward),episode)
    writer.add_scalar('episode/step',step,episode)


def train():
    agent = DQNAgent(ENV)
    agent.set_writer(writer)

    for episde in range(EPISODES):
        reward = []
        step = 0
        state = ENV.reset()
        while True:
            ENV.render()
            action = agent.act(state)
            next_state, reward, done, info = ENV.step(action)
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
             
