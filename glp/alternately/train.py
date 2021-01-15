import os
import torch
import gym
import numpy as np
from tensorboardX import SummaryWriter

import highway_env
from dqnagent import DQNAgent

ENV = gym.make('highway-v0')
EPISODES = 200000
STORE_PATH = "./model"

tensorboard_saved_dir = os.path.join(STORE_PATH,"tensorboard")
writer = SummaryWriter(tensorboard_saved_dir)

def write_tenboard(writer,episode,reward,step):
    writer.add_scalar('episode/reward',np.sum(reward),episode)
    writer.add_scalar('episode/step',step,episode)


def train():
    agent = DQNAgent(ENV)
    agent.set_writer(writer)
    agent.load("/media/dawei/data/glp/alternately/model/800")
    for episde in range(0,EPISODES):
        reward = []
        step = 0
        current_state, current_future_pos, current_past_pos = \
            ENV.reset()
        while True:
            ENV.render()
            action = agent.act(current_state, current_past_pos)
            next_state, next_future_pos, next_past_pos, reward, done, info = \
                ENV.step(action)
            #state, action, reward, next_state, done, info
            agent.record(current_state, current_future_pos, current_past_pos,\
                action,reward,\
                    next_state, next_future_pos, next_past_pos, \
                        done,info)
            step += 1
            
            if done:
                if episde % 100 ==0:
                    filename = os.path.join(STORE_PATH,str(episde))
                    agent.save(filename)
                write_tenboard(writer,episde,reward,step)
                print("step = ",step," sum reward = ",np.sum(reward))
                break

            current_state, current_future_pos, current_past_pos = \
                next_state, next_future_pos, next_past_pos

def test():
    agent = DQNAgent(ENV)
    agent.set_writer(writer)
    agent.eval()
    agent.load("/home/glp/highway_project/highway_tramultask/model/8100")
    for episde in range(EPISODES):
        reward = []
        step = 0
        current_state, current_future_pos, current_past_pos = \
            ENV.reset()
        done  = False
        average_speed = 0
        slowernum = 0
        step = 1

        while True:
            ENV.render()
            action = agent.act(current_state, current_past_pos)
            next_state, next_future_pos, next_past_pos, reward, done, info = \
                ENV.step(action)
            #state, action, reward, next_state, done, info
            current_state, current_future_pos, current_past_pos = \
                next_state, next_future_pos, next_past_pos
            
            average_speed = average_speed + (info['speed'] - average_speed) / step
            if info['action'] == 4:
                slowernum += 1
            step += 1

            if done:
                print("average speed = ",average_speed,' slowernum = ',slowernum,' crash = ',info['crashed'])
                break


if __name__ == "__main__":
    train()
    #test()
             
