import os
import torch
import gym
import numpy as np
from tensorboardX import SummaryWriter
from preprocess import ZFilter

import highway_env
from dqnagent import DQNAgent

ENV = gym.make('highway-v0')

running_state = ZFilter((84,84,3), clip=1.0)
state_norm = True

screen_width, screen_height = 84, 84
config = {
    "offscreen_rendering": False,
    "observation": {
        "type": "GrayscaleObservation",
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "stack_size": 3,
        "observation_shape": (screen_width, screen_height)
    },
    "screen_width": screen_width,
    "screen_height": screen_height,
    "scaling": 1.75,
    "policy_frequency": 1
}
ENV.configure(config)

EPISODES = 200000
STORE_PATH = "/home/dawei/highway_project/highway_envimage/model"

tensorboard_saved_dir = os.path.join(STORE_PATH,"tensorboard")
writer = SummaryWriter(tensorboard_saved_dir)

def write_tenboard(writer,episode,reward,step):
    writer.add_scalar('episode/reward',np.sum(reward),episode)
    writer.add_scalar('episode/step',step,episode)


def train():
    ENV.reset()
    agent = DQNAgent(ENV)
    agent.set_writer(writer)

    for episde in range(EPISODES):
        reward = []
        step = 0
        state = ENV.reset()

        if state_norm:
            state = running_state(state).reshape(3,84,84)
        while True:
            ENV.render()
            action = agent.act(state)
            next_state, reward, done, info = ENV.step(action)
            #state, action, reward, next_state, done, info
            if state_norm:
                next_state = running_state(next_state).reshape(3,84,84)
            agent.record(state,action,reward,next_state,done,info)
            step += 1
            
            if done:
                if episde % 100 ==0:
                    filename = os.path.join(STORE_PATH,str(episde))
                    agent.save(filename)
                write_tenboard(writer,episde,reward,step)
                break

            state = next_state

def test():
    agent = DQNAgent(ENV)
    agent.eval()
    agent.load("/home/glp/highway_project/highway_action/model/12900")

    for episde in range(EPISODES):
        reward = []
        step = 0
        state = ENV.reset()
        while True:
            ENV.render()
            action = agent.act(state)
            next_state, reward, done, info = ENV.step(action)
            #state, action, reward, next_state, done, info
            state = next_state

if __name__ == "__main__":
    train()
    #test()
             
