import gym
import time
from collections import deque
from sklearn import preprocessing
import numpy as np 

import highway_env
from processonline import preProcess,vehicle

class historyObs(object):
    def __init__(self):
        self.env = gym.make("highway-v0")
        self.frequency = 10
        self.preprocess = preProcess(num_history = 10,disRange = 112)

    def decomposeObs(self,current_obs):
        obs_list = []

        vehicle_num,features_num = current_obs.shape
        ego_posX = current_obs[0][1]
        ego_posY = current_obs[0][2]

        for i in range(1,vehicle_num):
            obs_list.append(
                vehicle(current_obs[i][1],current_obs[i][2])
            )

        return ego_posX,ego_posY,obs_list

    def preprocessObs(self,current_obs):
        ego_x,ego_y,vehicle_list = self.decomposeObs(current_obs) 
        obs = self.preprocess.getInput(
            ego_x,ego_y,vehicle_list
        )
        return obs

    def step(self,action):
        for i in range(self.frequency):
            obs,reward,done,info = self.env.step(action)
            obs = self.preprocessObs(obs)
            self.env.render()
            if done:
                break
        return obs,reward,done,info

    def reset(self):
        self.preprocess._init()
        obs = self.env.reset()
        obs = self.preprocessObs(obs)
        return obs

if __name__ == "__main__":
    ho = historyObs()
    obs = ho.reset()

    for i in range(1000):
        action = ho.env.action_type.actions_indexes["IDLE"]
        obs,reward,done,info = ho.step(action)
        print(obs.shape)
        ho.env.render()
        if done:
            print("true")
            ho.reset()
