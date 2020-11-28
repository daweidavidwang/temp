import gym
import highway_env
from collections import deque
import numpy as np

from processonline import vehicle,preProcess

class multiTaskRL(object):
    def __init__(self):
        self.env = gym.make("highway-v0")
        self.preprocess = preProcess(num_history = 20,disRange = 112)
        self.preprocessR = preProcess(num_history = 20, disRange = 112)

        self.egoX = deque(maxlen = 50)
        self.egoY = deque(maxlen = 50)
        self.Reward = deque(maxlen = 49)
        self.Done = deque(maxlen = 49)
        self.Info = deque(maxlen = 49)
        self.Action = deque(maxlen = 49)
        self.Obstacle = deque(maxlen = 50)

    def _cleardeque(self):
        self.egoX.clear()
        self.egoY.clear()
        self.Reward.clear()
        self.Done.clear()
        self.Info.clear()
        self.Action.clear()
        self.Obstacle.clear()

    def preprocessObs(self,current_obs):
        ego_x,ego_y,vehicle_list = self.decomposeObs(current_obs) 
        obs = self.preprocess.getInput(
            ego_x,ego_y,vehicle_list
        )
        return obs

    def record(self,action,current_obs,reward,done,info):
        ego_x,ego_y,vehicle_list = self.decomposeObs(current_obs)
        self.egoX.append(ego_x)
        self.egoY.append(ego_y)

        self.Reward.append(reward)
        self.Done.append(done)
        self.Info.append(info)
        self.Action.append(action)

        self.Obstacle.append(vehicle_list)

    def deComposeRecord(self,record):
        pre_obs = record[0]
        action  = record[1]
        cur_obs = record[2]
        reward  = record[3]
        done    = record[4]
        future_trajectory = record[5]
        info    = record[6]

        return pre_obs,action,cur_obs,reward,done,future_trajectory,info

    def getRecord(self):
        future_trajectory = []
        ego_xlist = []
        ego_ylist = []

        if len(self.egoX) >= 50:
            for i in range(20):
                ego_x = self.egoX[i]
                ego_y = self.egoY[i]
                vehicle_list = self.Obstacle[i]
                pre_obs = self.preprocessR.getInput(
                    ego_x,ego_y,vehicle_list
                )

            for i in range(20,50):
                ego_xlist.append(self.egoX[i])
                ego_ylist.append(self.egoY[i])

            relx,rely = self.preprocessR.get_relPos(ego_xlist,ego_ylist)
            future_trajectory.extend(relx)
            future_trajectory.extend(rely)
            future_trajectory = np.array(future_trajectory)

            for i in range(1,21):
                ego_x = self.egoX[i]
                ego_y = self.egoY[i]
                vehicle_list = self.Obstacle[i]
                cur_obs = self.preprocessR.getInput(
                    ego_x,ego_y,vehicle_list
                )

            action = self.Action[19]
            reward = self.Reward[19]
            done = self.Done[19]
            info = self.Info[19]
            
            return [pre_obs,action,cur_obs,reward,done,future_trajectory,info]
        else:
            return None

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

    def step(self,action):
        current_obs,reward,done,info = self.env.step(action)
        obs = self.preprocessObs(current_obs)
        self.record(action,current_obs,reward,done,info)
        return obs,done
    
    def reset(self):
        self.preprocess._init()
        self._cleardeque()

        current_obs = self.env.reset()
        ego_x,ego_y,vehicle_list = self.decomposeObs(current_obs)
        obs = self.preprocessObs(current_obs)

        self.egoX.append(ego_x)
        self.egoY.append(ego_y)
        self.Obstacle.append(vehicle_list)

        return obs

if __name__ == "__main__":
    mt = multiTaskRL()
    obs = mt.reset()

    for i in range(1000):
        action = mt.env.action_type.actions_indexes["IDLE"]
        obs,done = mt.step(action)
        mt.env.render()
        record = mt.getRecord()
        if record is not None:
            pre_obs,action,cur_obs,reward,done,future_trajectory,info = \
                mt.deComposeRecord(record)
            print(pre_obs.shape)    
        if done:
            mt.reset()





