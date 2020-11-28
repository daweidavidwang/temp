import numpy as np
import copy
import cv2 as cv
import time
from collections import deque

from traspin import traSpin

class egoCar(object):
    def __init__(self,num_history):
        self.num_history = num_history
        
        self.x_pos = deque(maxlen = self.num_history)
        self.y_pos = deque(maxlen = self.num_history)

    def initPos(self):
        for i in range(self.num_history):
            self.x_pos.append(0.0)
            self.y_pos.append(0.0)

    def updatePos(self,x,y):
        self.x_pos.append(x)
        self.y_pos.append(y)

        return self.x_pos,self.y_pos

class obsCar(object):
    def __init__(self,num_history,disRange):
        self.num_history = num_history
        
        self.x_lim = disRange
        self.y_lim = disRange
        
        self.x_posList = deque(maxlen = self.num_history)
        self.y_posList = deque(maxlen = self.num_history)

    def initPos(self):
        for i in range(self.num_history):
            self.x_posList.append(0.0)
            self.y_posList.append(0.0)

    
    def updatePos(self,vehicle_list,ego_x,ego_y):
        x_pos,y_pos = self.get_currentPos(vehicle_list,ego_x,ego_y)
        self.x_posList.append(x_pos)
        self.y_posList.append(y_pos)

        return self.x_posList,self.y_posList

    def get_currentPos(self,vehicle_list,ego_x,ego_y):
        x_currentPos = []
        y_currentPos = [] 
        for idx in range(len(vehicle_list)):
            obs_x = vehicle_list[idx].center.x
            obs_y = vehicle_list[idx].center.y

            if abs(obs_x - ego_x) <= self.x_lim and abs(obs_y - ego_y) <= self.y_lim:
                x_currentPos.append(obs_x)
                y_currentPos.append(obs_y)
        
        return x_currentPos,y_currentPos
            
class preProcess(object):
    def __init__(self,num_history = 20,disRange = 112):
        self.x_lim = disRange
        self.y_lim = disRange
        self.num = 0

        self.point_color = 0
        self.empty_area = 255

        self.num_history = num_history

        self.egoVehicle = egoCar(self.num_history)
        self.obsVehicle = obsCar(self.num_history,disRange)

        self.ts = traSpin(self.x_lim,self.y_lim)

        self._init()

    def _init(self):
        self.egoVehicle.initPos()
        self.obsVehicle.initPos()

    def _updatePos(self,ego_x,ego_y,vehicle_list):
        ego_xdeque ,ego_ydeque = self.egoVehicle.updatePos(ego_x,ego_y)
        obs_xdeque, obs_ydeque = self.obsVehicle.updatePos(vehicle_list,ego_x,ego_y)

        return ego_xdeque,ego_ydeque,obs_xdeque,obs_ydeque
    
    def get_relPos(self,posX,posY):
        relx,rely = self.ts.get_panpoint(posX,posY)
        spinx,spiny = self.ts.get_spinpoint(relx,rely)
        
        return spinx,spiny

    def getInput(self,ego_x,ego_y,vehicle_list):
        ego_xdeque,ego_ydeque,obs_xdeque,obs_ydeque = self._updatePos(ego_x,ego_y,vehicle_list)

        self.ts.get_pan(ego_xdeque,ego_ydeque)
        self.ts.get_angle(ego_xdeque,ego_ydeque)
        self.ts.get_spindirection(ego_xdeque,ego_ydeque)

        relx,rely = self.ts.get_panpoint(ego_xdeque,ego_ydeque)
        spinx,spiny = self.ts.get_spinpoint(relx,rely)

        obs_x = []
        obs_y = []
        for i in range(self.num_history):
            sub_xList = copy.deepcopy(obs_xdeque[i])
            sub_yList = copy.deepcopy(obs_ydeque[i])

            try:
                sub_relx,sub_rely = self.ts.get_panpoint(sub_xList,sub_yList)
                sub_spinx,sub_spiny = self.ts.get_spinpoint(sub_relx,sub_rely)
            except:
                sub_xList = [sub_xList]
                sub_yList = [sub_yList]
                sub_relx,sub_rely = self.ts.get_panpoint(sub_xList,sub_yList)
                sub_spinx,sub_spiny = self.ts.get_spinpoint(sub_relx,sub_rely)

            obs_x.append(sub_spinx)
            obs_y.append(sub_spiny)
        
        channel_list = []
        for i in range(self.num_history):
            x = [spinx[i]]
            y = [spiny[i]]

            channel = self.get_frame_matrix(x,y)
            channel_list.append(channel)

        for i in range(self.num_history):
            x = obs_x[i]
            y = obs_y[i]
            channel = self.get_frame_matrix(x,y)
            channel_list.append(channel)
        
        channel_tuple = tuple(channel_list)
        channel_array = np.stack((channel_tuple),axis=2)
        channel_array = channel_array.reshape(40,112,112)

        return channel_array
    

    def Normalize(self,obs):
        mx = self.empty_area
        mn = self.point_color
        return obs / mx - mn
            
    def get_frame_matrix(self,x,y):
        img = np.ones((224, 224), np.uint8) * self.empty_area
        point_size = 1
        point_color = (self.point_color, self.point_color)
        thickness = 4

        points_list = []
        for i in range(len(x)):
            points_list.append((int(x[i]),int(y[i])))
        for point in points_list:
            cv.circle(img, point, point_size, point_color, thickness)
        img_new = cv.resize(img,(112,112))

        # cv.namedWindow("image")
        # cv.imshow('image', img_new)
        # cv.waitKey (100) # 显示 10000 ms 即 10s 后消失
        # cv.destroyAllWindows()
        return self.Normalize(img_new)
        # return img_new

class center(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

class vehicle(object):
    def __init__(self,x,y):
        self.center = center(x,y)

if __name__ == "__main__":
    pp =preProcess(20,112)
    for i in range(20):
       out =  pp.getInput(i,i,[vehicle(2*i,3*i)])
       print(out.shape)
       print(type(out))




    