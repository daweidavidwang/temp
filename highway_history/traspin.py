import numpy as np
import math

class traSpin(object):
    def __init__(self,pointx,pointy):
        self.pointx = pointx
        self.pointy = pointy
        
        self.delta_x = 0.0
        self.delta_y = 0.0

        self.angle = 0.0

        self.direction = None

    def Nrotate(self,angle,valuex,valuey,pointx,pointy): 
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        nRotatex = (valuex-pointx)*math.cos(angle) - (valuey-pointy)*math.sin(angle) + pointx
        nRotatey = (valuex-pointx)*math.sin(angle) + (valuey-pointy)*math.cos(angle) + pointy
        return nRotatex, nRotatey

    def Srotate(self,angle,valuex,valuey,pointx,pointy):
        valuex = np.array(valuex)
        valuey = np.array(valuey)
        sRotatex = (valuex-pointx)*math.cos(angle) + (valuey-pointy)*math.sin(angle) + pointx
        sRotatey = (valuey-pointy)*math.cos(angle) - (valuex-pointx)*math.sin(angle) + pointy
        return sRotatex,sRotatey


    def get_angle(self,xlist,ylist):
        startx = xlist[-2]
        endx = xlist[-1]
        starty = ylist[-2]
        endy = ylist[-1]

        dx = endx - startx
        dy = endy - starty
        angle1 = math.atan2(dy, dx)
        angle1 = angle1 * 180 / math.pi

        angle2 = math.atan2(0, 1)
        angle2 = angle2 * 180 / math.pi

        if angle1*angle2 >=0:
            include_angle = abs(angle1 - angle2)
        else:
            include_angle = abs(angle1) + abs(angle2)
            if include_angle > 180:
                include_angle = 360 - include_angle
        self.angle = include_angle

    def get_pan(self,xlist,ylist):
        self.delta_x = float(xlist[-1]) - self.pointx
        self.delta_y = float(ylist[-1]) - self.pointy

    def get_spindirection(self,xlist,ylist):
        startx = xlist[-2]
        endx = xlist[-1]
        starty = ylist[-2]
        endy = ylist[-1]

        if (endy - starty) >= 0:
            self.direction = "S"
        else:
            self.direction = 'N'

    def get_panpoint(self,xlist,ylist):
        rel_x = []
        rel_y = []
        for i in range(len(xlist)):
            x = xlist[i] - self.delta_x
            y = ylist[i] - self.delta_y
            rel_x.append(x)
            rel_y.append(y)
        return rel_x, rel_y

    def get_spinpoint(self,xlist,ylist):
        angle = math.radians(self.angle)
        spinx = []
        spiny = []

        if self.direction == "S":
            for i in range(len(xlist)):
                x, y = self.Srotate(angle, xlist[i], ylist[i], self.pointx, self.pointy)
                spinx.append(x)
                spiny.append(y)
        else:
            for i in range(len(xlist)):
                x, y = self.Nrotate(angle, xlist[i], ylist[i], self.pointx, self.pointy)
                spinx.append(x)
                spiny.append(y)

        return spinx,spiny

if __name__ == "__main__":
    ts = traSpin()
    ts.calangle(0,0,-1,-1.3)