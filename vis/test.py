import os
import torch 
import numpy as np
from model import MultitaskNetwork
import matplotlib.pyplot as plt

class test(object):
    def __init__(self,weight_path = None,save_path = None):
        self.config ={'encoder':{
            "in_channels": 5,
            "in_height": 112,
            "in_width": 112,
            }
        }
        self.weight_path = weight_path
        self.model = MultitaskNetwork(self.config)
        self.model.eval()
        self.save_path = save_path
        self.numimg = 0

        self.max_speed = 180
        self.lane_width = 16

    def renormalize(self,value,bound):
        delax = 2 * bound
        delay = 2
        return ((value + 1) * delax) / delay + bound


    def get_result(self,input):
        output = self.model(input)
        outputs = output.data.cpu().numpy()
        return outputs

    def load_dict(self):
        checkpoint = torch.load(self.weight_path,map_location=torch.device('cpu'))
        pretrained_dict = checkpoint['state_dict']
        
        model_dict = self.model.state_dict()
        model_state_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict.keys()}

        model_dict.update(model_state_dict)
        self.model.load_state_dict(model_dict)
        self.model.load_state_dict(model_dict)

    def reshape(self,inputs):
        inputs = inputs.reshape(-1,2)
        car1_x = []
        car1_y = []

        car2_x = []
        car2_y = []

        car3_x = []
        car3_y = []

        car4_x = []
        car4_y = []

        car5_x = []
        car5_y = []

        carx = []
        cary = []

        for i in range(10):
            car1_x.append(self.renormalize(inputs[i * 5][0],self.max_speed))
            car1_y.append(self.renormalize(inputs[i * 5 + 1][1],self.lane_width))

            car2_x.append(self.renormalize(inputs[i * 5 + 1][0],self.max_speed))
            car2_y.append(self.renormalize(inputs[i * 5 + 1][1],self.lane_width))

            car3_x.append(self.renormalize(inputs[i * 5 + 2][0],self.max_speed))
            car3_y.append(self.renormalize(inputs[i * 5 + 2][1],self.lane_width))

            car4_x.append(self.renormalize(inputs[i * 5 + 3][0],self.max_speed))
            car4_y.append(self.renormalize(inputs[i * 5 + 3][1],self.lane_width))

            car5_x.append(self.renormalize(inputs[i * 5 + 4][0],self.max_speed))
            car5_y.append(self.renormalize(inputs[i * 5 + 4][1],self.lane_width))
        
        carx.append(car1_x)
        carx.append(car2_x)
        carx.append(car3_x)
        carx.append(car4_x)
        carx.append(car5_x)

        cary.append(car1_y)
        cary.append(car2_y)
        cary.append(car3_y)
        cary.append(car4_y)
        cary.append(car5_y)
        
        return carx,cary

    def render_future(self,labelx_future,labely_future,predictx_future,predicty_future):
        fig,ax = plt.subplots()
        for i in range(5):
            ax.plot(np.array(labelx_future[i]) + 112,np.array(labely_future[i]) + 112,label = 'label',color = 'r')
            ax.plot(np.array(predicty_future[i]) + 112,np.array(predicty_future[i]) + 112,label = 'predict',linestyle = "--",color = 'b')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.set_xlim(0,224)
        ax.set_ylim(0,224)

        filename = self.save_path + str(self.numimg)+".png"
        plt.savefig(filename)
        self.numimg += 1
        plt.savefig(filename)

if __name__ == "__main__":
    save_path = './vis'
    t = test(save_path=save_path)
    pastPosition = torch.randn(1,5,112,112)
    output = t.get_result(pastPosition)
    
    pastPosition = torch.randn(1,5,112,112)
    output2 = t.get_result(pastPosition)

    carx,cary = t.reshape(output)
    prex,prey = t.reshape(output2)

    t.render_future(carx,cary,prex,prey)

    print(carx)
    print(cary)