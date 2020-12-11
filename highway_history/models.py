import numpy as np
import torch
import torch.nn as nn
import collections
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from configuration import Configurable

def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))

def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def size_model_config(env,model_config):
    model_config["action"] = env.action_space.n

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class BaseModule(torch.nn.Module):
    """
        Base torch.nn.Module implementing basic features:
            - initialization factory
            - normalization parameters
    """
    def __init__(self, activation_type="RELU", reset_type="XAVIER", normalize=None):
        super().__init__()
        self.activation = activation_factory(activation_type)
        self.reset_type = reset_type
        self.normalize = normalize
        self.mean = None
        self.std = None

    def _init_weights(self, m):
        if hasattr(m, 'weight'):
            if self.reset_type == "XAVIER":
                torch.nn.init.xavier_uniform_(m.weight.data)
            elif self.reset_type == "ZEROS":
                torch.nn.init.constant_(m.weight.data, 0.)
            else:
                raise ValueError("Unknown reset type")
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.)

    def set_normalization_params(self, mean, std):
        if self.normalize:
            std[std == 0.] = 1.
        self.std = std
        self.mean = mean

    def reset(self):
        self.apply(self._init_weights)

    def forward(self, *input):
        if self.normalize:
            input = (input.float() - self.mean.float()) / self.std.float()
        return NotImplementedError

class VGG(BaseModule,Configurable):
    def __init__(self,architecture,config):
        super().__init__()
        Configurable.__init__(self,config)
        
        self.encoder = self.make_layers(architecture, self.config["in_channels"])
        self.action = nn.Sequential(
            nn.Linear(512 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, config['action']),
        )

    def make_layers(self, cfg, in_channels, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self,x):
        latent = self.encoder(x)
        latent = latent.view(latent.size(0), -1)
        action = self.action(latent)

        return action

    @classmethod
    def default_config(cls):
        return {
            "trajectory_prediction" : 60,
            "action" : 3,
            "in_channels" : 40,
            "model" : 'E'
        }

def vgg11(config, pretrained=False, model_root=None):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(cfg['A'], config)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(config, pretrained=False, model_root=None):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), config)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(config, pretrained=False, model_root=None):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), config)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(config, pretrained=False, model_root=None):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(cfg['E'], config)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)

class TrajectoryFCN(BaseModule,Configurable):
    def __init__(self,config):
        super().__init__()
        Configurable.__init__(self,config)
        
        
        self.encoder_obs = nn.Sequential(
            nn.Conv1d(int(self.config["in_channels"] / 2),32,kernel_size = (112,1)),
            nn.ReLU(inplace = True)
        )

        self.encoder_ego = nn.Sequential(
            nn.Conv1d(int(self.config["in_channels"] / 2),32,kernel_size = (112,1)),
            nn.ReLU(inplace = True)
        )

        # self.encoder_map = nn.Sequential(
        #     nn.Conv1d(3,12,kernel_size = (224,1)),
        #     nn.ReLU(inplace = True)
        # )

        self.predict = nn.Sequential(
            nn.Linear(32 * 1 * 112 + 32 * 1 * 112, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, 60),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(60,self.config["action"])
        )
        self.reset()

    def forward(self, x):
        ego = x[:,:10:,:,:]
        obs = x[:,10:,:,:]
        #maps = x[:,42:,:,:]

        latent_obs = self.encoder_obs(obs)
        latent_ego = self.encoder_ego(ego)
        #latent_maps = self.encoder_map(maps)

        latent_obs = latent_obs.view(latent_obs.size(0), -1)
        latent_ego = latent_ego.view(latent_ego.size(0), -1)
        #latent_maps = latent_maps.view(latent_maps.size(0), -1)

        latent = torch.cat((latent_ego,latent_obs),1)


        predict = self.predict(latent)
        return predict

    @classmethod
    def default_config(cls):
        return {
            "action" : 3,
            "in_channels" : 20,
        }


def model_factory(config: dict) -> nn.Module:
    if config["type"] == "VGG11":
        return vgg11(config)
    elif config["type"] == "VGG13":
        return vgg13(config)
    elif config["type"] == "VGG16":
        return vgg16(config)
    elif config["type"] == "VGG19":
        return vgg19(config)
    elif config["type"] == "VGG19":
        return vgg19(config)
    elif config["type"] == "FCN":
        return TrajectoryFCN(config)
    else:
        raise ValueError("Unknown model type")


if __name__ == "__main__":
    config = {
        "action" : 5
    }
    model = vgg11(config, pretrained = False)
    print(model)