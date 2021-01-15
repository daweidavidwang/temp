import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F

from configuration import Configurable

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

def size_model_config(env, model_config):
    """
        Update the configuration of a model depending on the environment observation/action spaces

        Typically, the input/output sizes.

    :param env: an environment
    :param model_config: a model configuration
    """
    if model_config["type"] == "ConvolutionalNetwork":  # Assume CHW observation space
        model_config["in_channels"] = int(env.observation_space.shape[2])
        model_config["in_height"] = int(env.observation_space.shape[0])
        model_config["in_width"] = int(env.observation_space.shape[1])
    else:
        model_config["in"] = int(np.prod(env.observation_space.shape))
    model_config["out"] = env.action_space.n

def trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def activation_factory(activation_type):
    if activation_type == "RELU":
        return F.relu
    elif activation_type == "TANH":
        return torch.tanh
    else:
        raise ValueError("Unknown activation_type: {}".format(activation_type))

class MultiLayerPerceptron(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        sizes = [self.config['in']] + self.config["layers"]
        self.activation = activation_factory(self.config["activation"])
        layers_list = [nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.layers = nn.ModuleList(layers_list)
        if self.config.get("out", None):
            self.predict = nn.Linear(sizes[-1], self.config["out"])
        
    @classmethod
    def default_config(cls):
        return {"in": None,
                "layers": [256,512,256],
                "activation": "RELU",
                "reshape": "True",
                "out": None,
                }

    def forward(self,x):
        if self.config["reshape"]:
            x = x.reshape(x.shape[0], -1)  # We expect a batch of vectors
        for layer in self.layers:
            x = self.activation(layer(x))
        if self.config.get("out", None):
            values = self.predict(x)
        return values

class ConvolutionalNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.activation = activation_factory(self.config["activation"])
        self.conv1 = nn.Conv2d(self.config["in_channels"], 16, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=2, stride=2)
        # MLP Head
        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        self.convw = conv2d_size_out \
            (conv2d_size_out\
                (conv2d_size_out \
                    (conv2d_size_out\
                    (self.config["in_width"]))))
        self.convh = conv2d_size_out \
            (conv2d_size_out\
                (conv2d_size_out \
                    (conv2d_size_out\
                    (self.config["in_height"]))))
    
    @classmethod
    def default_config(cls):
        return {
            "in_channels": None,
            "in_height": None,
            "in_width": None,
            "activation": "RELU",
        }

    def forward(self,pastPosition):
        """
            Forward convolutional network
        :param x: tensor of shape BCHW
        """
        pastPosition = self.activation((self.conv1(pastPosition)))
        pastPosition = self.activation((self.conv2(pastPosition)))
        pastPosition = self.activation((self.conv3(pastPosition)))
        pastPosition = self.activation((self.conv4(pastPosition)))
        return pastPosition

class MultitaskNetwork(BaseModule, Configurable):
    def __init__(self, config):
        super().__init__()
        Configurable.__init__(self, config)
        self.encoder = model_factory(self.config["encoder"])
        self.rl = model_factory(self.config["rl_mlp"])
        
        self.reset()
    
    def reset(self):
        self.encoder.reset()
        self.rl.reset()

    @classmethod
    def default_config(cls):
        return {
            "encoder":{
                "type": "ConvolutionalNetwork",
                "in_channels": None,
                "in_height": None,
                "in_width": None,
                "activation": "RELU",
            },
            "rl_mlp": {
                "type": "MultiLayerPerceptron",
                "in": 1593,
                "layers": [256,256],
                "activation": "RELU",
                "reshape": False,
                "out": 5
            },
        }
    
    def forward(self,currentState,pastPosition):
        pastPosition = self.encoder(pastPosition)
        pastPosition = pastPosition.reshape(pastPosition.shape[0],-1)
        rlInputs = torch.cat(
            (pastPosition,currentState.reshape(currentState.shape[0],-1)),1
        )
        rlOutputs = self.rl(rlInputs)

        return rlOutputs

def model_factory(config: dict) -> nn.Module:
    if config["type"] == "MultiLayerPerceptron":
        return MultiLayerPerceptron(config)
    if config["type"] == "ConvolutionalNetwork":
        return ConvolutionalNetwork(config)
    else:
        raise ValueError("Unknown model type")

if __name__ == "__main__":
    config ={'encoder':{
            "in_channels": 6,
            "in_height": 112,
            "in_width": 112,
        }
    }
    #model = ConvolutionalNetwork(config)
    model = MultitaskNetwork(config)
    print(model)
    pastPosition = torch.randn(1,6,112,112)
    current_state = torch.randn(1,5,5)
    rloutputs = model(current_state,pastPosition)
    print(rloutputs)
