import logging
import torch

from utils import choose_device
from memory import Transition
from agent import AbstractDQNAgent
from model import model_factory,size_model_config,trainable_parameters
from optimizers import loss_function_factory, optimizer_factory

logger = logging.getLogger(__name__)

class DQNAgent(AbstractDQNAgent):
    def __init__(self,env,config = None):
        super(DQNAgent,self).__init__(env,config)
        size_model_config(self.env,self.config["model"])

        self.value_net = model_factory(self.config["model"])
        self.target_net = model_factory(self.config["model"])
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        
        logger.debug("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))
        self.device = choose_device(self.config["device"])
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_function = loss_function_factory(self.config["loss_function"])
        self.optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.parameters(),
                                           **self.config["optimizer"])
        self.steps = 0

    def step_optimizer(self,loss):
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            state = torch.cat(tuple(torch.tensor([batch.state], dtype=torch.float))).to(self.device)
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            prediction = torch.cat(tuple(torch.tensor([batch.trajectory], dtype=torch.float))).to(self.device)
            batch = Transition(state, action, reward, next_state, terminal, prediction,batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        trajectory_prediction, state_action_value = self.value_net(batch.state)
        state_action_values = state_action_value.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    prediction, state_action_value = self.value_net(batch.next_state)
                    _,best_actions = state_action_value.max(1)
                    # Double Q-learning: estimate action values from target network
                    target_prediction,target_state_action_values = self.target_net(batch.next_state)
                    best_values = target_state_action_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    target_prediction,target_state_action_values = self.target_net(batch.next_state)
                    best_values, _ = target_state_action_values.max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        prediction_loss = self.loss_function(trajectory_prediction,prediction)
        RL_loss = self.loss_function(state_action_values, target_state_action_value)
        print(prediction_loss + RL_loss)
        return RL_loss + prediction_loss , target_state_action_value, batch

    def get_batch_state_values(self, states):
        prediction, state_action_values = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device))
        values, actions = state_action_values.max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, states):
        prediction, state_action_values = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device))
        return state_action_values.data.cpu().numpy()

    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)

