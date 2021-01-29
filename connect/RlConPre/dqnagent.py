import logging
import torch

from memory import Transition
from models import MultitaskNetwork, trainable_parameters
from optimizers import loss_function_factory, optimizer_factory
from utils import choose_device
from agent import AbstractDQNAgent

logger = logging.getLogger(__name__)


class DQNAgent(AbstractDQNAgent):
    def __init__(self, env, config=None):
        super(DQNAgent, self).__init__(env, config)
        self.value_net = MultitaskNetwork(self.config["model"])
        self.target_net = MultitaskNetwork(self.config["model"])
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()
        
        logger.debug("Number of trainable parameters: {}".format(trainable_parameters(self.value_net)))
        self.device = choose_device(self.config["device"])
        self.value_net.to(self.device)
        self.target_net.to(self.device)
        
        self.rl_lossFunction = loss_function_factory(self.config["rl_lossfunction"])
        self.predict_lossfunction = loss_function_factory(self.config['predict_lossfunction'])

        self.rl_optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.rl_updatePara(),
                                           **self.config["optimizer"])
        
        self.pre_optimizer = optimizer_factory(self.config["optimizer"]["type"],
                                           self.value_net.pre_updatePara(),
                                           **self.config["optimizer"])
        
        self.steps = 0

    def step_optimizer(self, loss, optimizer):
        # Optimize the model
        optimizer.zero_grad()
        loss.backward(retain_graph = True)

    def compute_bellman_residual(self, batch, target_state_action_value=None):
        # Compute concatenate the batch elements
        if not isinstance(batch.current_state, torch.Tensor):
            # logger.info("Casting the batch to torch.tensor")
            current_state = torch.cat(tuple(torch.tensor([batch.current_state], dtype=torch.float))).to(self.device)
            current_future_pos = torch.cat(tuple(torch.tensor([batch.current_future_pos], dtype=torch.float))).to(self.device)
            current_past_pos = torch.cat(tuple(torch.tensor([batch.current_past_pos], dtype=torch.float))).to(self.device)
            
            action = torch.tensor(batch.action, dtype=torch.long).to(self.device)
            reward = torch.tensor(batch.reward, dtype=torch.float).to(self.device)
            
            next_state = torch.cat(tuple(torch.tensor([batch.next_state], dtype=torch.float))).to(self.device)
            next_future_pos = torch.cat(tuple(torch.tensor([batch.next_future_pos], dtype=torch.float))).to(self.device)
            next_past_pos = torch.cat(tuple(torch.tensor([batch.next_past_pos], dtype=torch.float))).to(self.device)
            
            terminal = torch.tensor(batch.terminal, dtype=torch.bool).to(self.device)
            batch = Transition(current_state, current_future_pos, current_past_pos,\
                action, reward,\
                    next_state, next_future_pos, next_past_pos,\
                        terminal, batch.info)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        current_state_action_values,current_trajectory = self.value_net(batch.current_state,batch.current_past_pos)
        state_action_values = current_state_action_values.gather(1, batch.action.unsqueeze(1)).squeeze(1)

        if target_state_action_value is None:
            with torch.no_grad():
                # Compute V(s_{t+1}) for all next states.
                next_state_values = torch.zeros(batch.reward.shape).to(self.device)
                if self.config["double"]:
                    # Double Q-learning: pick best actions from policy network
                    next_state_action_values,next_trajectory = self.value_net(batch.next_state,batch.next_past_pos)
                    _, best_actions = next_state_action_values.max(1)
                    # Double Q-learning: estimate action values from target network
                    next_target_state_action_values ,next_target_tragectory= self.target_net(batch.next_state,batch.next_past_pos)
                    best_values = next_target_state_action_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
                else:
                    next_state_action_values,next_trajectory = self.target_net(batch.next_state,bacth.next_past_pos)
                    best_values, _ = next_state_action_values.max(1)
                next_state_values[~batch.terminal] = best_values[~batch.terminal]
                # Compute the expected Q values
                target_state_action_value = batch.reward + self.config["gamma"] * next_state_values

        # Compute loss
        rl_loss = self.rl_lossFunction(state_action_values, target_state_action_value)
        predict_loss = self.predict_lossfunction(current_trajectory,batch.current_future_pos)
        
        self.writer.add_scalar('step/rl_loss',rl_loss,self.step)
        self.writer.add_scalar('step/predict_loss',predict_loss,self.step)
        
        return rl_loss,predict_loss, target_state_action_value, batch

    def get_batch_state_values(self, states):
        values, actions = self.value_net(torch.tensor(states, dtype=torch.float).to(self.device)).max(1)
        return values.data.cpu().numpy(), actions.data.cpu().numpy()

    def get_batch_state_action_values(self, current_state, current_past_pos):
        values,trajectorys = self.value_net(torch.tensor(current_state, dtype=torch.float).to(self.device),\
            torch.tensor(current_past_pos, dtype=torch.float).to(self.device))
        values = values.data.cpu().numpy()
        return values
    
    def save(self, filename):
        state = {'state_dict': self.value_net.state_dict(),
                 'rl_optimizer': self.rl_optimizer.state_dict(),
                 'pre_optimizer': self.pre_optimizer.state_dict()}
        torch.save(state, filename)
        return filename

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['state_dict'])
        self.target_net.load_state_dict(checkpoint['state_dict'])
        self.rl_optimizer.load_state_dict(checkpoint['rl_optimizer'])
        self.pre_optimizer.load_state_dict(checkpoint['pre_optimizer'])
        return filename

    def initialize_model(self):
        self.value_net.reset()

    def set_writer(self, writer):
        super().set_writer(writer)
        self.writer.add_scalar("agent/trainable_parameters", trainable_parameters(self.value_net), 0)
