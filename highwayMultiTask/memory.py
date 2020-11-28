import random
from collections.__init__ import namedtuple
import collections
from gym.core import Env

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'terminal', 'trajectory', 'info'))

class Configurable(object):
    """
        This class is a container for a configuration dictionary.
        It allows to provide a default_config function with pre-filled configuration.
        When provided with an input configuration, the default one will recursively be updated,
        and the input configuration will also be updated with the resulting configuration.
    """
    def __init__(self, config=None):
        self.config = self.default_config()
        if config:
            # Override default config with variant
            Configurable.rec_update(self.config, config)
            # Override incomplete variant with completed variant
            Configurable.rec_update(config, self.config)

    @classmethod
    def default_config(cls):
        """
            Override this function to provide the default configuration of the child class
        :return: a configuration dictionary
        """
        return {}

    @staticmethod
    def rec_update(d, u):
        """
            Recursive update of a mapping
        :param d: a mapping
        :param u: a mapping
        :return: d updated recursively with u
        """
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = Configurable.rec_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

class ReplayMemory(Configurable):
    """
        Container that stores and samples transitions.
    """
    def __init__(self, config=None, transition_type=Transition):
        super(ReplayMemory, self).__init__(config)
        self.capacity = int(self.config['memory_capacity'])
        self.transition_type = transition_type
        self.memory = []
        self.position = 0

    @classmethod
    def default_config(cls):
        return dict(memory_capacity=10000,
                    n_steps=1,
                    gamma=0.99)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.position = len(self.memory) - 1
        elif len(self.memory) > self.capacity:
            self.memory = self.memory[:self.capacity]
        # Faster than append and pop
        self.memory[self.position] = self.transition_type(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, collapsed=True):
        """
            Sample a batch of transitions.

            If n_steps is greater than one, the batch will be composed of lists of successive transitions.
        :param batch_size: size of the batch
        :param collapsed: whether successive transitions must be collapsed into one n-step transition.
        :return: the sampled batch
        """
        # TODO: use agent's np_random for seeding
        if self.config["n_steps"] == 1:
            # Directly sample transitions
            return random.sample(self.memory, batch_size)
        else:
            # Sample initial transition indexes
            indexes = random.sample(range(len(self.memory)), batch_size)
            # Get the batch of n-consecutive-transitions starting from sampled indexes
            all_transitions = [self.memory[i:i+self.config["n_steps"]] for i in indexes]
            # Collapse transitions
            return map(self.collapse_n_steps, all_transitions) if collapsed else all_transitions

    def collapse_n_steps(self, transitions):
        """
            Collapse n transitions <s,a,r,s',t> of a trajectory into one transition <s0, a0, Sum(r_i), sp, tp>.

            We start from the initial state, perform the first action, and then the return estimate is formed by
            accumulating the discounted rewards along the trajectory until a terminal state or the end of the
            trajectory is reached.
        :param transitions: A list of n successive transitions
        :return: The corresponding n-step transition
        """
        state, action, cumulated_reward, next_state, done, info = transitions[0]
        discount = 1
        for transition in transitions[1:]:
            if done:
                break
            else:
                _, _, reward, next_state, done, info = transition
                discount *= self.config['gamma']
                cumulated_reward += discount*reward
        return state, action, cumulated_reward, next_state, done, info

    def __len__(self):
        return len(self.memory)

    def is_full(self):
        return len(self.memory) == self.capacity

    def is_empty(self):
        return len(self.memory) == 0