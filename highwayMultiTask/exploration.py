import collections
from gym import spaces
from gym.utils import seeding
import numpy as np
from abc import ABC,abstractmethod

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

class DiscreteDistribution(Configurable, ABC):
    def __init__(self, config=None, **kwargs):
        super(DiscreteDistribution, self).__init__(config)
        self.np_random = None

    @abstractmethod
    def get_distribution(self):
        """
        :return: a distribution over actions {action:probability}
        """
        raise NotImplementedError()

    def sample(self):
        """
        :return: an action sampled from the distribution
        """
        distribution = self.get_distribution()
        return self.np_random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

    def seed(self, seed=None):
        """
            Seed the policy randomness source
        :param seed: the seed to be used
        :return: the used seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_time(self, time):
        """ Set the local time, allowing to schedule the distribution temperature. """
        pass

class Greedy(DiscreteDistribution):
    """
        Always use the optimal action
    """

    def __init__(self, action_space, config=None):
        super(Greedy, self).__init__(config)
        self.action_space = action_space
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.values = None
        self.seed()

    def get_distribution(self):
        optimal_action = np.argmax(self.values)
        return {action: 1 if action == optimal_action else 0 for action in range(self.action_space.n)}

    def update(self, values, step_time=False):
        self.values = values

class EpsilonGreedy(DiscreteDistribution):
    """
        Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
    """

    def __init__(self, action_space, config=None):
        super(EpsilonGreedy, self).__init__(config)
        self.action_space = action_space
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.config['final_temperature'] = min(self.config['temperature'], self.config['final_temperature'])
        self.optimal_action = None
        self.epsilon = 0
        self.time = 0
        self.writer = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(temperature=1.0,
                    final_temperature=0.1,
                    tau=5000)

    def get_distribution(self):
        distribution = {action: self.epsilon / self.action_space.n for action in range(self.action_space.n)}
        distribution[self.optimal_action] += 1 - self.epsilon
        return distribution

    def update(self, values, step_time=True):
        """
            Update the action distribution parameters
        :param values: the state-action values
        :param step_time: whether to update epsilon schedule
        """
        self.optimal_action = np.argmax(values)
        self.epsilon = self.config['final_temperature'] + \
            (self.config['temperature'] - self.config['final_temperature']) * \
            np.exp(- self.time / self.config['tau'])
        if step_time:
            self.time += 1
        if self.writer:
            self.writer.add_scalar('exploration/epsilon', self.epsilon, self.time)

    def set_time(self, time):
        self.time = time

    def set_writer(self, writer):
        self.writer = writer

class Boltzmann(DiscreteDistribution):
    """
        Uniform distribution with probability epsilon, and optimal action with probability 1-epsilon
    """

    def __init__(self, action_space, config=None):
        super(Boltzmann, self).__init__(config)
        self.action_space = action_space
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("The action space should be discrete")
        self.values = None
        self.seed()

    @classmethod
    def default_config(cls):
        return dict(temperature=0.5)

    def get_distribution(self):
        actions = range(self.action_space.n)
        if self.config['temperature'] > 0:
            weights = np.exp(self.values / self.config['temperature'])
        else:
            weights = np.zeros((len(actions),))
            weights[np.argmax(self.values)] = 1
        return {action: weights[action] / np.sum(weights) for action in actions}

    def update(self, values, step_time=False):
        self.values = values

def exploration_factory(exploration_config, action_space):
    """
        Handles creation of exploration policies
    :param exploration_config: configuration dictionary of the policy, must contain a "method" key
    :param action_space: the environment action space
    :return: a new exploration policy
    """
    if exploration_config['method'] == 'Greedy':
        return Greedy(action_space, exploration_config)
    elif exploration_config['method'] == 'EpsilonGreedy':
        return EpsilonGreedy(action_space, exploration_config)
    elif exploration_config['method'] == 'Boltzmann':
        return Boltzmann(action_space, exploration_config)
    else:
        raise ValueError("Unknown exploration method")