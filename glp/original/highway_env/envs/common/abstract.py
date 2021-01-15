import copy
import os
from typing import List, Tuple, Optional, Callable
import gym
from gym import Wrapper
import pandas as pd
from gym.utils import seeding
import numpy as np
import cv2 as cv
from collections import deque

from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.finite_mdp import finite_mdp
from highway_env.envs.common.graphics import EnvViewer
from highway_env.vehicle.behavior import IDMVehicle, LinearVehicle
from highway_env.vehicle.controller import MDPVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class AbstractEnv(gym.Env):

    """
    A generic environment for various tasks involving a vehicle driving on a road.

    The environment contains a road populated with vehicles, and a controlled ego-vehicle that can change lane and
    speed. The action space is fixed, but the observation space and reward function must be defined in the
    environment implementations.
    """
    observation_type: ObservationType
    action_type: ActionType
    automatic_rendering_callback: Optional[Callable]
    metadata = {'render.modes': ['human', 'rgb_array']}

    PERCEPTION_DISTANCE = 6.0 * MDPVehicle.SPEED_MAX
    """The maximum distance of any vehicle present in the observation [m]"""
    FEATURES: List[str] = ['presence', 'x', 'y', 'vx', 'vy']

    def __init__(self, config: dict = None) -> None:
        # Configuration
        self.config = self.default_config()
        if config:
            self.config.update(config)

        self.features = self.FEATURES
        # Seeding
        self.np_random = None
        self.seed()

        # Scene
        self.road = None
        self.controlled_vehicles = []

        # Spaces
        self.action_type = None
        self.action_space = None
        self.observation_type = None
        self.observation_space = None
        self.define_spaces()

        # Running
        self.time = 0  # Simulation time
        self.steps = 0  # Actions performed
        self.done = False

        # Rendering
        self.viewer = None
        self.automatic_rendering_callback = None
        self.should_update_rendering = True
        self.rendering_mode = 'human'
        self.enable_auto_render = False

        self.record_past = deque(maxlen = 5)

        self.store_img = False
        self.numimg = 0

        self.reset()

    @property
    def vehicle(self) -> Vehicle:
        """First (default) controlled vehicle."""
        return self.controlled_vehicles[0] if self.controlled_vehicles else None

    @vehicle.setter
    def vehicle(self, vehicle: Vehicle) -> None:
        """Set a unique controlled vehicle."""
        self.controlled_vehicles = [vehicle]

    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().
        :return: a configuration dict
        """
        return {
            "observation": {
                "type": "TimeToCollision"
            },
            "action": {
                "type": "DiscreteMetaAction"
            },
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 1,  # [Hz]
            "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
            "screen_width": 600,  # [px]
            "screen_height": 150,  # [px]
            "centering_position": [0.3, 0.5],
            "scaling": 5.5,
            "show_trajectories": False,
            "render_agent": True,
            "offscreen_rendering": os.environ.get("OFFSCREEN_RENDERING", "0") == "1",
            "manual_control": False,
            "real_time_rendering": False
        }

    def seed(self, seed: int = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def configure(self, config: dict) -> None:
        if config:
            self.config.update(config)

    def define_spaces(self) -> None:
        """
        Set the types and spaces of observation and action from config.
        """
        self.observation_type = observation_factory(self, self.config["observation"])
        self.action_type = action_factory(self, self.config["action"])
        self.observation_space = self.observation_type.space()
        self.action_space = self.action_type.space()

    def _reward(self, action: Action) -> float:
        """
        Return the reward associated with performing a given action and ending up in the current state.

        :param action: the last action performed
        :return: the reward
        """
        raise NotImplementedError

    def _is_terminal(self) -> bool:
        """
        Check whether the current state is a terminal state

        :return:is the state terminal
        """
        raise NotImplementedError

    def _cost(self, action: Action) -> float:
        """
        A constraint metric, for budgeted MDP.

        If a constraint is defined, it must be used with an alternate reward that doesn't contain it as a penalty.
        :param action: the last action performed
        :return: the constraint signal, the alternate (constraint-free) reward
        """
        raise NotImplementedError

    def reset(self) -> Observation:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.define_spaces()  # First, to set the controlled vehicle class depending on action space
        self.time = self.steps = 0
        self.done = False
        self._reset()
        self.define_spaces()

        pastPos = np.zeros((112,112,6)).reshape(6,112,112) 
        obs = self.observation_type.observe().reshape(5,5,1)
        # Second, to link the obs and actions to the vehicles once the scene is created
        return obs,pastPos

    def _reset(self) -> None:
        """
        Reset the scene: roads and vehicles.

        This method must be overloaded by the environments.
        """
        raise NotImplementedError()

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or self.vehicle is None:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self.steps += 1

        self._simulate(action)

        pastPos = self._renderPastPosition().reshape(6,112,112)
        obs = self.observation_type.observe().reshape(5,5,-1)
        
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "speed": self.vehicle.speed,
            "crashed": self.vehicle.crashed,
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        return obs, pastPos, reward, terminal, info
    
    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for i in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            
            if i % 3 == 0:
                self.record_past.append(
                    copy.deepcopy(self.road.vehicles)
                )

            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False
    
    def _renderPastPosition(self):
        point_size = 1
        point_color = (225, 225, 225) # BGR
        thickness = 4
        savepath = "./vis/"

        pastPosition = self._getPastPosition()
        
        ego_pointslist = []
        img_list = []
        for tp in pastPosition:
            points_list = []
            img = np.ones((224,224,3),np.uint8)*0
            for index,ip in enumerate(tp):
                if index > 0:
                    x_pos = int(round(ip[0])) + 112
                    y_pos = int(round(ip[1])) + 112
                    points_list.append((x_pos,y_pos))
                else:
                    x_pos = int(round(ip[0])) + 112
                    y_pos = int(round(ip[1])) + 112
                    ego_pointslist.append((x_pos,y_pos))
            
            for point in points_list:
                cv.circle(img, point, point_size, point_color, thickness)

            img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
            img_new = cv.resize(img_gray,(112,112))

            if self.store_img:
                filename = savepath + str(self.numimg)+".png"
                cv.imwrite(filename,img_new)
                self.numimg += 1
            
            img_list.append(img_new)
        
        img = np.ones((224,224,3),np.uint8)*0
        for point in ego_pointslist:
            cv.circle(img, point, point_size, point_color, thickness)
        img_gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
        img_new = cv.resize(img_gray,(112,112))

        if self.store_img:
            filename = savepath + str(self.numimg)+".png"
            cv.imwrite(filename,img_new)
            self.numimg += 1
        
        img_list.append(img_new)

        
        array_tuple = tuple(img_list)
        input_array = np.stack(
            (array_tuple),axis = 2
        )
        return input_array
    
    def _getClostestVehicles(self,vehicles_list):
        time_postion = []
        ego_vehicles = pd.DataFrame.from_records([self.road.vehicles[0].to_dict()])[self.features].values.copy()
        for vehicles in vehicles_list:
            vehiInfo = pd.DataFrame.from_records([vehicles.to_dict()])[self.features].values.copy()
            relx = vehiInfo[0][1] - ego_vehicles[0][1]
            rely = vehiInfo[0][2] - ego_vehicles[0][2]

            if abs(relx) <= 112:
                time_postion.append([relx,rely])
        return time_postion

    def _getPastPosition(self):
        pastPosition = []
        
        for i in range(len(self.record_past)):
            pastPosition.append(self._getClostestVehicles(
                self.record_past[i]
            ))
        
        return pastPosition

    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.

        Create a viewer if none exists, and use it to render an image.
        :param mode: the rendering mode
        """
        self.rendering_mode = mode

        if self.viewer is None:
            self.viewer = EnvViewer(self)

        self.enable_auto_render = True

        # If the frame has already been rendered, do nothing
        if self.should_update_rendering:
            self.viewer.display()

        if not self.viewer.offscreen:
            self.viewer.handle_events()
        if mode == 'rgb_array':
            image = self.viewer.get_image()
            return image
        self.should_update_rendering = False

    def close(self) -> None:
        """
        Close the environment.

        Will close the environment viewer if it exists.
        """
        self.done = True
        if self.viewer is not None:
            self.viewer.close()
        self.viewer = None

    def get_available_actions(self) -> List[int]:
        """
        Get the list of currently available actions.

        Lane changes are not available on the boundary of the road, and speed changes are not available at
        maximal or minimal speed.

        :return: the list of available actions
        """
        if not isinstance(self.action_type, DiscreteMetaAction):
            raise ValueError("Only discrete meta-actions can be unavailable.")
        actions = [self.action_type.actions_indexes['IDLE']]
        for l_index in self.road.network.side_lanes(self.vehicle.lane_index):
            if l_index[2] < self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_LEFT'])
            if l_index[2] > self.vehicle.lane_index[2] \
                    and self.road.network.get_lane(l_index).is_reachable_from(self.vehicle.position) \
                    and self.action_type.lateral:
                actions.append(self.action_type.actions_indexes['LANE_RIGHT'])
        if self.vehicle.speed_index < self.vehicle.SPEED_COUNT - 1 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['FASTER'])
        if self.vehicle.speed_index > 0 and self.action_type.longitudinal:
            actions.append(self.action_type.actions_indexes['SLOWER'])
        return actions

    def _automatic_rendering(self) -> None:
        """
        Automatically render the intermediate frames while an action is still ongoing.

        This allows to render the whole video and not only single steps corresponding to agent decision-making.

        If a callback has been set, use it to perform the rendering. This is useful for the environment wrappers
        such as video-recording monitor that need to access these intermediate renderings.
        """
        if self.viewer is not None and self.enable_auto_render:
            self.should_update_rendering = True

            if self.automatic_rendering_callback is not None:
                self.automatic_rendering_callback()
            else:
                self.render(self.rendering_mode)

    def simplify(self) -> 'AbstractEnv':
        """
        Return a simplified copy of the environment where distant vehicles have been removed from the road.

        This is meant to lower the policy computational load while preserving the optimal actions set.

        :return: a simplified environment state
        """
        state_copy = copy.deepcopy(self)
        state_copy.road.vehicles = [state_copy.vehicle] + state_copy.road.close_vehicles_to(
            state_copy.vehicle, self.PERCEPTION_DISTANCE)

        return state_copy

    def change_vehicles(self, vehicle_class_path: str) -> 'AbstractEnv':
        """
        Change the type of all vehicles on the road

        :param vehicle_class_path: The path of the class of behavior for other vehicles
                             Example: "highway_env.vehicle.behavior.IDMVehicle"
        :return: a new environment with modified behavior model for other vehicles
        """
        vehicle_class = utils.class_from_path(vehicle_class_path)

        env_copy = copy.deepcopy(self)
        vehicles = env_copy.road.vehicles
        for i, v in enumerate(vehicles):
            if v is not env_copy.vehicle:
                vehicles[i] = vehicle_class.create_from(v)
        return env_copy

    def set_preferred_lane(self, preferred_lane: int = None) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        if preferred_lane:
            for v in env_copy.road.vehicles:
                if isinstance(v, IDMVehicle):
                    v.route = [(lane[0], lane[1], preferred_lane) for lane in v.route]
                    # Vehicle with lane preference are also less cautious
                    v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 1000
        return env_copy

    def set_route_at_intersection(self, _to: str) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.set_route_at_intersection(_to)
        return env_copy

    def set_vehicle_field(self, args: Tuple[str, object]) -> 'AbstractEnv':
        field, value = args
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if v is not self.vehicle:
                setattr(v, field, value)
        return env_copy

    def call_vehicle_method(self, args: Tuple[str, Tuple[object]]) -> 'AbstractEnv':
        method, method_args = args
        env_copy = copy.deepcopy(self)
        for i, v in enumerate(env_copy.road.vehicles):
            if hasattr(v, method):
                env_copy.road.vehicles[i] = getattr(v, method)(*method_args)
        return env_copy

    def randomize_behaviour(self) -> 'AbstractEnv':
        env_copy = copy.deepcopy(self)
        for v in env_copy.road.vehicles:
            if isinstance(v, IDMVehicle):
                v.randomize_behavior()
        return env_copy

    def to_finite_mdp(self):
        return finite_mdp(self, time_quantization=1/self.config["policy_frequency"])

    def __deepcopy__(self, memo):
        """Perform a deep copy but without copying the environment viewer."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ['viewer', 'automatic_rendering_callback']:
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, None)
        return result


class MultiAgentWrapper(Wrapper):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        reward = info["agents_rewards"]
        done = info["agents_dones"]
        return obs, reward, done, info