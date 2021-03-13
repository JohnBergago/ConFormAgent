from typing import Dict, List, Tuple, Callable
import logging

from gym import spaces, Space
import numpy as np

from conform_agent.env import storage_env_controller
import conform_agent.env.config.storage_config as storage_config
from conform_agent.env.unity.util import start_unity_env
from conform_agent.env.storage_env_controller import StorageEnvController

from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.typing import MultiAgentDict, PolicyID, AgentID

logger = logging.getLogger(__name__)

class RLLibStorageEnv(Unity3DEnv):

    def __init__(self, config=storage_config.DEFAULT_ENV_CONFIG):
        
        # init MultiAgentEnv
        super(Unity3DEnv, self).__init__()
        self.config = config
        self.env_controller = StorageEnvController(config)
        self.unity_env = self.env_controller.env
        self.unity_env.reset()

        self._behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        decision_steps, _ = self.unity_env.get_steps(self._behavior_name)
        self._num_agents = len(decision_steps)
        print("Connected to StorageEnvironment with", self._num_agents, "agents.")

         # ML-Agents API version.
        self.api_version = self.unity_env.API_VERSION.split(".")
        self.api_version = [int(s) for s in self.api_version]
        
        # Reset entire env every this number of step calls.
        self.episode_horizon = self.config.get("episode_horizon")
        # Keep track of how many times we have called `step` so far.
        self.episode_timesteps = 0
        

    @property
    def action_space(self):
        group_spec = self.unity_env.behavior_specs[self._behavior_name]

        # get action space for discrete actions
        if group_spec.action_spec.is_discrete():
            branches = group_spec.action_spec.discrete_branches
            return spaces.MultiDiscrete(branches)
        # get action space if there ar continuous actions
        elif group_spec.action_spec.is_continuous():
            high = np.array([1] * group_spec.action_spec.continuous_size)
            return spaces.Box(-high, high, dtype=np.float32)
        else:
            raise Exception(
                "The RLLibStorageEnv does not provide explicit support for "
                "both discrete and continuous actions."
            )

    @property
    def observation_space(self):
        group_spec = self.unity_env.behavior_specs[self._behavior_name]

        list_spaces: List[Space] = []
        multi_dim_shapes = []
        vector_obs_size = 0
        for obs_spec in group_spec.observation_specs:
            if len(obs_spec.shape) > 1:
                multi_dim_shapes.append(obs_spec.shape)
            elif len(obs_spec.shape) == 1:
                vector_obs_size += obs_spec.shape[0]
        for shape in multi_dim_shapes:
            list_spaces.append(spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=shape))
        if vector_obs_size > 0:
            high = np.array([np.inf] * vector_obs_size)
            list_spaces.append(spaces.Box(-high, high, dtype=np.float32))
        if (len(list_spaces)>1):
            obs_space_single = spaces.Tuple(list_spaces)
        else:
            obs_space_single = list_spaces[0]

        if (self._num_agents > 1):
            return spaces.Tuple([obs_space_single] * self._num_agents)
        return obs_space_single

    @property
    def behavior_name(self):
        name = self._behavior_name.split("?")[0]
        return name
    
    def get_policy_configs_for_game(
            game_name: str) -> Tuple[dict, Callable[[AgentID], PolicyID]]:
        raise AttributeError("'RLLibStorageEnv' has no attribute 'get_policy_configs_for_game'")

    def close(self):
        self.unity_env.close()

