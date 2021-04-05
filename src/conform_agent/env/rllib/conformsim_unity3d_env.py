from logging import info
from typing import List, Dict, Optional, Tuple
import numpy as np
import gym
from gym import spaces, Space
from ray.rllib.env import ExternalEnv
from ray.rllib.utils.annotations import override
from conform_agent.env.conformsim_unity_env_controller import ConFormSimUnityEnvController

from mlagents_envs.base_env import ActionTuple

class ConFormSimUnity3DEnv(ExternalEnv):
    """Wrap ConFormSim Unity 3D environments as RLlib VectorEnv.

    The ConFormSim Unity environments are allowed to include more than one
    agent. However, all the agents are of the same behavior but performing their
    own training episodes. The RLlib Unity3DEnv does not handle this properly,
    as it is based on RLlib's MultiAgentEnv. These envs assume that all agents
    are part of the same episode. This is not feasible for the ConFormSim
    environments. Our Unity envs host multiple single agent environments. Thus
    they are more like VectorEnvs.

    """
    def __init__(self, env_controller:ConFormSimUnityEnvController):

        # get the environment controller and retrieve unity environment
        self.env_controller = env_controller
        self.unity_env = self.env_controller.env
        
        # retrieve information about the environment
        self.unity_env.reset()

        # Check brain configuration
        if len(self.unity_env.behavior_specs) != 1:
            raise Exception(
                "There can only be one behavior in a UnityEnvironment "
                "if it is wrapped in a ConFormSim environment."
            )

        self._behavior_name = list(self.unity_env.behavior_specs.keys())[0]
        decision_steps, _ = self.unity_env.get_steps(self._behavior_name)
        # get number of agents
        num_envs = len(decision_steps)
        print("Connected to {} with {} agents.".format(
                                                self._behavior_name, 
                                                num_envs
                                                ))
        # get observation space and action space
        observation_space = self._get_observation_space()
        action_space = self._get_action_space()

         # ML-Agents API version.
        self.api_version = self.unity_env.API_VERSION.split(".")
        self.api_version = [int(s) for s in self.api_version]

        # agent id to episode id mapping
        self.agentID_to_episodeID = {}

        super().__init__(action_space, observation_space, num_envs*2)

    def _get_observation_space(self):
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
            obs_space = spaces.Tuple(list_spaces)
        else:
            obs_space = list_spaces[0]
        return obs_space

    def _get_action_space(self):
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
                "The ConFormSimUnity3DEnv does not provide explicit support for"
                " both discrete and continuous actions."
            )

    @override(ExternalEnv)
    def run(self):
        # reset unity environment before start
        self.unity_env.reset()
        
        while True:
            decision_steps, terminal_steps = self.unity_env.get_steps(
                self._behavior_name)
            # first process all envs/"agents" in decision steps
            actions = []
            # process all envs/"agents" that finished their episodes in terminal
            # steps
            for agent_id in terminal_steps.agent_id:
                # first check if a new episode needs to be started
                if agent_id not in self.agentID_to_episodeID.keys():
                    episode_id = self.start_episode() 
                    self.agentID_to_episodeID[agent_id] = episode_id
                episode_id = self.agentID_to_episodeID[agent_id]
                # get observation, rewards and info
                obs = terminal_steps[agent_id].obs
                obs = obs[0] if len(obs) == 1 else obs
                reward = terminal_steps[agent_id].reward
                info = {"interrupted": terminal_steps[agent_id].interrupted}
                self.log_returns(episode_id, reward, info)
                # end episode and remove agent_id from self.agentID_to_episodeID
                self.end_episode(episode_id, obs)
                self.agentID_to_episodeID.pop(agent_id)
                
            for agent_id  in decision_steps.agent_id:
                # first check if a new episode needs to be started
                if agent_id not in self.agentID_to_episodeID.keys():
                    episode_id = self.start_episode() 
                    self.agentID_to_episodeID[agent_id] = episode_id
                episode_id = self.agentID_to_episodeID[agent_id]
                # get observation and reward and request action
                obs = decision_steps[agent_id].obs
                obs = obs[0] if len(obs) == 1 else obs
                reward = decision_steps[agent_id].reward
                # log reward and request action 
                self.log_returns(episode_id, reward)
                actions.append(self.get_action(episode_id, obs))
            # set actions in Unity environment
            if actions:
                if actions[0].dtype == np.float32:
                    action_tuple = ActionTuple(continuous=np.array(actions))
                else:
                    action_tuple = ActionTuple(discrete=np.array(actions))
            self.unity_env.set_actions(self._behavior_name, action_tuple)

            self.unity_env.step()

    def update_config(self, config:Dict):
        """Update the environment configuration with new values and apply them.
        This can be used for curriculum learning. 

        Args:
            config (Dict): The new configuration for the environment.
        """
        self.env_controller.update_config(config)
        # to apply config reset environment and end all running episodes
        self.unity_env.reset()
        decision_steps = self.unity_env.get_steps(self._behavior_name)
        for agent_id in decision_steps.agent_id:
            episode_id = self.agentID_to_episodeID[agent_id]
            self.end_episode(episode_id, decision_steps[agent_id].obs)

    def close(self):
        self.unity_env.close()




                
                
