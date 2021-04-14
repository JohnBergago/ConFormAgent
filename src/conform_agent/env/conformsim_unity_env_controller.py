
import os
import copy
from typing import Dict, List
from mlagents_envs.side_channel import SideChannel
from ray.tune.utils import deep_update
import conform_agent

from conform_agent.env.unity.util import start_unity_env

class ConFormSimUnityEnvController:

    _BASE_PORT = 5004
    
    def __init__(self, config:Dict, default_env_config:Dict, side_channels:List[SideChannel]=[]):

        self.DEFAULT_ENV_CONFIG = default_env_config    
        
        # update the configuration for the first time and check back with
        # DEFAULT_ENV_CONFIG
        self.config = dict()
        self.update_config(config)
        
        # try to init the UnityEnvironment
        if not self.config.get('env_name') or self.config.get("env_name") == None:
            env_dir = None
        else:
            env_dir = conform_agent.ROOT_DIR + "/unity_executables/" + self.config.get('env_name')
            # env_dir = (proj_dir + "/opt/ConFormSim/build/" + self.config.get('env_name'))
        
        # check if there are any infos about no_graphic mode in the environment
        # configuration
        if self.config.get("no_graphics"):
            no_graphics = self.config.get("no_graphics")
        else:
            no_graphics = False

        # start unity environment
        self.env = start_unity_env(
            file_name=env_dir,
            worker_id=0,
            base_port=self._BASE_PORT,
            no_graphics=no_graphics,
            side_channels=side_channels,
        )        

    def apply_config():
        """Apply the configuration to the environment. This method is called 
        from the update_config() method. 
        
        At this point the configuration 
        parameters should be written to the corresponding side channels. 
        The new settings are then applied with the next step or reset of the 
        environment. 

        Examples::
            >>> self.env_param_channel.set_float_parameter(
                    "numTrainAreas", self.config.get("num_train_areas")
                )
            >>> # Read engine config 
            >>> engine_config = self.config.get("engine_config") 
            >>> # Configure the Engine
            >>> engine_config = EngineConfig(
                    width=engine_config.get("window_width"),
                    height=engine_config.get("window_height"),
                    quality_level=engine_config.get("quality_level"),
                    time_scale=engine_config.get("sim_speed"),
                    target_frame_rate=engine_config.get("target_frame_rate"),
                    capture_frame_rate=60,
                )
            >>> self.engine_channel.set_configuration(engine_config)
        """
        raise NotImplementedError

    def update_config(self, config:Dict):
        """Update the environment configuration with new values and apply them.
        This can be used for curriculum learning. 

        Args:
            config (Dict): The new configuration for the environment.
        """
        # update existing config with new keys
        self.config.update(config)
        # check if everything is valid
        self.config = deep_update(
            copy.deepcopy(self.DEFAULT_ENV_CONFIG),
            self.config, 
            new_keys_allowed=False, 
            allow_new_subkey_list = [])
        self.apply_config()