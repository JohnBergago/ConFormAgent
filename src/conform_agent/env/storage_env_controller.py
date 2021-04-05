
import os
import copy
from conform_agent.env.config.storage_config import CAMERA_TYPES, DEFAULT_ENV_CONFIG
from conform_agent.env.unity.int_list_property_channel import IntListPropertiesChannel
from conform_agent.env.conformsim_unity_env_controller import ConFormSimUnityEnvController
import mlagents_envs
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel, EngineConfig

from ray.tune.utils import deep_update

from conform_agent.env.unity.util import start_unity_env

class StorageEnvController(ConFormSimUnityEnvController):

    _BASE_PORT = 5004
    
    def __init__(self, config=DEFAULT_ENV_CONFIG):
        """
        Environment initialization
        :param config: Configuration of the environment.
        """
            
        # create side channels
        self.env_param_channel = EnvironmentParametersChannel()
        self.engine_channel = EngineConfigurationChannel()
        self.color_pool_channel = IntListPropertiesChannel()
        
        side_channels = [
            self.env_param_channel,
            self.engine_channel,
            self.color_pool_channel,]
        
        # flag whether the config has been apllied to the environment
        self.is_already_initialized = False
        # create environment with config and side channels
        super().__init__(config, DEFAULT_ENV_CONFIG, side_channels=side_channels)


    def apply_config(self):
        # set FloatProperties
        grid_size_x = self.config.get("grid_size_x")
        if not isinstance(grid_size_x, list) or len(grid_size_x) != 2:
            raise ("The provided grid_size_x parameter is no list of type "
                    "[min, max]. Please correct this.")
        grid_size_y = self.config.get("grid_size_y")
        if not isinstance(grid_size_y, list) or len(grid_size_y) != 2:
            raise ("The provided grid_size_y parameter is no list of type "
                    "[min, max]. Please correct this.")

        vis_obs_size = self.config.get("vis_obs_size")
        if not isinstance(vis_obs_size, list) or len(vis_obs_size) != 2:
            raise ("The provided vis_obs_size parameter is no list of type "
                    "[min, max]. Please correct this.")

        base_size_x = self.config.get("base_size_x")
        if not isinstance(base_size_x, list) or len(base_size_x) != 2:
            raise ("The provided base_size_x parameter is no list of type "
                    "[min, max]. Please correct this.")
        base_size_y = self.config.get("base_size_x")
        if not isinstance(base_size_x, list) or len(base_size_x) != 2:
            raise ("The provided base_size_x parameter is no list of type "
                    "[min, max]. Please correct this.")
        num_per_base_type = self.config.get("num_per_base_type")
        if not isinstance(num_per_base_type, list) or len(num_per_base_type) != 2:
            raise ("The provided num_per_base_type parameter is no list of type "
                    "[min, max]. Please correct this.")      
        
        num_per_item = self.config.get("num_per_item")
        if not isinstance(num_per_item, list) or len(num_per_item) != 2:
            raise ("The provided num_per_item parameter is no list of type "
                    "[min, max]. Please correct this.")

        color_pool = self.config.get("color_pool")
        if not isinstance(color_pool, list):
            raise ("The provided color_pool parameter is not of type list. "
                    "Please correct this.")  

        camera_type = self.config.get("camera_type")
        camera_type_f: float = CAMERA_TYPES[camera_type] or 0.0

        # set properties in reset channel
        self.env_param_channel.set_float_parameter("minGridSizeX", grid_size_x[0])
        self.env_param_channel.set_float_parameter("maxGridSizeX", grid_size_x[1])
        self.env_param_channel.set_float_parameter("minGridSizeY", grid_size_y[0])
        self.env_param_channel.set_float_parameter("maxGridSizeY", grid_size_y[1])
        self.env_param_channel.set_float_parameter("cameraType", camera_type_f)
        # area settings
        # check if num train areas should be set
        if self.is_already_initialized:
            print("You're trying to change the number of "
                "train areas, during runtime. This is only possible at " 
                "initialization.")
        else:
            self.env_param_channel.set_float_parameter("numTrainAreas", self.config.get("num_train_areas"))
            
        self.env_param_channel.set_float_parameter("numBaseTypesToUse", self.config.get("num_base_types"))
        self.env_param_channel.set_float_parameter("numberPerBaseTypeMax", num_per_base_type[1])
        self.env_param_channel.set_float_parameter("numberPerBaseTypeMin", num_per_base_type[0])
        self.env_param_channel.set_float_parameter("baseSizeXMax", base_size_x[1])
        self.env_param_channel.set_float_parameter("baseSizeXMin", base_size_x[0])
        self.env_param_channel.set_float_parameter("baseSizeZMax", base_size_y[1])
        self.env_param_channel.set_float_parameter("baseSizeZMin", base_size_y[0])
        self.env_param_channel.set_float_parameter("baseInCornersOnly", 1 if self.config.get("base_in_corners_only") else 0)
        self.env_param_channel.set_float_parameter("boxesVanish", 1 if self.config.get("boxes_vanish") else 0)
        self.env_param_channel.set_float_parameter("boxesNeedDrop", 1 if self.config.get("boxes_need_drop") else 0)
        self.env_param_channel.set_float_parameter("sparseReward", 1 if  self.config.get("sparse_reward_only")  else 0)
        # color settings
        self.env_param_channel.set_float_parameter("noBaseFillColor", 1 if self.config.get("no_base_fill_color") else 0)
        self.env_param_channel.set_float_parameter("brighterBases", 1 if self.config.get("brighter_bases") else 0)
        self.env_param_channel.set_float_parameter("full_base_line", 1 if self.config.get("fullBaseLine") else 0)
        # item settings
        self.env_param_channel.set_float_parameter("numItemTypesToUse", self.config.get("num_item_types"))
        self.env_param_channel.set_float_parameter("numberPerItemTypeMax", num_per_item[1])
        self.env_param_channel.set_float_parameter("numberPerItemTypeMin", num_per_item[0])
        # general settings
        self.env_param_channel.set_float_parameter("noDisplay", 1 if self.config.get("no_display") else 0)
        self.env_param_channel.set_float_parameter("visObsWidth", vis_obs_size[0])
        self.env_param_channel.set_float_parameter("visObsHeight", vis_obs_size[1])
        self.env_param_channel.set_float_parameter("useVisual", 1 if self.config.get("use_visual") and not self.config.get("use_object_property_camera") else 0)
        self.env_param_channel.set_float_parameter("useRayPerception", 1 if self.config.get("use_ray_perception") else 0)
        self.env_param_channel.set_float_parameter("useObjectPropertyCamera", 1 if self.config.get("use_object_property_camera") else 0)
        self.env_param_channel.set_float_parameter("maxSteps", self.config.get("max_steps"))
        self.env_param_channel.set_float_parameter("taskLevel", self.config.get("task_level"))

        # Read engine config 
        engine_config = self.config.get("engine_config") 
        # Configure the Engine
        engine_config = EngineConfig(
            width=engine_config.get("window_width"),
            height=engine_config.get("window_height"),
            quality_level=engine_config.get("quality_level"),
            time_scale=engine_config.get("sim_speed"),
            target_frame_rate=engine_config.get("target_frame_rate"),
            capture_frame_rate=60)
        self.engine_channel.set_configuration(engine_config)

        # set list properties
        self.color_pool_channel.set_property(
            "colorPool", 
            self.config.get("color_pool"))
        self.is_already_initialized = True
