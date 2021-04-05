"""Default configurations for the ConFormSim storage environment."""

from typing import Dict
from conform_agent.utils import rgb_to_hex

"""Assign a float value to each camera type."""
CAMERA_TYPES: Dict[str, float] = {
    "TopDownCamera": 0.0,
    "TopDownFollowCamera": 1.0,
    "EgoCamera": 2.0,
}

"""Default engine configuration for the ConFormSim storage environment.
"""
DEFAULT_ENGINE_CONFIG = {
    # Factor which is applied to the simulation speed from 1 to 100. Faster will 
    # speed up training, but might break physics
    "sim_speed": 20,
    # Width of the window which the simulator creates.
    "window_width": 80,
    # Height of the window which the simulator creates.
    "window_height": 80,
    # Quality level of the simulation.
    "quality_level": 1,
    # Target framerate to achieve during training.
    "target_frame_rate": -1,
}

# Configuration of the environment
DEFAULT_ENV_CONFIG = {
    #################################################################################
    # General Settings (Unity Env)
    #################################################################################
    # The UnityEnvironment path or file to be wrapped in the gym.
    "env_name": None,
    # Whether to use visual observations or vector observation of the full env.
    "use_visual" : False,
    # Maximum number of steps until a single agent in the environment will be reset.
    "max_steps": 200,
    # Task difficulty to fulfill. Currently there are 3 levels:
    # 1 - As soon as an item is picked up the episode ends.
    # 2 - As soon as an item was brought to the correct target, the episode ends.
    # 3 - Only if all items are on their correct target, the episode ends.
    # If max_steps is reached, the episode ends.
    "task_level": 3,
    # Whether to use ray perception with 30 rays around the agent detecting all items
    # and base areas. Using this and visual observations might lead to strange
    # behaviour.
    "use_ray_perception" : False,
    # Whether to use a object property camera, that renders for each pixel of an
    # image the features of the object at that position on screen.
    "use_object_property_camera": False,
    # Return visual observations as uint8 (0-255) matrices instead of float (0.0-1.0).
    "uint8_visual": False,
    # If True, turn branched discrete action spaces into a Discrete space rather than
    # MultiDiscrete.
    "flatten_branched": False,
    # Whether to run the Unity simulator in no-graphics mode.
    "no_graphics": False,
    # # If True, return a list of visual observations instead of only one.
    # "allow_multiple_visual_obs": False,
    # Seed to use for instantiating an environment. (Not Implemented)
    "seed": 0,
    # Don't provide visual output of the observations for the user. But the user can
    # still switch through the statistics.
    "no_display": False,
    # More technical configurations of the simulation engine. More details in 
    # DEFAULT_ENGINE_CONFIG
    "engine_config": DEFAULT_ENGINE_CONFIG,

    #################################################################################
    # Observation Settings
    #################################################################################
    # The type of camera that should be used for visual observations:
    #   TopDownCamera: Overview over the complete area
    #   TopDownFollowCamera: A fixed size camera following the agent.
    #   EgoCamera: Ego perspective of the agent
    "camera_type": "TopDownCamera",
    # size of the visual observation (images) in pixels [width, height]
    "vis_obs_size": [84, 84],

    #################################################################################
    # Area Settings
    #################################################################################
    # Number of areas to instantiate per Worker. More may decrease training speed.
    "num_train_areas": 8,
    # [Min, Max] size of the environment grid in x direction
    "grid_size_x": [6, 6],
    # [Min, Max] size of the environment grid in y direction
    "grid_size_y": [6, 6],
    # Boxes will vanish as soon as they are in the correct area. If
    # boxes_need_drop is true, they will vanish as soon as they are dropped.
    "boxes_vanish": False,
    # Boxes have to be dropped in order to receive reward. Otherwise reward will
    # be given as a soon as a box is within the target area
    "boxes_need_drop": True,
    # Whether only a very sparse reward and some step penalty is applied or a more
    # distributed reward.
    "sparse_reward_only": False,
    # Defines the number of base types that should be used. The environment will 
    # always use the first num_base_types prefabs in the list.
    "num_base_types": 2,
    # Defines how many elements [MIN, MAX] per base type should be in the complete 
    # area. A 2x2 base consists of 4 base elements.
    "num_per_base_type": [4, 4],
    # [Min, Max] size of the bases in one area in x direction.
    "base_size_x": [2,2],
    # [Min, Max] size of the bases in one area in y direction.
    "base_size_y": [2,2],
    # Whether bases should only be deployed in corners. take care to set above
    # accordingly. Otherwise bases might overlap.
    "base_in_corners_only": True,

    # Color Settings
    # Whether to not fill the bases with color and instead just use the corner
    # markers/surrounding lines.
    "no_base_fill_color": False,
    # Instead of using the color in the color pool, light up the color to make it
    # better distinguishable from the items. 
    "brighter_bases": False,
    # Instead of highlighting the corners only, surround the bases with a full line.
    "full_base_line": False,
    # List of hex colors that will be used for bases and items. Bases and items of
    # the same color belong to each other. The format in hex is 0xRRGGBBAA
    "color_pool":[
        rgb_to_hex(255, 0, 0),
        0x01DD16FF,
        ],

    #################################################################################
    # Item Settings
    #################################################################################
    # Defines the number of item types that should be used. The environment will 
    # choose randomly from all available colors in the color pool for each episode.
    "num_item_types": 2,
    # Defines how many elements [MIN, MAX] per item type should be in the complete 
    # area.
    "num_per_item": [1, 1],
}