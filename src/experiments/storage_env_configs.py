easy_vector_obs = {
    "env_name":"StorageEnvironmentGrid",
    # Whether to use visual observations or vector observation of the full env.
    "use_visual" : False,
    # Maximum number of steps until a single agent in the environment will be reset.
    "max_steps": 200,
    # Task difficulty to fulfill. Currently there are 3 levels:
    # 1 - As soon as an item is picked up the episode ends.
    # 2 - As soon as an item was brought to the correct target, the episode ends.
    # 3 - Only if all items are on their correct target, the episode ends.
    # If episode_horizon is reached, the episode ends.
    "task_level": 3,
    # Whether to use ray perception with 30 rays around the agent detecting all items
    # and base areas. Using this and visual observations might lead to strange
    # behaviour.
    "use_ray_perception" : False,
    # Whether to use a object property camera, that renders for each pixel of an
    # image the features of the object at that position on screen.
    "use_object_property_camera": False,

    # Whether to run the Unity simulator in no-graphics mode.
    "no_graphics": False,

    "num_train_areas": 8,
    #  More technical configurations of the simulation engine. More details in 
    # DEFAULT_ENGINE_CONFIG
    "engine_config": {
        # Factor which is applied to the simulation speed from 1 to 100. Faster will 
        # speed up training, but might break physics
        "sim_speed": 100,
        # Width of the window which the simulator creates.
        "window_width": 640,
        # Height of the window which the simulator creates.
        "window_height": 360,
    },
}
easy_visual_obs = {
    "env_name":"StorageEnvironmentGrid",
    # Whether to use visual observations or vector observation of the full env.
    "use_visual" : True,
    # Maximum number of steps until a single agent in the environment will be reset.
    "max_steps": 200,
    # Task difficulty to fulfill. Currently there are 3 levels:
    # 1 - As soon as an item is picked up the episode ends.
    # 2 - As soon as an item was brought to the correct target, the episode ends.
    # 3 - Only if all items are on their correct target, the episode ends.
    # If episode_horizon is reached, the episode ends.
    "task_level": 3,
    # Whether to use ray perception with 30 rays around the agent detecting all items
    # and base areas. Using this and visual observations might lead to strange
    # behaviour.
    "use_ray_perception" : False,
    # Whether to use a object property camera, that renders for each pixel of an
    # image the features of the object at that position on screen.
    "use_object_property_camera": False,

    # Whether to run the Unity simulator in no-graphics mode.
    "no_graphics": False,

    "num_train_areas": 8,

    #################################################################################
    # Observation Settings
    #################################################################################
    # The type of camera that should be used for visual observations:
    #   TopDownCamera: Overview over the complete area
    #   TopDownFollowCamera: A fixed size camera following the agent.
    #   EgoCamera: Ego perspective of the agent
    "camera_type": "TopDownFollowCamera",
    # size of the visual observation (images) in pixels [width, height]
    "vis_obs_size": [42, 42],

    #  More technical configurations of the simulation engine. More details in 
    # DEFAULT_ENGINE_CONFIG
    "engine_config": {
        # Factor which is applied to the simulation speed from 1 to 100. Faster will 
        # speed up training, but might break physics
        "sim_speed": 100,
        # Width of the window which the simulator creates.
        "window_width": 640,
        # Height of the window which the simulator creates.
        "window_height": 360,
    },
}