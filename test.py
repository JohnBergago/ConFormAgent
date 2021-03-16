from conform_agent.env.rllib.storage_env import RLLibConFormSimStorageEnv
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.impala import ImpalaTrainer


env_config = {
    "env_name":"StorageEnvironmentGrid",
    # Whether to use visual observations or vector observation of the full env.
    "use_visual" : False,
    # Maximum number of steps until a single agent in the environment will be reset.
    "max_steps": 150,
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

def on_episode_end(info):
    episode = info["episode"]
    episode.custom_metrics["fraction_solved"] =[]
    for agent, info in episode._agent_to_last_info.items():
        if 'solved' in info:
            episode.custom_metrics["fraction_solved"].append(int(info['solved']))

# ray initialization and stuff
# ray.init(local_mode=True, num_cpus=3, num_gpus=1)
ray.init(address='auto')
register_env("StorageEnv", RLLibConFormSimStorageEnv)

config={
    "env": "StorageEnv",
    "env_config": env_config,
    "num_gpus" : 1,
    # "num_sgd_iter": 10,
    "num_workers": 4,
    # "callbacks": {
    #     "on_episode_end": on_episode_end,
    # }
}


result = tune.run(
    "IMPALA",
    stop={
        "timesteps_total": 16000000,
    },
    config = config,
    max_failures=1 
    )




