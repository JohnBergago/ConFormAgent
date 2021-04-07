from conform_agent.env.rllib.storage_env import RLLibConFormSimStorageEnv
from conform_agent.models.tf.simple_rcnn import SimpleRCNNModel
import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.impala import ImpalaTrainer

from typing import Dict, Optional, TYPE_CHECKING
import random

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import AgentID, PolicyID
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import PopulationBasedTraining


env_config = {
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

class ConFormCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, 
                       worker: "RolloutWorker", 
                       base_env: BaseEnv, 
                       policies: Dict[PolicyID, Policy], 
                       episode: MultiAgentEpisode, 
                       env_index: Optional[int], **kwargs) -> None:
        episode.custom_metrics["fraction_solved"] =[]
        for agent, info in episode._agent_to_last_info.items():
            if 'interrupted' in info:
                episode.custom_metrics["fraction_solved"].append(int(not info['interrupted']))
        

# ray initialization and stuff
ray.init(local_mode=True, num_cpus=4, num_gpus=1)
# ray.init(address='auto')
register_env("StorageEnv", RLLibConFormSimStorageEnv)
ModelCatalog.register_custom_model("SimpleRCNNModel", SimpleRCNNModel)

config={
    "env": "StorageEnv",
    "env_config": env_config,
    "num_gpus" : 1.0,
    "num_workers": 3, 
    
    "model":{
        "custom_model": "SimpleRCNNModel",
        "custom_model_config": {
            # Defines the convolutiontional layers. For each layer there has
            # to be [num_filters, kernel, stride]. 
            "conv_layers": [],
            # Defines the dense layers following the convolutional layers (if
            # any). For each layer the num_hidden units has to be defined. 
            "dense_layers": [128]*5, 
            # whether to use a LSTM layer after the dense layers.
            "use_recurrent": False,
        },
    },

    # Should use a critic as a baseline (otherwise don't use value baseline;
    # required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # The GAE (lambda) parameter.
    "lambda": 0.95,
    # Initial coefficient for KL divergence.
    "kl_coeff": 0.2,
    # Size of batches collected from each worker.
    "rollout_fragment_length": 64,
    # Number of timesteps collected for each SGD round. This defines the size
    # of each SGD epoch.
    "train_batch_size": 2048,
    # Total SGD batch size across all devices for SGD. This defines the
    # minibatch size within each epoch.
    "sgd_minibatch_size": 64,
    # Whether to shuffle sequences in the batch when training (recommended).
    "shuffle_sequences": True,
    # Number of SGD iterations in each outer loop (i.e., number of epochs to
    # execute per train batch).
    "num_sgd_iter": 3,
    # Stepsize of SGD.
    "lr": 3e-4,
    # Learning rate schedule.
    "lr_schedule": [[0, 0.0003], [64000000, 0]],
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set vf_share_layers=True inside your model's config.
    "vf_loss_coeff": 1.0,

    # Coefficient of the entropy regularizer.
    "entropy_coeff": 1.5e-2,
    # Decay schedule for the entropy regularizer.
    "entropy_coeff_schedule": None,
    # PPO clip parameter.
    "clip_param": 0.2,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount.
    "grad_clip": None,
    # Target value for KL divergence.
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes".
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation.
    "observation_filter": "NoFilter",

    "callbacks": ConFormCallbacks,
}

scheduler = PopulationBasedTraining(
    time_attr="training_iteration",
    perturbation_interval=60,
    hyperparam_mutations={
        "lr": lambda: random.uniform(1e-4, 2e-2),
        "vf_loss_coeff": lambda: random.uniform(0.5, 1.0),
        "entropy_coeff": lambda: random.uniform(5e-3, 2e-2),
    }
)

result = tune.run(
    "PPO",
    name="storage_ppo_vector_obs",
    # scheduler=scheduler,
    # metric="episode_reward_mean",
    # mode="max",
    stop={
        "timesteps_total": 64000000,
    },
    reuse_actors=False,
    checkpoint_freq=60,
    checkpoint_at_end=True,
    config=config,
    num_samples=1,
    # resume = True,
)
print("Best hyperparameters found were: ", result.best_config)
# result = tune.run(
#     "IMPALA",
#     stop={
#         # "timesteps_total": 10000000,
#     },
#     checkpoint_at_end=True,
#     checkpoint_freq=50,
#     config = config, 
#     resume=True,
#     )




