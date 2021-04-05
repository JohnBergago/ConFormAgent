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
# ray.init(local_mode=True, num_cpus=3, num_gpus=1)
ray.init(address='auto')
register_env("StorageEnv", RLLibConFormSimStorageEnv)
ModelCatalog.register_custom_model("SimpleRCNNModel", SimpleRCNNModel)

config={
    "env": "StorageEnv",
    "env_config": env_config,
    "num_gpus" : 0.5,
    
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

    # System params.
    #
    # == Overview of data flow in IMPALA ==
    # 1. Policy evaluation in parallel across `num_workers` actors produces
    #    batches of size `rollout_fragment_length * num_envs_per_worker`.
    # 2. If enabled, the replay buffer stores and produces batches of size
    #    `rollout_fragment_length * num_envs_per_worker`.
    # 3. If enabled, the minibatch ring buffer stores and replays batches of
    #    size `train_batch_size` up to `num_sgd_iter` times per batch.
    # 4. The learner thread executes data parallel SGD across `num_gpus` GPUs
    #    on batches of size `train_batch_size`.
    #
    "rollout_fragment_length": 200,
    "train_batch_size": 4000,
    "min_iter_time_s": 10,
    "num_workers": 2,
    # number of GPUs the learner should use.
    "num_gpus": 0.5,
    # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # proportionally with the number of buffers.
    "num_data_loader_buffers": 1,
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 20,
    # number of passes to make over each train batch
    "num_sgd_iter": 3,
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    "replay_proportion": 0.2,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    "replay_buffer_num_slots": 20,
    # max queue size for train batches feeding into the learner
    "learner_queue_size": 16,
    # wait for train batches to be available in minibatch buffer queue
    # this many seconds. This may need to be increased e.g. when training
    # with a slow environment
    "learner_queue_timeout": 300,
    # level of queuing for sampling.
    "max_sample_requests_in_flight_per_worker": 2,
    # max number of workers to broadcast one set of weights to
    "broadcast_interval": 1,
    # use intermediate actors for multi-level aggregation. This can make sense
    # if ingesting >2GB/s of samples, or if the data requires decompression.
    "num_aggregation_workers": 0,

    # Learning params.
    "grad_clip": 40.0,
    # either "adam" or "rmsprop"
    "opt_type": "adam",
    "lr": 3e-4,
    "lr_schedule": None,
    
    # rmsprop considered
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # balancing the three losses
    "vf_loss_coeff": 0.8,
    "entropy_coeff": 0.01,
    "entropy_coeff_schedule": None,

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
    "IMPALA",
    name="pbt_impala_vector_obs",
    scheduler=scheduler,
    metric="episode_reward_mean",
    mode="max",
    reuse_actors=False,
    checkpoint_freq=60,
    checkpoint_at_end=True,
    config=config,
    num_samples=3,
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




