import random
import numpy as np
from conform_agent.env.rllib.storage_env import RLLibConFormSimStorageEnv
from conform_agent.models.tf.simple_rcnn import SimpleRCNNModel
import ray
from ray import tune
from ray.tune.registry import register_env
from conform_agent.conform_callbacks import ConFormCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.schedulers import ASHAScheduler

import experiments.storage_env_configs as StorageEnvConfig

random.seed(42)
np.random.seed(42)

# ray initialization and stuff
# ray.init(local_mode=True, num_cpus=4, num_gpus=1)
ray.init(address='auto')

register_env("StorageEnv", RLLibConFormSimStorageEnv)
ModelCatalog.register_custom_model("SimpleRCNNModel", SimpleRCNNModel)

config={
    "env": "StorageEnv",
    "env_config": StorageEnvConfig.easy_vector_obs,
    
    "model":{
        "custom_model": "SimpleRCNNModel",
        "custom_model_config": {
            # Defines the convolutiontional layers. For each layer there has
            # to be [num_filters, kernel, stride]. 
            "conv_layers": [],
            # Defines the dense layers following the convolutional layers (if
            # any). For each layer the num_hidden units has to be defined. 
            "dense_layers": tune.grid_search([[128]*4,[64]*5, [48]*6]), 
            # whether to use a LSTM layer after the dense layers.
            "use_recurrent": False,
        },
    },

    # Whether to use V-trace weighted advantages. If false, PPO GAE
    # advantages will be used instead.
    "vtrace": True,

    # == These two options only apply if vtrace: False ==
    # Should use a critic as a baseline (otherwise don't use value
    # baseline; required for using GAE).
    "use_critic": True,
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 0.95,

    # == PPO surrogate loss options ==
    "clip_param": 0.2,

    # == PPO KL Loss options ==
    "use_kl_loss": True,
    "kl_coeff": tune.uniform(0.3, 1),
    "kl_target": tune.loguniform(3e-3, 3e-2),

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
    "rollout_fragment_length": 64,
    "train_batch_size": 2048,
    "min_iter_time_s": 10,
    "num_workers": 4,
    # number of GPUs the learner should use.
    "num_gpus": 0.5,
    # set >1 to load data into GPUs in parallel. Increases GPU memory usage
    # proportionally with the number of buffers.
    "num_data_loader_buffers": 1,
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 30,
    # number of passes to make over each train batch
    "num_sgd_iter": 30,
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    "replay_proportion": 0.0,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    "replay_buffer_num_slots": 0,
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
    "lr":  tune.loguniform(1e-5, 5e-3),
    "lr_schedule": None,
    
    # rmsprop considered
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.1,
    # balancing the three losses
    "vf_loss_coeff": tune.uniform(0.5, 1.0),
    "entropy_coeff": tune.uniform(0, 0.01),
    "entropy_coeff_schedule": None,

    # Discount factor of the MDP.
    "gamma": 0.90,

    "callbacks": ConFormCallbacks,
}

stopping_criteria = {
    # "training_iteration": 180,
    # # "time_total_s" : 1800,
}

scheduler = ASHAScheduler(
    metric="episode_reward_mean",
    mode="max",
    time_attr="training_iteration",
    grace_period=50,
    max_t=200, 
)

result = tune.run(
    "APPO",
    name="appo_vector_obs_param_search_4",
    scheduler=scheduler,
    stop=stopping_criteria,
    reuse_actors=False,
    checkpoint_freq=100,
    checkpoint_at_end=True,
    config=config,
    num_samples=20,
    max_failures=3,
    # resume = True,
)
print("Best hyperparameters found were: ", result.get_best_config(metric="episode_reward_mean", mode="max"))