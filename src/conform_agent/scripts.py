import argparse 

from ray.rllib.scripts import cli as ray_cli
from ray.rllib import train
from ray.rllib import rollout

from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from conform_agent.models.tf.simple_rcnn import SimpleRCNNModel
from conform_agent.conform_callbacks import ConFormCallbacks
from conform_agent.env.rllib.storage_env import RLLibConFormSimStorageEnv

# register all ConFormAgent specific libraries
register_env("StorageEnv", RLLibConFormSimStorageEnv)
ModelCatalog.register_custom_model("SimpleRCNNModel", SimpleRCNNModel)

EXAMPLE_USAGE = """
Example usage for training:
    conform-rllib train --run DQN --env StorageEnv
Example usage for rollout:
    conform-rllib rollout /trial_dir/checkpoint_000001/checkpoint-1 --run DQN
"""

def cli():
    parser = argparse.ArgumentParser(
        description="Train or Run an RLlib Trainer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE)
    subcommand_group = parser.add_subparsers(
        help="Commands to train or run an RLlib agent.", dest="command")

    # see _SubParsersAction.add_parser in
    # https://github.com/python/cpython/blob/master/Lib/argparse.py
    train_parser = train.create_parser(
        lambda **kwargs: subcommand_group.add_parser("train", **kwargs))
    rollout_parser = rollout.create_parser(
        lambda **kwargs: subcommand_group.add_parser("rollout", **kwargs))
    options = parser.parse_args()

    if options.command == "train":
        train.run(options, train_parser)
    elif options.command == "rollout":
        rollout.run(options, rollout_parser)
    else:
        parser.print_help()
