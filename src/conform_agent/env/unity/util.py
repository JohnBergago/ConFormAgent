import time
from typing import List, Optional
import random 

import mlagents_envs
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.environment import UnityEnvironment

def start_unity_env(
    file_name: Optional[str] = None,
    worker_id: int = 0,
    base_port: Optional[int] = None,
    seed: int = 0,
    no_graphics: bool = False,
    timeout_wait: int = 60,
    additional_args: Optional[List[str]] = None,
    side_channels: Optional[List[SideChannel]] = None,
    log_folder: Optional[str] = None,) -> UnityEnvironment:
    """Starts a new Unity environment and establishes connection with the environment.
    Notice: Basically it calls the ML-Agents function to create a new environment. 
    However, this function is extended by a mechanism to catch UnityWorkerInUseExceptions
    and spawn environments connecting via different ports. This comes in handy, when more
    than one environment are started/running concurrently. 

    Args:
        file_name (str, optional): Name of Unity environment binary. Defaults to None.
        worker_id (int, optional): Offset from base_port. Used for training multiple environments simultaneously. 
            Defaults to 0.
        base_port (int, optional): Baseline port number to connect to Unity environment over. 
            worker_id increments over this. If no environment is specified (i.e. file_name is None), 
            the DEFAULT_EDITOR_PORT will be used. Defaults to None.
        seed (int, optional): Seed to intitialize environement with. Defaults to 0.
        no_graphics (bool, optional): Whether to run the Unity simulator in no-graphics mode. Defaults to False.
        timeout_wait (int, optional): Time (in seconds) to wait for connection from environment. Defaults to 60.
        additional_args (List[str], optional): Addition Unity command line arguments. Defaults to None.
        side_channels (List[SideChannel], optional): Additional side channel for no-rl communication with Unity. 
            Defaults to None.
        log_folder (str, optional): Optional folder to write the Unity Player log file into. Requires absolute path. 
            Defaults to None.

    Returns:
        UnityEnvironment: Newly created Unity environment
    """
    unity_env = None
    # Try connecting to the Unity3D game instance. If a port is blocked
    while True:
        # Sleep for random time to allow for concurrent startup of many
        # environments (num_workers >> 1). Otherwise, would lead to port
        # conflicts sometimes.
        time.sleep(random.randint(1, 10))
        port_ = base_port
        base_port += 1
        try:
            unity_env = UnityEnvironment(
                file_name=file_name,
                worker_id=worker_id,
                base_port=port_,
                seed=seed,
                no_graphics=no_graphics,
                timeout_wait=timeout_wait,
                side_channels=side_channels,
                additional_args=additional_args,
                log_folder=log_folder
            )
            print("Created UnityEnvironment for port {}".format(port_))
        except mlagents_envs.exception.UnityWorkerInUseException:
            pass
        else:
            break
    return unity_env