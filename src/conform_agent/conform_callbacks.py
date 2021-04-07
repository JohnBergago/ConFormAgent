from typing import Dict, Optional, TYPE_CHECKING
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import AgentID, PolicyID

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
        