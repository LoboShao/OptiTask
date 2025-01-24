from typing import List, Dict, Any
import yaml
from simulation.machine import Machine
from simulation.rack import Rack
from simulation.gpu import GPU, GPU_MODELS
from simulation.cluster import Cluster
from simulation.user import User, UserType, UserTier

from simulation.cluster_builder import create_cluster_from_config
from simulation.cluster import QueueType


def test_user_creation():
    cluster = create_cluster_from_config('testing_configs/large_config1.yaml')

    # Create a premium LLM user
    premium_user = User(cluster, UserType.LLM, UserTier.PREMIUM)

    # Create a basic CV user
    basic_user = User(cluster, UserType.CV, UserTier.BASIC)

    premium_user.submit_job()
    cluster.print_queue_status()


if __name__ == '__main__':
    test_user_creation()