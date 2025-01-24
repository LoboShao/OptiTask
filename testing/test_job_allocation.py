from typing import List, Dict, Any

import numpy as np
import yaml
from simulation.machine import Machine
from simulation.rack import Rack
from simulation.gpu import GPU, GPU_MODELS
from simulation.cluster import Cluster
from simulation.user import User, UserType, UserTier
from simulation.job import Job
from simulation.cluster_builder import create_cluster_from_config
from simulation.cluster import QueueType


def test_single_machine_allocation():
    cluster = create_cluster_from_config('testing_configs/large_config1.yaml')

    premium_user = User(cluster, UserType.LLM, UserTier.PREMIUM)
    standard_user = User(cluster, UserType.LLM, UserTier.STANDARD)
    basic_user = User(cluster, UserType.LLM, UserTier.BASIC)

    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()

    standard_user.submit_job()
    next_job = cluster.queues[QueueType.HIGH_PRIORITY].get_next_job()
    selected_gpus = cluster.allocate_job(next_job)
    assert next_job.required_gpus == len(selected_gpus)

if __name__ == '__main__':
    test_single_machine_allocation()