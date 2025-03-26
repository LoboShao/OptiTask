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

    premium_user = User(0, cluster, UserType.LLM, UserTier.PREMIUM)

    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()

    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()

    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()
    premium_user.submit_job()

    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()

    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()

    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.schedule_next_job()
    cluster.print_cluster_status()


if __name__ == '__main__':
    test_single_machine_allocation()