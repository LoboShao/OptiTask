from typing import List, Dict, Tuple

from simulation.cluster import Cluster
from simulation.job import Job, JobStatus
from simulation.jobqueue import JobQueue


class ClusterEnvironment:
    def __init__(self, cluster: Cluster, job_queue: JobQueue):
        self.cluster = cluster
        self.job_queue = job_queue
        self.current_time = 0
        self.running_jobs: Dict[str, Job] = {}
