from simulation.cluster import Cluster
from simulation.job import Job
from simulation.job_types import JOB_PROFILES
from simulation.gpu_model import GPU_MODELS
from enum import Enum
from typing import List, Dict, Tuple, Optional, Union
import numpy as np


class UserTier(Enum):
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"


# Define GPU access levels for each tier
TIER_GPU_ACCESS = {
    UserTier.BASIC: ["T4"],
    UserTier.STANDARD: ["T4", "A100_40GB"],
    UserTier.PREMIUM: ["T4", "A100_40GB", "A100_80GB", "H100_80GB"]
}

# Define maximum GPU counts for each tier
TIER_MAX_GPUS = {
    UserTier.BASIC: 2,
    UserTier.STANDARD: 4,
    UserTier.PREMIUM: 8
}


class UserType(Enum):
    LLM = "llm_user"
    CV = "cv_user"
    ML = "ml_user"
    MIXED = "mixed_user"


def generate_job_params(user_type: UserType, tier: UserTier) -> Dict:
    """Generate job parameters based on user type and tier"""
    available_gpus = TIER_GPU_ACCESS[tier]
    max_gpus = TIER_MAX_GPUS[tier]

    if user_type == UserType.LLM:
        job_type = "llm"
        batch_size = np.random.choice([8, 16, 32, 64, 256])
        # Filter high-memory GPUs for LLM workloads
        gpu_options = [gpu for gpu in available_gpus if "80GB" in gpu]
        if not gpu_options:  # Fallback to available GPUs if no high-memory ones
            gpu_options = available_gpus
        gpu_type = np.random.choice(gpu_options)
        required_gpus = min(np.random.choice([2, 8]), max_gpus)

    elif user_type == UserType.CV:
        job_type = np.random.choice(["classification", "segmentation"])
        batch_size = np.random.choice([32, 64, 128])
        gpu_type = np.random.choice(available_gpus)
        required_gpus = min(np.random.choice([1, 2, 4]), max_gpus)

    elif user_type == UserType.MIXED:
        job_type = np.random.choice(["llm", "classification", "segmentation"])
        batch_size = np.random.choice([8, 16, 32, 64])
        gpu_type = np.random.choice(available_gpus)
        required_gpus = min(np.random.choice([1, 2, 4, 8]), max_gpus)

    # Calculate resources using profiles
    flops, memory_gb = JOB_PROFILES[job_type](batch_size)
    return {
        "job_type": job_type,
        "batch_size": batch_size,
        "gpu_type": gpu_type,
        "required_gpus": required_gpus,
        "gpu_memory_per_gpu": int(memory_gb * 1024),  # Convert to MB
        "cpu_memory_total": int(memory_gb * 1.5 * 1024),  # 1.5x GPU memory in MB
        "total_episodes": np.random.randint(100, 1000),
        "max_runtime_hours": np.random.uniform(1, 24)
    }


class User:
    def __init__(self,
                 id: int,
                 cluster: Cluster,
                 user_type: Union[str, UserType],
                 tier: Union[str, UserTier] = UserTier.STANDARD):
        self.cluster = cluster

        # Convert string to enum if necessary
        if isinstance(user_type, str):
            user_type = UserType(user_type)
        if isinstance(tier, str):
            tier = UserTier(tier)

        self.id = id
        self.user_type = user_type
        self.tier = tier
        self.job_id_counter = 0

    def submit_job(self, priority_score: float = 0.0):
        """Submit a job with generated parameters based on user type and tier"""
        job_id = f"{self.id}-{self.job_id_counter}"
        params = generate_job_params(self.user_type, self.tier)

        job = Job(id=job_id, **params)
        self.job_id_counter += 1

        queue_type = self.cluster.select_queue(job)
        self.cluster.queues[queue_type].add_job(job, priority_score)

    @property
    def available_gpus(self) -> List[str]:
        """Get list of GPUs available to this user based on their tier"""
        return TIER_GPU_ACCESS[self.tier]

    @property
    def max_gpus(self) -> int:
        """Get maximum number of GPUs this user can request based on their tier"""
        return TIER_MAX_GPUS[self.tier]
