from simulation.cluster import Cluster
from simulation.job import Job

from simulation.job_generators import generate_llm_job_params, generate_classification_job_params, generate_segmentation_job_params

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


def generate_job_params(user_type, tier):
    """Generate job parameters based on user type and tier"""
    available_gpus = TIER_GPU_ACCESS[tier]
    max_gpus = TIER_MAX_GPUS[tier]

    # Determine job type based on user type
    if user_type == UserType.LLM:
        job_type = "llm"
        # Filter high-memory GPUs for LLM workloads
        gpu_options = [gpu for gpu in available_gpus if "80GB" in gpu or "H100" in gpu]
        if not gpu_options:  # Fallback to available GPUs if no high-memory ones
            gpu_options = [gpu for gpu in available_gpus if "40GB" in gpu or "V100" in gpu]
            if not gpu_options:  # Further fallback
                gpu_options = available_gpus
    elif user_type == UserType.CV:
        job_type = np.random.choice(["classification", "segmentation"], p=[0.6, 0.4])
        gpu_options = available_gpus
    elif user_type == UserType.MIXED:
        job_type = np.random.choice(["llm", "classification", "segmentation"], p=[0.4, 0.35, 0.25])
        if job_type == "llm":
            gpu_options = [gpu for gpu in available_gpus if "80GB" in gpu or "H100" in gpu]
            if not gpu_options:
                gpu_options = [gpu for gpu in available_gpus if "40GB" in gpu or "V100" in gpu]
                if not gpu_options:
                    gpu_options = available_gpus
        else:
            gpu_options = available_gpus
    else:
        raise ValueError(f"Unknown user type: {user_type}")

    # Select GPU type
    gpu_type = np.random.choice(gpu_options)

    # Generate job parameters based on job type
    if job_type == "llm":
        params = generate_llm_job_params()
        # Set required GPUs based on model size
        if params["hidden"] >= 2048 or params["dataset_size"] >= 1_000_000:
            required_gpus = min(max(2, np.random.choice([2, 4, 8])), max_gpus)
        else:
            required_gpus = min(max(1, np.random.choice([1, 2, 4])), max_gpus)

    elif job_type == "classification":
        params = generate_classification_job_params()
        # Set required GPUs based on dataset and batch size
        if params["dataset_size"] >= 100_000 or params["batch_size"] >= 256:
            required_gpus = min(max(1, np.random.choice([1, 2, 4])), max_gpus)
        else:
            required_gpus = min(1, max_gpus)

    elif job_type == "segmentation":
        params = generate_segmentation_job_params()
        # Set required GPUs based on image size and batch size
        if params["image_size"] >= 768 or params["batch_size"] >= 32:
            required_gpus = min(max(1, np.random.choice([1, 2, 4])), max_gpus)
        else:
            required_gpus = min(1, max_gpus)

    # Calculate memory requirements (now we just do a simple approximation)
    if job_type == "llm":
        memory_per_param = 16  # bytes per parameter for mixed precision training
        model_size_params = params["hidden"] * params["hidden"] * params["layers"] * 12  # rough approximation
        batch_memory = params["batch_size"] * params["seq_len"] * params["hidden"] * 4  # activations
        total_memory_bytes = (model_size_params * memory_per_param) + batch_memory
        memory_gb = total_memory_bytes / 1e9
    elif job_type == "classification":
        memory_gb = params["batch_size"] * params["image_size"] * params["image_size"] * 3 * 4 / 1e9 * 10
    else:  # segmentation
        memory_gb = params["batch_size"] * params["image_size"] * params["image_size"] * 4 * 4 / 1e9 * 15

    # Create the final job parameters dictionary
    job_params = {
        "job_type": job_type,
        "batch_size": params["batch_size"],
        "dataset_size": params["dataset_size"],
        "gpu_type": gpu_type,
        "required_gpus": required_gpus,
        "gpu_memory_per_gpu": int(memory_gb * 1024 / required_gpus),  # Memory per GPU in MB
        "cpu_memory_total": int(memory_gb * 1024 * 1.5),  # 1.5x GPU memory in MB
        "total_episodes": np.random.randint(100, 1000),
        "max_runtime_hours": np.random.uniform(1, 24)
    }

    # Add job type specific parameters
    if job_type == "llm":
        job_params.update({
            "seq_len": params["seq_len"],
            "hidden": params["hidden"],
            "layers": params["layers"],
            "epochs": params["epochs"]
        })
    elif job_type == "classification":
        job_params.update({
            "image_size": params["image_size"],
            "initial_channels": params["initial_channels"],
            "epochs": params["epochs"]
        })
    elif job_type == "segmentation":
        job_params.update({
            "image_size": params["image_size"],
            "base_channels": params["base_channels"],
            "epochs": params["epochs"]
        })

    return job_params


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

        # Generate job parameters based on user type and tier
        params = generate_job_params(self.user_type, self.tier)

        # Create the job with the generated parameters
        job = Job(id=job_id, **params)

        # Increment job counter for this user
        self.job_id_counter += 1

        # Determine which queue to use
        queue_type = self.cluster.select_queue(job)

        # Add the job to the selected queue
        self.cluster.queues[queue_type].add_job(job, priority_score)

        return job

    @property
    def available_gpus(self) -> List[str]:
        """Get list of GPUs available to this user based on their tier"""
        return TIER_GPU_ACCESS[self.tier]

    @property
    def max_gpus(self) -> int:
        """Get maximum number of GPUs this user can request based on their tier"""
        return TIER_MAX_GPUS[self.tier]
