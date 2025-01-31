from typing import Dict

from simulation.job import Job
from simulation.job_types import JOB_PROFILES

def calculate_distributed_memory(self, num_gpus_on_machine: int, is_cross_rack: bool = False) -> Dict[str, int]:
    """
    Calculate CPU memory needed for distributed training on a specific machine

    Args:
        num_gpus_on_machine: Number of GPUs to be allocated on this machine
        is_cross_rack: Whether this allocation involves cross-rack communication

    Returns:
        Dict containing:
            - base_memory: Base memory allocation
            - communication_overhead: Additional memory for distributed training
            - total: Total memory needed
    """
    # Get job profile characteristics
    profile_fn = JOB_PROFILES.get(
        self.job_type,
        lambda b: (1e9 * b, 1.0 * b)  # default fallback
    )
    _, data_per_episode_gb = profile_fn(self.batch_size)

    # Base memory allocation proportional to GPUs on this machine
    base_memory = (self.cpu_memory_total * num_gpus_on_machine) // self.required_gpus

    # Calculate communication overhead if distributed
    communication_overhead = 0
    if len(self.allocated_machines) > 0 or num_gpus_on_machine < self.required_gpus:
        # Base communication buffer proportional to data movement
        communication_overhead = int(data_per_episode_gb * 1024)  # Convert to MB

        # Scale based on number of GPUs and their memory
        gpu_memory_factor = self.gpu_memory_per_gpu / 1024  # Convert to GB
        communication_overhead = int(communication_overhead *
                                     (num_gpus_on_machine / self.required_gpus) *
                                     min(1.0, gpu_memory_factor / 80))  # Normalize to 80GB as baseline

        if is_cross_rack:
            # Add 50% more overhead for cross-rack communication
            communication_overhead = int(communication_overhead * 1.5)

        # Add base overhead for distributed coordination
        communication_overhead += 2048  # 2GB base overhead

    total_memory = base_memory + communication_overhead

    return {
        "base_memory": base_memory,
        "communication_overhead": communication_overhead,
        "total": total_memory
    }