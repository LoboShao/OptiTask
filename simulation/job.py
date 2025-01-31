from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum
import numpy as np
from typing import Optional, List

from simulation.gpu import GPU, GPU_MODELS
from simulation.job_types import JOB_PROFILES

import time

class JobStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    job_type: str  # e.g., "llm", "classification", "segmentation"
    required_gpus: int
    gpu_type: str  # e.g., "A100", "T4"
    gpu_memory_per_gpu: int
    cpu_memory_total: int  # This will be the memory needed if run on single machine
    total_episodes: int
    batch_size: int
    max_runtime_hours: float
    current_episode: int = 0
    progress: float = 0.0
    start_time: Optional[float] = None  # Track when job started running
    priority: int = 0
    status: JobStatus = JobStatus.QUEUED
    allocated_gpus: Optional[List['GPU']] = None
    machine_memory_allocations: Dict[str, int] = field(default_factory=dict)
    allocated_machines: Set['Machine'] = field(default_factory=set)
    allocated_racks: Set['Rack'] = field(default_factory=set)


    def __post_init__(self):
        gpu_spec = GPU_MODELS[self.gpu_type]

        # Get the function for this job_type, or default if not found
        profile_fn = JOB_PROFILES.get(
            self.job_type,
            lambda b: (1e9 * b, 1.0 * b)  # default fallback
        )

        # Call the profile function
        flops_per_episode, data_per_episode_gb = profile_fn(self.batch_size)

        # 1) Compute-limited time
        compute_time_per_episode = flops_per_episode / gpu_spec.peak_ops
        total_compute_time = compute_time_per_episode * self.total_episodes

        # 2) Memory-limited time
        memory_time_per_episode = data_per_episode_gb / gpu_spec.effective_bandwidth
        total_memory_time = memory_time_per_episode * self.total_episodes

        # 3) Bottleneck time
        time_per_episode = max(compute_time_per_episode, memory_time_per_episode)
        estimated_time_seconds = time_per_episode * self.total_episodes

        # 4) Multi-GPU scaling (simple heuristic: 90% efficiency per extra GPU)
        if self.required_gpus > 1:
            scaling_efficiency = 0.9 ** (self.required_gpus - 1)
            # Invert because more GPUs -> less time
            estimated_time_seconds *= 1.0 / scaling_efficiency
        # Store training estimates
        self._training_estimates = {
            "compute_time_seconds": total_compute_time,
            "memory_time_seconds": total_memory_time,
            "bottleneck_time_seconds": estimated_time_seconds,
            "estimated_time_hours": estimated_time_seconds / 3600,
            "episode_per_second": 1/time_per_episode
        }

    def apply_machine_rack_scaling(self):
        """Apply multi-machine scaling after machine allocation."""
        num_machines = len(self.allocated_machines)
        num_racks = len(self.allocated_racks)
        if num_machines > 1:
            machine_scaling_efficiency = 0.8 ** (num_machines - 1)  # 80% per extra machine
            self._training_estimates["bottleneck_time_seconds"] /= machine_scaling_efficiency
            self._training_estimates["estimated_time_hours"] = self._training_estimates[
                                                                   "bottleneck_time_seconds"] / 3600
            self._training_estimates["episode_per_second"] *= machine_scaling_efficiency  # Update episodes per second
        if num_racks > 1:
            machine_scaling_efficiency = 0.7 ** (num_racks - 1)  # 80% per extra machine
            self._training_estimates["bottleneck_time_seconds"] /= machine_scaling_efficiency
            self._training_estimates["estimated_time_hours"] = self._training_estimates[
                                                                   "bottleneck_time_seconds"] / 3600
            self._training_estimates["episode_per_second"] *= machine_scaling_efficiency  # Update episodes per second
        return self._training_estimates


    def get_training_estimates(self) -> Dict:
        """Return stored training estimates"""
        return self._training_estimates or {"error": f"No estimates available for GPU type: {self.gpu_type}"}

    def check_timeout(self, current_time: float) -> bool:
        """Check if job has exceeded max runtime"""
        if self.start_time is None or self.status != JobStatus.RUNNING:
            return False
        return (current_time - self.start_time) / 3600 > self.max_runtime_hours


    def step(self, time_interval: float) -> bool:
        """Simulate training progress for one time unit"""
        if self.status != JobStatus.RUNNING:
            return False
        current_progress  = time_interval * self._training_estimates["episode_per_second"]
        current_progress *= np.random.uniform(0.9, 1.1)  # Add ±10% random variation

        self.current_episode += current_progress
        self.progress = self.current_episode/self.total_episodes
        if self.current_episode >= self.total_episodes:
            self.status = JobStatus.COMPLETED
            return True

        return False



if __name__ == "__main__":
    # Example: Large language model job with batch_size=4 on an A100
    job_llm = Job(
        id="job_llm_001",
        job_type="llm",
        required_gpus=4,
        gpu_type="A100_80GB",
        gpu_memory_per_gpu=40_000,
        cpu_memory_total=256_000,
        total_episodes=100,
        batch_size=256,
        max_runtime_hours=48.0,
    )

    est_llm = job_llm.get_training_estimates()
    print("=== LLM Job Estimates ===")
    print(f"Compute-limited total time (sec): {est_llm['compute_time_seconds']:.2f}")
    print(f"Memory-limited total time (sec): {est_llm['memory_time_seconds']:.2f}")
    print(f"Bottleneck time (sec): {est_llm['bottleneck_time_seconds']:.2f}")
    print(f"Estimated time (hours): {est_llm['estimated_time_hours']:.2f}")

    job_llm.status = JobStatus.RUNNING
    print(job_llm)
    # Example: Classification job with batch_size=128 on a T4
    job_cls = Job(
        id="job_cls_001",
        job_type="classification",
        required_gpus=2,
        gpu_type="T4",
        gpu_memory_per_gpu=16_000,
        cpu_memory_total=64_000,
        total_episodes=200,
        batch_size=1024,
        max_runtime_hours=24.0,
    )

    est_cls = job_cls.get_training_estimates()
    print("\n=== Classification Job Estimates ===")
    print(f"Compute-limited total time (sec): {est_cls['compute_time_seconds']:.2f}")
    print(f"Memory-limited total time (sec): {est_cls['memory_time_seconds']:.2f}")
    print(f"Bottleneck time (sec): {est_cls['bottleneck_time_seconds']:.2f}")
    print(f"Estimated time (hours): {est_cls['estimated_time_hours']:.2f}")