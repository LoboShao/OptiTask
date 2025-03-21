from dataclasses import dataclass
from typing import Dict, Optional
from simulation.gpu import GPUModel, GPU_MODELS
import math


@dataclass
class TrainingJob:
    model_size_gb: float  # Model size in GB
    batch_size: int
    total_samples: int
    precision: str = "fp16"  # "fp16" or "fp32"


def estimate_training_time(job: TrainingJob, gpu: GPUModel) -> dict:
    """
    Estimate training time based on job requirements and GPU specifications.
    Returns a dictionary with time estimates and utilization metrics.
    """
    # Memory utilization check
    memory_overhead = 1.2  # 20% overhead for optimizers, gradients, etc
    total_memory_needed = job.model_size_gb * memory_overhead

    if total_memory_needed > gpu.memory:
        return {
            "error": "Model too large for GPU memory",
            "memory_required": total_memory_needed,
            "gpu_memory": gpu.memory
        }

    # Calculate theoretical memory bandwidth utilization
    # Amount of data moved per sample (model params + activations + gradients)
    data_per_sample = job.model_size_gb * 3

    # Adjust for precision
    bytes_multiplier = 2 if job.precision == "fp16" else 4
    data_per_sample_bytes = data_per_sample * bytes_multiplier * 1024 * 1024 * 1024  # Convert to bytes

    # Theoretical time per batch based on memory bandwidth
    time_per_batch_memory = (data_per_sample_bytes * job.batch_size) / (gpu.bandwidth * 1e9)

    # Compute capability factor (newer architectures are more efficient)
    compute_efficiency = {
        "7.5": 0.6,  # T4 (Turing)
        "8.0": 0.8,  # A100 (Ampere)
        "9.0": 1.0,  # H100 (Hopper)
    }.get(gpu.compute_capability, 0.5)
    # Adjust time based on architecture efficiency and tensor cores
    tensor_core_factor = (gpu.tensor_cores / 432) * compute_efficiency  # Normalized to A100
    compute_time_factor = 1.0 / tensor_core_factor

    # Final time estimation
    time_per_batch = time_per_batch_memory * compute_time_factor
    total_batches = math.ceil(job.total_samples / job.batch_size)
    estimated_total_time = time_per_batch * total_batches

    # Calculate efficiency metrics
    memory_utilization = (total_memory_needed / gpu.memory) * 100
    bandwidth_utilization = (data_per_sample_bytes * job.batch_size / time_per_batch) / (gpu.bandwidth * 1e9) * 100

    return {
        "estimated_time_seconds": estimated_total_time,
        "estimated_time_hours": estimated_total_time / 3600,
        "time_per_batch_seconds": time_per_batch,
        "memory_utilization_percent": memory_utilization,
        "bandwidth_utilization_percent": bandwidth_utilization,
        "total_batches": total_batches,
        "gpu_compute_efficiency": compute_efficiency,
    }


if __name__ == '__main__':

    # Example usage:
    job = TrainingJob(
        model_size_gb=10,  # 10GB model
        batch_size=32,
        total_samples=1000000,
        precision="fp16"
    )

    # Estimate for different GPUs
    for gpu_name, gpu_model in GPU_MODELS.items():
        print(f"\nEstimating for {gpu_name}:")
        estimate = estimate_training_time(job, gpu_model)
        for key, value in estimate.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")