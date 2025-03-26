from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from simulation.gpu import GPU, GPUStatus, GPU_MODELS
from simulation.job import Job

import logging


@dataclass
class Machine:
    id: str
    gpus: List[GPU]
    rack_id: str
    total_cpu_memory: int = 256_000  # Default to 256GB in MB
    used_cpu_memory: int = 0
    cpu_allocations: Dict[str, int] = field(default_factory=dict)

    @property
    def available_cpu_memory(self) -> int:
        return self.total_cpu_memory - self.used_cpu_memory

    @property
    def available_gpus(self) -> List[GPU]:
        """Get list of available GPUs"""
        return [gpu for gpu in self.gpus if gpu.status == GPUStatus.IDLE]

    def available_gpus_by_model(self, model_name: GPU_MODELS) -> List[GPU]:
        """
        Get available GPUs of specific model
        Args:
            model_name: The specific GPU model type to filter for. If None, returns all available GPUs.
        Returns:
            List of available GPUs matching the specified model
        """
        if model_name is None:
            return self.available_gpus
        return [gpu for gpu in self.available_gpus if gpu.model.name == model_name]

    def count_available_gpus_by_model(self, model_name: Optional[str] = None) -> int:
        """
        Count available GPUs of specific model
        Args:
            model_name: The specific GPU model type to count. If None, counts all available GPUs.
        Returns:
            Number of available GPUs matching the specified model
        """
        return len(self.available_gpus_by_model(model_name))

    # def allocate_gpus(self, num_gpus: int, job: Job, model_name: GPU_MODELS) -> List[GPU]:
    #     """
    #     Allocate GPUs of specific model if specified
    #     Args:
    #         num_gpus: Number of GPUs to allocate
    #         job: Training job
    #         model_name: Specific GPU model type required. If None, any GPU type can be allocated.
    #     Returns:
    #         List of allocated GPUs, empty list if allocation failed
    #     """
    #     available_gpus = self.available_gpus_by_model(model_name)
    #     if len(available_gpus) < num_gpus:
    #         print('xx'*100)
    #         return []
    #     job.allocated_machines.add(self.id)
    #     self.allocate_cpu_memory(job_id=job.id,
    #                              required_memory=int(job.cpu_memory_total * (num_gpus/job.required_gpus)))
    #     # Instead of random selection, prioritize GPUs based on physical proximity
    #     # This can help with better performance for distributed training
    #     selected_gpus = available_gpus[:num_gpus]  # Take first n available GPUs
    #     for gpu in selected_gpus:
    #         gpu.allocate(job.id)
    #     return selected_gpus
    def allocate_gpus(self, num_gpus: int, job: Job, model_name: GPU_MODELS) -> List[GPU]:
        available_gpus = self.available_gpus_by_model(model_name)
        if len(available_gpus) < num_gpus:
            # Log a warning or raise an exception
            logging.warning(f"Machine {self.id} does not have enough GPUs for job {job.id}")
            return []
        job.allocated_machines.add(self.id)
        self.allocate_cpu_memory(job_id=job.id,
                                 required_memory=int(job.cpu_memory_total * (num_gpus / job.required_gpus)))
        selected_gpus = available_gpus[:num_gpus]
        for gpu in selected_gpus:
            gpu.allocate(job.id)
        return selected_gpus

    def get_gpu_availability_by_model(self) -> Dict[str, int]:
        """
        Get count of available GPUs grouped by model type
        Returns:
            Dictionary mapping GPU model names to count of available GPUs
        """
        availability = {}
        for gpu in self.available_gpus:
            model = gpu.model.name
            if model not in availability:
                availability[model] = 0
            availability[model] += 1
        return availability

    def allocate_cpu_memory(self, job_id: str, required_memory: int) -> bool:
        """
        Attempt to allocate CPU memory for a job
        Returns True if allocation successful
        """
        if self.available_cpu_memory >= required_memory:
            self.used_cpu_memory += required_memory
            self.cpu_allocations[job_id] = required_memory
            return True
        return False

    def release_cpu_memory(self, job_id: str) -> None:
        """Release CPU memory allocated to a job"""
        if job_id in self.cpu_allocations:
            self.used_cpu_memory -= self.cpu_allocations[job_id]
            del self.cpu_allocations[job_id]

    def release_resources(self, job_id: str) -> None:
        """Release both GPU and CPU resources for a job"""
        # Release GPUs
        for gpu in self.gpus:
            if gpu.current_job == job_id:
                gpu.release()
        # Release CPU memory
        self.release_cpu_memory(job_id)

    def get_resource_usage(self) -> dict:
        """Get current resource usage statistics"""
        gpu_usage_by_model = self.get_gpu_availability_by_model()
        total_gpus_by_model = {}
        for gpu in self.gpus:
            model = gpu.model.name
            if model not in total_gpus_by_model:
                total_gpus_by_model[model] = 0
            total_gpus_by_model[model] += 1

        return {
            "total_gpus_by_model": total_gpus_by_model,
            "available_gpus_by_model": gpu_usage_by_model,
            "total_cpu_memory": self.total_cpu_memory,
            "used_cpu_memory": self.used_cpu_memory,
            "jobs": list(self.cpu_allocations.keys())
        }

    def has_gpu_type(self, gpu_type: str) -> bool:
        """
        Check if machine has any GPUs of the specified type.
        Args:
            gpu_type: Type of GPU to check for (e.g., "A100_80GB")
        Returns:
            True if machine has at least one GPU of specified type
        """
        return any(gpu.model.name == gpu_type for gpu in self.gpus)
