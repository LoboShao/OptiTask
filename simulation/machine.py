from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from simulation.gpu import GPU, GPUStatus, GPU_MODELS
from simulation.job import Job


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
        available = self.available_gpus
        if model_name is None:
            return available
        return [gpu for gpu in available if gpu.model.name == model_name]

    def count_available_gpus_by_model(self, model_name: Optional[str] = None) -> int:
        """
        Count available GPUs of specific model
        Args:
            model_name: The specific GPU model type to count. If None, counts all available GPUs.
        Returns:
            Number of available GPUs matching the specified model
        """
        return len(self.available_gpus_by_model(model_name))

    def allocate_gpus(self, num_gpus: int, job_id: str, model_name: GPU_MODELS) -> List[GPU]:
        """
        Allocate GPUs of specific model if specified
        Args:
            num_gpus: Number of GPUs to allocate
            job_id: ID of the job requesting allocation
            model_name: Specific GPU model type required. If None, any GPU type can be allocated.
        Returns:
            List of allocated GPUs, empty list if allocation failed
        """
        available_gpus = self.available_gpus_by_model(model_name)
        if len(available_gpus) < num_gpus:
            return []

        # Instead of random selection, prioritize GPUs based on physical proximity
        # This can help with better performance for distributed training
        selected_gpus = available_gpus[:num_gpus]  # Take first n available GPUs
        for gpu in selected_gpus:
            gpu.allocate(job_id)
        return selected_gpus

    def has_sufficient_resources(self, required_gpus: int, required_memory: int,
                                 gpu_type: Optional[str] = None) -> bool:
        """
        Check if machine has sufficient resources for a job
        Args:
            required_gpus: Number of GPUs required
            required_memory: Amount of CPU memory required (in MB)
            gpu_type: Specific GPU model type required. If None, any GPU type is acceptable.
        Returns:
            True if machine has sufficient resources, False otherwise
        """
        available_matching_gpus = self.count_available_gpus_by_model(gpu_type)
        return (available_matching_gpus >= required_gpus and
                self.available_cpu_memory >= required_memory)

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

if __name__ == '__main__':
    # Create a machine with 4 GPUs
    machine = Machine(
        id="node1",
        gpus=[
            GPU(id="gpu_1", model=GPU_MODELS['T4']),
            GPU(id="gpu_2", model=GPU_MODELS['T4']),
            GPU(id="gpu_3", model=GPU_MODELS['T4']),
            GPU(id="gpu_4", model=GPU_MODELS['T4'])
        ],
        rack_id="rack1",
        total_cpu_memory=256_000  # 256GB
    )

    # Example 1: Successfully allocate resources for a job
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

    # Check if we have enough resources
    if machine.has_sufficient_resources(job_cls.required_gpus, job_cls.cpu_memory_total):
        # Try to allocate GPUs
        allocated_gpus = machine.allocate_gpus(job_cls.required_gpus, job_cls.id)
        if allocated_gpus and machine.allocate_cpu_memory(job_cls.id, job_cls.cpu_memory_total):
            print(f"Job {job_cls.id} successfully allocated resources:")
            print(f"- GPUs: {[gpu.id for gpu in allocated_gpus]}")
            print(f"- CPU Memory: {job_cls.cpu_memory_total / 1000}GB")
        else:
            # If CPU allocation failed, release any allocated GPUs
            for gpu in allocated_gpus:
                gpu.release()
            print("Resource allocation failed")
    print(machine.get_resource_usage())

    # Example 3: Try to allocate more resources than available
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

    if machine.has_sufficient_resources(job_llm.required_gpus, job_llm.cpu_memory_total):
        print("\nCan allocate job 2")
    else:
        print("\nInsufficient resources for job 2")
        print(f"Available GPUs: {len(machine.available_gpus)}")
        print(f"Available CPU Memory: {machine.available_cpu_memory / 1000}GB")

    # Example 4: Release resources
    machine.release_resources(job_llm.id)
    print(machine.get_resource_usage())