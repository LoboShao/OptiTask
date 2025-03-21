from dataclasses import dataclass
from platform import machine
from typing import List, Optional, Dict, Tuple

from simulation.machine import Machine
from simulation.gpu_model import GPU_MODELS
from simulation.gpu import GPU
from simulation.job import Job

@dataclass
class Rack:
    id: str
    name: str
    machines: List[Machine]


    def count_available_gpus_by_model(self, model_name: Optional[str] = None) -> int:
        return sum([machine.count_available_gpus_by_model(model_name=model_name) for machine in self.machines])

    def count_avalible_memory(self) -> int:
        return sum([machine.available_cpu_memory for machine in self.machines])

    # def allocate_gpus(self, num_gpus: int, job: Job, model_name: GPU_MODELS):
    #     selected_gpus = []
    #     rest = num_gpus
    #     rest_memory = job.cpu_memory_total
    #     machines_with_counts = [machine.count_available_gpus_by_model(model_name=model_name) for machine in self.machines]
    #     job.allocated_racks.add(self.id)
    #     for idx, count in enumerate(machines_with_counts):
    #         if count > 0 and rest > 0:
    #             if rest >= count:
    #                 selected_gpus.extend(self.machines[idx].allocate_gpus(num_gpus=count, job=job, model_name=model_name))
    #                 rest -= count
    #             else:
    #                 selected_gpus.extend(self.machines[idx].allocate_gpus(num_gpus=rest, job=job, model_name=model_name))
    #                 break
    #     return selected_gpus
    def allocate_gpus(self, num_gpus: int, job: Job, model_name: GPU_MODELS):
        """
        Allocate GPUs across multiple machines in this rack.
        Assumes that the rack has enough total GPUs and memory for the job.

        Args:
            num_gpus: Total number of GPUs to allocate
            job: Training job that needs resources
            model_name: Specific GPU model type required
        Returns:
            List of allocated GPUs
        """
        selected_gpus = []
        remaining_gpus = num_gpus

        # Get machines with available GPUs of the requested model
        machines_with_counts = [(machine, machine.count_available_gpus_by_model(model_name=model_name))
                                for machine in self.machines]

        # Sort machines by GPU availability to optimize allocation
        machines_with_counts.sort(key=lambda x: x[1], reverse=True)

        job.allocated_racks.add(self.id)

        for machine, available_count in machines_with_counts:
            if available_count == 0 or remaining_gpus == 0:
                continue

            # Calculate GPUs to allocate from this machine
            gpus_to_allocate = min(remaining_gpus, available_count)

            # Allocate GPUs from this machine
            allocated_gpus = machine.allocate_gpus(num_gpus=gpus_to_allocate, job=job, model_name=model_name)

            # Update tracking
            selected_gpus.extend(allocated_gpus)
            remaining_gpus -= len(allocated_gpus)

            # Break if we've allocated all requested GPUs
            if remaining_gpus == 0:
                break

        return selected_gpus
