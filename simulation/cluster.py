from typing import List, Dict, Tuple, Optional,Union
from dataclasses import dataclass, field
from simulation.machine import Machine
from simulation.rack import Rack
from simulation.job import Job, JobStatus
from simulation.gpu import GPU, GPU_MODELS, GPUStatus
from enum import Enum
from collections import OrderedDict

import numpy as np

class QueueType(Enum):
    HIGH_PRIORITY = "high_priority"    # H100/A100_80GB
    STANDARD = "standard"              # A100_40GB
    DEVELOPMENT = "development"        # T4
    AUTO = "auto"                      # For unspecified GPU types

class JobQueue:
    def __init__(self, queue_type: QueueType):
        self.type = queue_type
        self.queue = OrderedDict()  # job_id -> (priority_score, job)

    def add_job(self, job: Job, priority_score: float):
        self.queue[job.id] = (priority_score, job)

    def get_next_job(self) -> Optional[Job]:
        if not self.queue:
            return None
        job_id, (_, job) = max(self.queue.items(), key=lambda x: x[1][0])
        del self.queue[job_id]
        return job


class Cluster:
    def __init__(self, racks: List[Rack]):
        self.racks = racks
        self.machine_map = {
            machine.id: machine
            for rack in self.racks
            for machine in rack.machines
        }
        self.rack_map = {rack.id: rack for rack in self.racks}
        self.queues = {
            QueueType.HIGH_PRIORITY: JobQueue(QueueType.HIGH_PRIORITY),
            QueueType.STANDARD: JobQueue(QueueType.STANDARD),
            QueueType.DEVELOPMENT: JobQueue(QueueType.DEVELOPMENT),
            QueueType.AUTO: JobQueue(QueueType.AUTO)
        }
        self.running_jobs: Dict[str, Job] = {}
        self.completed_jobs: Dict[str, Job] = {}

    # ------------------------ Job Queue Functions ------------------------------ #
    def select_queue(self, job: Job) -> QueueType:
        if job.gpu_type is None:
            return QueueType.AUTO
        elif job.gpu_type in ["H100_80GB", "A100_80GB"]:
            return QueueType.HIGH_PRIORITY
        elif job.gpu_type == "A100_40GB":
            return QueueType.STANDARD
        else:
            return QueueType.DEVELOPMENT

    def add_running_job(self, job: Job):
        """Add a job to running jobs"""
        self.running_jobs[job.id] = job

    def print_queue_status(self):
        """
        Print and return detailed status of all job queues
        Returns:
            Dictionary containing queue statistics for each queue type
        """

        print("\n=== Job Queue Status ===")
        for queue_type in QueueType:
            queue = self.queues[queue_type]
            jobs = list(queue.queue.values())

            # Calculate queue statistics
            total_jobs = len(jobs)
            # Print queue information
            print(f"\n{queue_type.value.upper()} Queue:")
            print(f"Total jobs waiting: {total_jobs}")

            if jobs:
                print("\nJobs in queue:")
                print(f"{'Job ID':<15} {'Priority':<10} {'GPUs':<6} {'Memory (MB)':<12} {'GPU Type':<10}")
                print("-" * 55)

                for priority_score, job in jobs:
                    gpu_type = job.gpu_type or "Any"
                    print(f"{job.id:<15} {priority_score:<10.2f} {job.required_gpus:<6} "
                          f"{job.gpu_memory_per_gpu:<12} {gpu_type:<10}")
            else:
                print("No jobs in queue")

        # Print summary of running jobs
        print("\n=== Running Jobs ===")
        print(f"Total running jobs: {len(self.running_jobs)}")
        if self.running_jobs:
            print("\nCurrently running jobs:")
            print(f"{'Job ID':<15} {'GPUs':<6} {'Memory (MB)':<12} {'GPU Type':<10}")
            print("-" * 45)
            for job_id, job in self.running_jobs.items():
                gpu_type = job.gpu_type or "Any"
                print(f"{job_id:<15} {job.required_gpus:<6} "
                      f"{job.gpu_memory_per_gpu:<12} {gpu_type:<10}")


    # ------------------------ Job GPU Assignment Functions ------------------------------ #
    def allocate_job(self, job: Job):
        gpu_model = job.gpu_type
        required_count = job.required_gpus

        machine = self._find_machine_with_gpus(required_count, gpu_model)
        if machine:
            return machine.allocate_gpus(num_gpus=required_count, job_id=job.id, model_name=gpu_model)

        rack = self._find_rack_with_gpus(required_count, gpu_model)
        if rack:
            return rack.allocate_gpus(num_gpus=required_count, job_id=job.id, model_name=gpu_model)

        return self._allocate_across_racks(num_gpus=required_count, job_id=job.id, model_name=gpu_model)

    def _find_machine_with_gpus(self, required_count: int, gpu_model: GPU_MODELS) -> Union[Machine, None]:
        machines_with_counts = [
            (machine_id, machine.count_available_gpus_by_model(model_name=gpu_model))
            for machine_id, machine in self.machine_map.items()  # Use items() to get both key and value
        ]

        valid_machines = [(machine_id, count) for machine_id, count in machines_with_counts if count >= required_count]
        if not valid_machines:
            return None
        else:

            index = max(valid_machines, key=lambda x: x[1])[0]
            return self.machine_map[index]

    def _find_rack_with_gpus(self, required_count: int, gpu_model: GPU_MODELS) -> Union[Rack, None]:
        racks_with_counts = [
            (rack_id, rack.count_available_gpus_by_model(model_name=gpu_model))
            for rack_id, rack in self.rack_map.items()  # Use items() to get both key and value
        ]
        valid_racks = [(rack, count) for rack, count in racks_with_counts if count >= required_count]
        if not valid_racks:
            return None
        else:
            index = max(valid_racks, key=lambda x: x[1])[0]
            return self.rack_map[index]

    def _allocate_across_racks(self, num_gpus: int, job_id: str, model_name: GPU_MODELS) -> List[GPU]:
        racks_with_counts = [rack.count_available_gpus_by_model(model_name) for rack in self.racks]
        selected_gpus = []
        if sum(racks_with_counts) < num_gpus:
            return selected_gpus
        else:
            rest = num_gpus
            for idx, count in enumerate(racks_with_counts):
                if count > 0:
                    if rest >= count:
                        selected_gpus.extend(self.racks[idx].allocate_gpus(num_gpus=count, job_id=job_id, model_name=model_name))
                        rest -= count
                    else:
                        selected_gpus.extend(self.racks[idx].allocate_gpus(num_gpus=rest, job_id=job_id, model_name=model_name))
            return selected_gpus
