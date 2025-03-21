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

    def get_next_job(self, cluster) -> Optional[Job]:
        if not self.queue:
            return None

        # Filter to only schedulable jobs
        valid_jobs = {
            job_id: (score, job)
            for job_id, (score, job) in self.queue.items()
            if cluster.can_schedule_job(job)
        }

        if not valid_jobs:
            return None

        job_id, (_, job) = max(valid_jobs.items(), key=lambda x: x[1][0])
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
    def schedule_next_job(self) -> Optional[Job]:
        """
        Try scheduling highest priority job that can run with available resources.
        Returns scheduled job if successful, None if no jobs could be scheduled.
        """

        for queue_type in QueueType:
            queue = self.queues[queue_type]
            job = queue.get_next_job(self)  # Pass cluster reference for resource checking

            if job is None:
                continue

            # At this point job is known to be schedulable
            success = self.allocate_job(job)
            if success:
                job.apply_machine_rack_scaling()
                self.running_jobs[job.id] = job
                return job

        return None


    def allocate_job(self, job: Job) -> List[GPU]:
        gpu_model = job.gpu_type
        required_count = job.required_gpus
        cpu_memory = job.cpu_memory_total
        rack = self._find_rack_with_gpus(required_count, cpu_memory, gpu_model)
        if rack:
            return rack.allocate_gpus(num_gpus=required_count, job=job, model_name=gpu_model)
        else:
            return self._allocate_across_racks(num_gpus=required_count, job=job, model_name=gpu_model)


    def _find_rack_with_gpus(self, required_count: int, required_memory: int, gpu_model: GPU_MODELS) -> Union[Rack, None]:
        racks_with_counts = [
            (rack_id, rack.count_available_gpus_by_model(model_name=gpu_model), rack.count_avalible_memory())
            for rack_id, rack in self.rack_map.items()  # Use items() to get both key and value
        ]
        valid_racks = [(rack, count, memory) for rack, count, memory in racks_with_counts if count >= required_count and memory >= required_memory]
        if not valid_racks:
            return None
        else:
            index = max(valid_racks, key=lambda x: x[1])[0]
            return self.rack_map[index]

    def _allocate_across_racks(self, num_gpus: int, job: Job, model_name: GPU_MODELS) -> List[GPU]:
        racks_with_counts = [rack.count_available_gpus_by_model(model_name) for rack in self.racks]
        print(racks_with_counts)
        selected_gpus = []
        if sum(racks_with_counts) < num_gpus:
            print('invalid racks')
            return selected_gpus
        else:
            rest = num_gpus
            for idx, count in enumerate(racks_with_counts):
                if count > 0:
                    if rest >= count:
                        selected_gpus.extend(self.racks[idx].allocate_gpus(num_gpus=count, job=job, model_name=model_name))
                        rest -= count
                    else:
                        selected_gpus.extend(self.racks[idx].allocate_gpus(num_gpus=rest, job=job, model_name=model_name))
            return selected_gpus

    # def can_schedule_job(self, job: Job) -> bool:
    #     # Quick check if job can possibly be scheduled
    #     available_gpus = sum(
    #         rack.count_available_gpus_by_model(job.gpu_type)
    #         for rack in self.racks
    #     )
    #     return available_gpus >= job.required_gpus

    def can_schedule_job(self, job: Job) -> bool:
        gpu_count = 0
        total_memory = 0

        # Only check machines with matching GPU type
        for rack in self.racks:
            for machine in rack.machines:
                if machine.has_gpu_type(job.gpu_type):
                    gpu_count += machine.count_available_gpus_by_model(job.gpu_type)
                    total_memory += machine.available_cpu_memory

        # Both GPU and memory requirements must be met
        return (gpu_count >= job.required_gpus and
                total_memory >= job.cpu_memory_total)
