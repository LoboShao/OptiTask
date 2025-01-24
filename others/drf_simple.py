class GPUCluster:
    def __init__(self):
        """
        Initialize GPU cluster with different GPU types and their counts
        """
        self.gpu_pools = {}  # model -> {total: int, allocated: list of job_ids}
        self.jobs = {}  # job_id -> {gpu_model, gpu_count, memory, allocated: bool}
        self.allocated_resources = {}  # job_id -> {gpu_indices, memory}

    def add_gpu_pool(self, model: str, count: int, memory_per_gpu: int):
        """
        Add a pool of GPUs of specific model
        Example: add_gpu_pool('A100', 8, 80)  # 8 A100 GPUs with 80GB each
        """
        self.gpu_pools[model] = {
            'total': count,
            'memory_per_gpu': memory_per_gpu,
            'allocated': [],  # list of job_ids using these GPUs
            'available_indices': set(range(count))  # available GPU indices
        }

    def submit_job(self, job_id: str, gpu_model: str, gpu_count: int, memory_per_gpu: int):
        """
        Submit a job requesting specific GPU model and count
        Returns: bool indicating if job was accepted
        """
        if gpu_model not in self.gpu_pools:
            raise ValueError(f"Unknown GPU model: {gpu_model}")

        if memory_per_gpu > self.gpu_pools[gpu_model]['memory_per_gpu']:
            raise ValueError(f"Requested memory ({memory_per_gpu}GB) exceeds GPU capacity " +
                             f"({self.gpu_pools[gpu_model]['memory_per_gpu']}GB)")

        self.jobs[job_id] = {
            'gpu_model': gpu_model,
            'gpu_count': gpu_count,
            'memory_per_gpu': memory_per_gpu,
            'allocated': False,
            'submit_time': len(self.jobs)  # Use submission order as priority
        }
        return True

    def get_dominant_share(self, job_id: str) -> float:
        """
        Calculate job's dominant share of resources
        For GPU clusters, this is typically the GPU share
        """
        if job_id not in self.jobs or not self.jobs[job_id]['allocated']:
            return 0.0

        job = self.jobs[job_id]
        gpu_model = job['gpu_model']
        gpu_share = job['gpu_count'] / self.gpu_pools[gpu_model]['total']
        memory_share = (job['memory_per_gpu'] * job['gpu_count']) / \
                       (self.gpu_pools[gpu_model]['memory_per_gpu'] *
                        self.gpu_pools[gpu_model]['total'])

        return max(gpu_share, memory_share)

    def can_allocate(self, job_id: str) -> bool:
        """
        Check if job can be allocated required resources
        """
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        if job['allocated']:
            return False

        gpu_model = job['gpu_model']
        gpu_pool = self.gpu_pools[gpu_model]

        # Check if enough GPUs are available
        available_gpus = len(gpu_pool['available_indices'])
        if available_gpus < job['gpu_count']:
            return False

        return True

    def allocate_resources(self):
        """
        Implement DRF allocation for GPU cluster
        Allocates resources to jobs based on dominant resource shares
        """
        while True:
            # Find unallocated job with minimum dominant share
            min_share = float('inf')
            selected_job = None

            for job_id, job in self.jobs.items():
                if not job['allocated']:
                    share = self.get_dominant_share(job_id)
                    if share < min_share and self.can_allocate(job_id):
                        min_share = share
                        selected_job = job_id

            if selected_job is None:
                break

            # Allocate resources to selected job
            self._allocate_job(selected_job)

    def _allocate_job(self, job_id: str):
        """
        Allocate GPUs to a specific job
        """
        job = self.jobs[job_id]
        gpu_model = job['gpu_model']
        gpu_pool = self.gpu_pools[gpu_model]

        # Select GPUs for the job
        allocated_gpus = set()
        for _ in range(job['gpu_count']):
            gpu_idx = min(gpu_pool['available_indices'])
            allocated_gpus.add(gpu_idx)
            gpu_pool['available_indices'].remove(gpu_idx)

        # Update allocation records
        gpu_pool['allocated'].append(job_id)
        job['allocated'] = True
        self.allocated_resources[job_id] = {
            'gpu_indices': allocated_gpus,
            'memory_per_gpu': job['memory_per_gpu']
        }

    def release_job(self, job_id: str):
        """
        Release resources allocated to a job
        """
        if job_id not in self.jobs or not self.jobs[job_id]['allocated']:
            return

        job = self.jobs[job_id]
        gpu_model = job['gpu_model']
        gpu_pool = self.gpu_pools[gpu_model]

        # Release GPUs
        allocated_gpus = self.allocated_resources[job_id]['gpu_indices']
        gpu_pool['available_indices'].update(allocated_gpus)
        gpu_pool['allocated'].remove(job_id)

        # Update job status
        job['allocated'] = False
        del self.allocated_resources[job_id]

    def get_cluster_status(self):
        """
        Get current status of the cluster
        """
        status = {
            'gpu_pools': {},
            'jobs': {},
            'allocations': {}
        }

        # GPU pool status
        for model, pool in self.gpu_pools.items():
            status['gpu_pools'][model] = {
                'total': pool['total'],
                'available': len(pool['available_indices']),
                'memory_per_gpu': pool['memory_per_gpu']
            }

        # Job status
        for job_id, job in self.jobs.items():
            status['jobs'][job_id] = {
                'gpu_model': job['gpu_model'],
                'gpu_count': job['gpu_count'],
                'memory_per_gpu': job['memory_per_gpu'],
                'allocated': job['allocated'],
                'dominant_share': self.get_dominant_share(job_id)
            }

        # Allocation details
        status['allocations'] = self.allocated_resources

        return status


# Example usage
def main():
    # Initialize cluster
    cluster = GPUCluster()

    # Add GPU pools
    cluster.add_gpu_pool('A100', 8, 80)  # 8 A100 GPUs with 80GB memory each
    cluster.add_gpu_pool('V100', 16, 32)  # 16 V100 GPUs with 32GB memory each

    # Submit jobs
    jobs = [
        ('job1', 'A100', 2, 40),  # 2 A100s with 40GB memory each
        ('job2', 'V100', 4, 16),  # 4 V100s with 16GB memory each
        ('job3', 'A100', 4, 60),  # 4 A100s with 60GB memory each
        ('job4', 'V100', 8, 24),  # 8 V100s with 24GB memory each
    ]

    for job_id, gpu_model, gpu_count, memory in jobs:
        try:
            cluster.submit_job(job_id, gpu_model, gpu_count, memory)
            print(f"Submitted {job_id}: {gpu_count}x {gpu_model} with {memory}GB memory each")
        except ValueError as e:
            print(f"Failed to submit {job_id}: {e}")

    # Allocate resources using DRF
    cluster.allocate_resources()

    # Print cluster status
    status = cluster.get_cluster_status()

    print("\nCluster Status:")
    print("\nGPU Pools:")
    for model, pool in status['gpu_pools'].items():
        print(f"{model}: {pool['available']}/{pool['total']} available, "
              f"{pool['memory_per_gpu']}GB per GPU")

    print("\nJobs:")
    for job_id, job in status['jobs'].items():
        print(f"{job_id}: {job['gpu_count']}x {job['gpu_model']}, "
              f"{job['memory_per_gpu']}GB each, "
              f"Allocated: {job['allocated']}, "
              f"Share: {job['dominant_share']:.2%}")

    print("\nAllocations:")
    for job_id, alloc in status['allocations'].items():
        print(f"{job_id}: GPUs {alloc['gpu_indices']}, "
              f"{alloc['memory_per_gpu']}GB per GPU")


if __name__ == "__main__":
    main()