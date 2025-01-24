from simulation.cluster import Cluster
from simulation.gpu import GPU
from simulation.machine import Machine
from simulation.rack import Rack
from simulation.job import Job
from simulation.cluster_builder import create_cluster_from_config

def create_cluster_small():
    # Create a cluster with 2 racks, each having 2 machines with 4 GPUs each
    # Rack 0: machine_0 (4 GPUs), machine_1 (4 GPUs)
    # Rack 1: machine_2 (4 GPUs), machine_3 (4 GPUs)

    def create_gpus(machine_id: str, num_gpus: int = 4):
        return [GPU(id=f"{machine_id}_gpu_{i}", memory=40000) for i in range(num_gpus)]

    # Create machines for rack 0
    rack0_machines = [
        Machine(
            id=f"rack0_machine_{i}",
            gpus=create_gpus(f"rack0_machine_{i}"),
            rack_id="rack0",
            total_cpu_memory=256_000,  # 256GB
            used_cpu_memory=0
        ) for i in range(2)
    ]

    # Create machines for rack 1
    rack1_machines = [
        Machine(
            id=f"rack1_machine_{i}",
            gpus=create_gpus(f"rack1_machine_{i}"),
            rack_id="rack1",
            total_cpu_memory=256_000,  # 256GB
            used_cpu_memory=0
        ) for i in range(2)
    ]

    # Create racks
    racks = [
        Rack(id="rack0", machines=rack0_machines),
        Rack(id="rack1", machines=rack1_machines)
    ]

    return Cluster(racks)


def create_large_cluster():
    """Create a larger cluster with 4 racks, each having 4 machines with 8 GPUs each"""

    def create_gpus(machine_id: str, num_gpus: int = 8):
        return [GPU(id=f"{machine_id}_gpu_{i}", memory=80000) for i in range(num_gpus)]  # 80GB GPUs

    racks = []
    for rack_id in range(4):  # 4 racks
        rack_machines = [
            Machine(
                id=f"rack{rack_id}_machine_{i}",
                gpus=create_gpus(f"rack{rack_id}_machine_{i}"),
                rack_id=f"rack{rack_id}",
                total_cpu_memory=512_000,  # 512GB
                used_cpu_memory=0
            ) for i in range(4)  # 4 machines per rack
        ]
        racks.append(Rack(id=f"rack{rack_id}", machines=rack_machines))

    return Cluster(racks)

def print_allocation_info(job: Job):
    print(f"\nJob {job.id} allocation info:")
    print(f"Total GPUs required: {job.required_gpus}")
    print(f"Machines used: {len(job.allocated_machines)}")

    # Group machines by rack
    rack_machines = {}
    for machine in job.allocated_machines:
        rack_machines.setdefault(machine.rack_id, []).append(machine)

    for rack_id, machines in rack_machines.items():
        print(f"\nRack {rack_id}:")
        for machine in machines:
            gpu_count = len([gpu for gpu in machine.gpus if gpu.current_job == job.id])
            print(f"  Machine {machine.id}:")
            print(f"    - GPUs allocated: {gpu_count}")
            print(f"    - Memory allocated: {job.machine_memory_allocations[machine.id]}MB")


def test_small_cluster_allocation():
    cluster = create_cluster_small()

    # Scenario 1: Single machine job (4 GPUs)
    print("\n=== Scenario 1: Single Machine Job ===")
    job1 = Job(
        id="single_machine_job",
        required_gpus=4,
        gpu_memory_per_gpu=16000,
        cpu_memory_total=32000,
        total_episodes=1000
    )

    success = cluster.allocate_job(job1)
    if success:
        print_allocation_info(job1)

    # Scenario 2: Same rack job (6 GPUs)
    print("\n=== Scenario 2: Same Rack Job ===")
    job2 = Job(
        id="same_rack_job",
        required_gpus=6,
        gpu_memory_per_gpu=16000,
        cpu_memory_total=48000,
        total_episodes=1000
    )

    success = cluster.allocate_job(job2)
    if success:
        print_allocation_info(job2)

    # Scenario 3: Cross rack job (8 GPUs)
    print("\n=== Scenario 3: Cross Rack Job ===")
    job3 = Job(
        id="cross_rack_job",
        required_gpus=8,
        gpu_memory_per_gpu=16000,
        cpu_memory_total=64000,
        total_episodes=1000
    )

    success = cluster.allocate_job(job3)
    if success:
        print_allocation_info(job3)


def test_large_cluster_allocation():
    cluster = create_cluster_from_config('testing_configs/large_config1.yaml')

    # Print initial cluster state
    print("=== Initial Cluster State ===")
    print(f"Total GPUs available: {cluster.get_available_gpu_count()}")
    for rack_id, count in cluster.get_gpu_availability_by_rack().items():
        print(f"Rack {rack_id}: {count} GPUs available")

    # List of test jobs with different resource requirements
    test_jobs = [
        # Single machine jobs
        Job(
            id="single_machine_small",
            required_gpus=4,
            gpu_memory_per_gpu=16000,
            cpu_memory_total=32000,
            total_episodes=1000
        ),
        Job(
            id="single_machine_large",
            required_gpus=8,
            gpu_memory_per_gpu=32000,
            cpu_memory_total=64000,
            total_episodes=2000
        ),
        # Same rack jobs
        Job(
            id="same_rack_medium",
            required_gpus=12,
            gpu_memory_per_gpu=16000,
            cpu_memory_total=96000,
            total_episodes=1500
        ),
        Job(
            id="same_rack_large",
            required_gpus=16,
            gpu_memory_per_gpu=32000,
            cpu_memory_total=128000,
            total_episodes=3000
        ),
        # Cross rack jobs
        Job(
            id="cross_rack_medium",
            required_gpus=20,
            gpu_memory_per_gpu=16000,
            cpu_memory_total=160000,
            total_episodes=2000
        ),
        Job(
            id="cross_rack_large",
            required_gpus=32,
            gpu_memory_per_gpu=32000,
            cpu_memory_total=256000,
            total_episodes=4000
        ),
        # Very large job that should use multiple racks
        Job(
            id="multi_rack_huge",
            required_gpus=48,
            gpu_memory_per_gpu=64000,
            cpu_memory_total=384000,
            total_episodes=5000
        )
    ]

    # Try to allocate all jobs
    for job in test_jobs:
        print(f"\n=== Attempting to allocate {job.id} ===")
        print(f"Required GPUs: {job.required_gpus}")
        print(f"Required GPU Memory per GPU: {job.gpu_memory_per_gpu}MB")
        print(f"Required CPU Memory: {job.cpu_memory_total}MB")

        success = cluster.allocate_job(job)

        if success:
            print("\nAllocation successful!")
            print_allocation_info(job)
            print("\nRemaining cluster resources:")
            print(f"Total GPUs available: {cluster.get_available_gpu_count()}")
            for rack_id, count in cluster.get_gpu_availability_by_rack().items():
                print(f"Rack {rack_id}: {count} GPUs available")
        else:
            print("\nAllocation failed!")
            print("Available resources:")
            print(f"Total GPUs available: {cluster.get_available_gpu_count()}")
            for rack_id, count in cluster.get_gpu_availability_by_rack().items():
                print(f"Rack {rack_id}: {count} GPUs available")


if __name__ == '__main__':
    test_large_cluster_allocation()