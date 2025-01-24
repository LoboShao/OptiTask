from typing import List, Dict, Any
import yaml
from simulation.machine import Machine
from simulation.rack import Rack
from simulation.gpu import GPU, GPU_MODELS
from simulation.cluster import Cluster


def create_gpus_from_config(machine_id: str, gpu_configs: List[Dict[str, Any]]) -> List[GPU]:
    """Create a list of GPUs based on model configurations"""
    gpus = []
    gpu_index = 0

    for gpu_config in gpu_configs:
        model_name = gpu_config['model']
        count = gpu_config['count']

        if model_name not in GPU_MODELS:
            raise ValueError(f"Unknown GPU model: {model_name}")

        model = GPU_MODELS[model_name]

        for _ in range(count):
            gpu = GPU(
                id=f"{machine_id}_gpu_{gpu_index}",
                model=model
            )
            gpus.append(gpu)
            gpu_index += 1

    return gpus


def create_machine(machine_config: Dict[str, Any], rack_id: str) -> Machine:
    """Create a machine from configuration"""
    machine_id = machine_config['id']

    gpus = create_gpus_from_config(
        machine_id=machine_id,
        gpu_configs=machine_config['gpus']
    )

    return Machine(
        id=machine_id,
        gpus=gpus,
        rack_id=rack_id,
        total_cpu_memory=machine_config['cpu_memory'],
        used_cpu_memory=0
    )


def create_rack(rack_config: Dict[str, Any]) -> Rack:
    """Create a rack from configuration"""
    rack_id = rack_config['id']
    rack_name = rack_config['name']
    machines = [
        create_machine(machine_config, rack_id)
        for machine_config in rack_config['machines']
    ]
    return Rack(id=rack_id, name=rack_name, machines=machines)


def create_cluster_from_config(config_path: str) -> Cluster:
    """Create a cluster from a YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    cluster_config = config['cluster_config']
    racks = [create_rack(rack_config) for rack_config in cluster_config['racks']]

    return Cluster(racks)