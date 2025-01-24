from dataclasses import dataclass
from typing import Dict


@dataclass
class GPUModel():
    name: str
    memory: int  # Memory in MB
    tensor_cores: int
    cuda_cores: int
    architecture: str
    bandwidth: float  # GB/s
    clock_speed_ghz: float  # Approximate GPU clock speed in GHz
    compute_efficiency: float  # Architecture-specific compute efficiency
    memory_efficiency: float  # Architecture-specific memory efficiency
    ops_per_tc: int  # Operations per tensor core per clock

    @property
    def effective_bandwidth(self) -> float:
        """Get effective memory bandwidth in GB/s"""
        return self.bandwidth * self.memory_efficiency

    @property
    def peak_ops(self) -> float:
        """Get peak operations per second considering architecture"""
        if self.architecture in ["Ampere", "Hopper"]:
            # Using tensor cores for ML workloads
            return (self.tensor_cores * self.ops_per_tc *
                    self.clock_speed_ghz * 1e9 * self.compute_efficiency)
        else:
            # Traditional CUDA cores
            return (self.cuda_cores * self.clock_speed_ghz * 1e9 * 2 *
                    self.compute_efficiency)


GPU_MODELS: Dict[str, GPUModel] = {
    "A100_80GB": GPUModel(
        name="A100_80GB",
        memory=80 * 1024,
        tensor_cores=432,
        cuda_cores=6912,
        architecture="Ampere",
        bandwidth=2039.0,
        clock_speed_ghz=1.41,
        compute_efficiency=0.6,
        memory_efficiency=0.8,
        ops_per_tc=256,
    ),
    "A100_40GB": GPUModel(
        name="A100_40GB",
        memory=40 * 1024,
        tensor_cores=432,
        cuda_cores=6912,
        architecture="Ampere",
        bandwidth=1555.0,
        clock_speed_ghz=1.37,
        compute_efficiency=0.6,
        memory_efficiency=0.8,
        ops_per_tc=256,
    ),
    "H100_80GB": GPUModel(
        name="H100_80GB",
        memory=80 * 1024,
        tensor_cores=528,
        cuda_cores=16896,
        architecture="Hopper",
        bandwidth=3350.0,
        clock_speed_ghz=1.45,
        compute_efficiency=0.7,
        memory_efficiency=0.85,
        ops_per_tc=256,
    ),
    "T4": GPUModel(
        name="T4",
        memory=16 * 1024,
        tensor_cores=320,
        cuda_cores=2560,
        architecture="Turing",
        bandwidth=320.0,
        clock_speed_ghz=1.59,
        compute_efficiency=0.5,
        memory_efficiency=0.7,
        ops_per_tc=128,  # Turing has different tensor core architecture
    ),
}