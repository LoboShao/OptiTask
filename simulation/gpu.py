from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict

from simulation.gpu_model import GPUModel, GPU_MODELS

class GPUStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"



@dataclass
class GPU:
    id: str
    model: GPUModel
    status: GPUStatus = GPUStatus.IDLE
    current_job: Optional[str] = None

    @property
    def memory(self) -> int:
        """Return memory in GB"""
        return self.model.memory

    def allocate(self, job_id: str) -> bool:
        if self.status == GPUStatus.IDLE:
            self.status = GPUStatus.BUSY
            self.current_job = job_id
            return True
        return False

    def release(self):
        self.status = GPUStatus.IDLE
        self.current_job = None


if __name__ == '__main__':
    # Creating instances
    gpu1 = GPU(id="gpu_1", model=GPU_MODELS['T4'])
    gpu2 = GPU(id="gpu_2",model=GPU_MODELS['T4'])

    # Automatic string representation
    print(gpu1)  # GPU(id='gpu_1', memory=32, status=<GPUStatus.IDLE>, current_job=None)

    # Automatic comparison
    print(gpu1 == gpu2)  # False

    # Accessing attributes
    print(gpu1.memory)  # 32