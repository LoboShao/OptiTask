from dataclasses import dataclass
from platform import machine
from typing import List, Optional, Dict, Tuple

from simulation.machine import Machine
from simulation.gpu_model import GPU_MODELS
from simulation.gpu import GPU

@dataclass
class Rack:
    id: str
    name: str
    machines: List[Machine]


    def count_available_gpus_by_model(self, model_name: Optional[str] = None) -> int:

        return sum([machine.count_available_gpus_by_model(model_name=model_name) for machine in self.machines])

    def allocate_gpus(self, num_gpus: int, job_id: str, model_name: GPU_MODELS):
        selected_gpus = []
        rest = num_gpus
        machines_with_counts = [machine.count_available_gpus_by_model(model_name=model_name) for machine in self.machines]
        for idx, count in enumerate(machines_with_counts):
            if count > 0:
                if rest >= count:
                    selected_gpus.extend(self.machines[idx].allocate_gpus(num_gpus=count, job_id=job_id, model_name=model_name))
                    rest -= count
                else:
                    selected_gpus.extend(self.machines[idx].allocate_gpus(num_gpus=rest, job_id=job_id, model_name=model_name))
        return selected_gpus
