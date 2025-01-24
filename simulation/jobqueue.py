from typing import Optional
from queue import PriorityQueue

from simulation.job import Job

class JobQueue:
    def __init__(self):
        self.queue = PriorityQueue()

    def add_job(self, job: Job):
        self.queue.put(job)

    def get_next_job(self) -> Optional[Job]:
        if not self.queue.empty():
            return self.queue.get()
        return None

    def peek(self) -> Optional[Job]:
        """Look at next job without removing it"""
        if not self.queue.empty():
            return self.queue.queue[0]
        return None