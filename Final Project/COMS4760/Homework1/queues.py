from abc import ABC, abstractmethod
from abs_base_classes import StateSpace
from queue import PriorityQueue
from typing import List, Dict, Any

class Queue(ABC):
    def __init__(self):
        self.elements = []
        self.parents = {}
    
    @abstractmethod
    def insert(self, x, parent):
        pass
    
    def pop(self):
        return self.elements.pop(0)
    
    def __len__(self):
        return len(self.elements)

class QueueBFS(Queue):
    def insert(self, x, parent):
        self.elements.append(x)
        self.parents[x] = parent

class QueueAstar(Queue):
    def __init__(self, X: StateSpace, XG: List):
        super().__init__()
        self.X = X
        self.XG = XG
        self.pq = PriorityQueue()
        self.costs = {}
    
    def insert(self, x, parent):
        if x not in self.costs or self.costs[x] > self.costs[parent] + 1:
            self.costs[x] = self.costs.get(parent, 0) + 1
            priority = self.costs[x] + min(self.X.get_distance_lower_bound(x, g) for g in self.XG)
            self.pq.put((priority, x))
            self.parents[x] = parent
    
    def pop(self):
        return self.pq.get()[1]
    
    def __len__(self):
        return self.pq.qsize()

def get_queue(alg: str, X: StateSpace, XG: List) -> Queue:
    if alg == "bfs":
        return QueueBFS()
    elif alg == "astar":
        return QueueAstar(X, XG)
    else:
        raise ValueError("Invalid algorithm specified")