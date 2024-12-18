from abc import ABC, abstractmethod

class StateSpace(ABC):
    """A base class to specify a state space X"""

    @abstractmethod
    def __contains__(self, x) -> bool:
        """Return whether the given state x is in the state space"""
        pass

    def get_distance_lower_bound(self, x1, x2) -> float:
        """Return the lower bound on the distance
        between the given states x1 and x2
        """
        return 0

class ActionSpace(ABC):
    """A base class to specify an action space"""

    @abstractmethod
    def __call__(self, x) -> list:
        """Return the list of all the possible actions
        at the given state x
        """
        pass

class StateTransition(ABC):
    """A base class to specify a state transition function"""

    @abstractmethod
    def __call__(self, x, u):
        """Return the new state obtained by applying action u at state x"""
        pass