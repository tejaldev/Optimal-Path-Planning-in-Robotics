from abs_base_classes import StateSpace, ActionSpace, StateTransition

class WarehouseStateSpace(StateSpace):
    def __init__(self, X_max, Y_max, obstacles):
        self.X_max = X_max
        self.Y_max = Y_max
        self.obstacles = obstacles
    
    def __contains__(self, x):
        return (0 <= x[0] <= self.X_max and 0 <= x[1] <= self.Y_max and x not in self.obstacles)
    
    def get_distance_lower_bound(self, x1, x2):
        return abs(x1[0] - x2[0]) + abs(x1[1] - x2[1])

class WarehouseActionSpace(ActionSpace):
    def __call__(self, x):
        return ["Up", "Down", "Left", "Right"]

class WarehouseStateTransition(StateTransition):
    def __call__(self, x, u):
        if u == "Up":
            return (x[0], x[1] + 1)
        elif u == "Down":
            return (x[0], x[1] - 1)
        elif u == "Left":
            return (x[0] - 1, x[1])
        elif u == "Right":
            return (x[0] + 1, x[1])
