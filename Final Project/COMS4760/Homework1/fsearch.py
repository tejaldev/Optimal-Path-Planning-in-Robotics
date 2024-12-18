from queues import get_queue

"""FSEARCH (Forward Search) Implementation"""

def fsearch(X, U, f, xI, XG, alg):
    """
    Perform forward search on the state space X with the given action space U,
    state transition function f, initial state xI, goal state XG (can be empty), and algorithm alg (either "bfs" or "astar").
    Returns a dictionary with the following structure: {"visited": visited_states, "path": path}
    Where visited_states is a list of states visited during the search, in the order they were visited.
    and path is a list of states representing a path from xI to a state in XG (if one exists).
    """
    if XG is None:
        return {"visited": [], "path": []}
    
    Q = get_queue(alg, X, XG)
    Q.insert(xI, None)
    visited = []


    while len(Q) > 0:
        x = Q.pop()
        visited.append(x)
        
        if x in XG:
            path = []
            while x is not None:
                path.append(x)
                x = Q.parents[x]
            return {"visited": visited, "path": list(reversed(path))}
        
        for u in U(x):
            x_new = f(x, u)
            if x_new in X and x_new not in Q.parents:
                Q.insert(x_new, x)
    
    return {"visited": visited, "path": []}