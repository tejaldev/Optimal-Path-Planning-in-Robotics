from derived import WarehouseStateSpace, WarehouseActionSpace, WarehouseStateTransition
from fsearch import fsearch

X_max = 4
Y_max = 4
obstacles = [(1,3), (2,3), (1,2), (1,1), (2,1), (3,1)]

X = WarehouseStateSpace(X_max, Y_max, obstacles)

xI = (0,0)
XG = [(4,4)]
U = WarehouseActionSpace()
f = WarehouseStateTransition()
alg1 = "bfs"
alg2 = "astar"

print("BFS:")
result1 = fsearch(X, U, f, xI, XG, alg1)
print(result1)

print("A* Search:")
result2 = fsearch(X, U, f, xI, XG, alg2)
print(result2)

print("Minimizes number of moves:")

if len(result1["visited"]) <= len(result2["visited"]):
    print("BFS WINS : ", result1)
else:
    print("A* WINS : ", result2)

