import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import argparse

# ======================================
# Phase 1: Distance-Only RRT*
# ======================================

parser = argparse.ArgumentParser(description='Distance-Only RRT* + Battery Cost Calculation')

# Add command line arguments
parser.add_argument('--max_iterations', type=int, default=10000,
                    help='Maximum number of iterations for RRT* (default: 10000)')
parser.add_argument('--goal_threshold', type=float, default=0.5,
                    help='Distance threshold to consider goal reached (default: 0.5)')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility (default: 42)')
parser.add_argument('--terrain_type', type=str, default='random',
                    choices=['random', 'type1', 'all T1', 'all T3', 'crater'],
                    help='Preset terrain type (default: random)')
parser.add_argument('--start_x', type=float, default=0.5,
                    help='Start position x-coordinate (default: 0.5)')
parser.add_argument('--start_y', type=float, default=0.5,
                    help='Start position y-coordinate (default: 0.5)')
parser.add_argument('--goal_x', type=float, default=9.5,
                    help='Goal position x-coordinate (default: 9.5)')
parser.add_argument('--goal_y', type=float, default=9.5,
                    help='Goal position y-coordinate (default: 9.5)')

args = parser.parse_args()

np.random.seed(args.seed)
max_iterations = args.max_iterations
goal_threshold = args.goal_threshold
choice = args.terrain_type
start_pos = (args.start_x, args.start_y)
goal_pos = (args.goal_x, args.goal_y)

X_MIN, X_MAX = 0, 10
Y_MIN, Y_MAX = 0, 10
ROBOT_RADIUS = 0.25
B_MAX = 100  # Not used in Phase 1, but used in battery cost calc
b_factor = 1.0
NEAR_RADIUS = 1.0

def cost_per_unit(terrain_label):
    """Return the battery cost factor for a given terrain label."""
    if terrain_label == '1':
        return 0
    elif terrain_label == '2':
        return 2 * b_factor
    elif terrain_label == '3':
        return 3 * b_factor
    else:
        raise ValueError(f"Invalid terrain label: {terrain_label}")

# ======================================
# Generate Terrain Grid & Obstacles
# ======================================
fig, ax = plt.subplots(figsize=(10, 10))

values = np.random.choice(['3','1','2'], size=(10, 10))  # random default
if choice == 'type1':
    values = np.array([
        ['1','1','1','1','1','1','1','1','1','1'],
        ['1','1','1','1','1','1','1','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1'],
        ['2','2','2','2','2','2','2','1','1','1']
    ])
elif choice == 'all T1':
    values = np.array([['1']*10]*10)
elif choice == 'all T3':
    values = np.array([['3']*10]*10)
elif choice == 'crater':
    values = np.array([
        ['1','1','1','1','1','1','1','1','1','1'],
        ['1','1','1','1','1','1','1','1','1','1'],
        ['1','1','1','1','1','1','1','1','1','1'],
        ['1','1','1','1','1','1','1','1','1','1'],
        ['1','1','1','1','1','1','1','1','1','1'],
        ['1','1','1','1','1','2','2','2','2','2'],
        ['1','1','1','1','1','2','3','3','3','3'],
        ['1','1','1','1','1','2','3','3','3','3'],
        ['1','1','1','1','1','2','3','3','3','3'],
        ['1','1','1','1','1','2','3','3','3','3'],
    ])

# draw grid
for i in range(11):
    ax.axhline(y=i, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=i, color='gray', linestyle='-', alpha=0.5)

# label terrain
for i in range(10):
    for j in range(10):
        ax.text(j + 0.5, i + 0.5, values[i, j], 
                horizontalalignment='center', 
                verticalalignment='center',
                color='red', fontsize=12, fontweight='bold')

# random obstacles
num_obstacles = 7
obstacles = []
for _ in range(num_obstacles):
    x = np.random.uniform(0, 10)
    y = np.random.uniform(0, 10)
    radius = np.random.choice([0.5, 1.0])
    obstacles.append((x, y, radius))
    circle = Circle((x, y), radius, fill=True, alpha=0.2,
                    color='red', edgecolor='red', linewidth=2)
    ax.add_patch(circle)

# draw start & goal
start_node_circle = Circle(start_pos, 0.25, fill=True, color='red')
goal_node_circle  = Circle(goal_pos,  0.25, fill=True, color='green')
ax.add_patch(start_node_circle)
ax.add_patch(goal_node_circle)

ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_aspect('equal')
ax.set_title('Terrain & Obstacles')
plt.grid(True)
plt.show()

# ======================================
# Utility Functions
# ======================================
def in_bounds(x, y):
    return (0.25 <= x <= 9.75) and (0.25 <= y <= 9.75)

def check_collision(x, y):
    """Check whether point (x,y) collides with any obstacle."""
    for (ox, oy, r) in obstacles:
        dist = math.sqrt((x - ox)**2 + (y - oy)**2)
        if dist <= (r + ROBOT_RADIUS):
            return True
    return False

def terrain_category(x, y):
    """Return the terrain label ('1','2','3') for location (x,y)."""
    i = int(y)
    j = int(x)
    i = max(0, min(i, 9))
    j = max(0, min(j, 9))
    return values[i, j]

def energy_cost(x1, y1, x2, y2):
    """
    Battery cost (energy) of moving from (x1,y1) to (x2,y2).
    Takes average terrain cost between the two terrain patches 
    and multiplies by Euclidean distance.
    """
    h1 = terrain_category(x1, y1)
    h2 = terrain_category(x2, y2)
    c_mid = 0.5*(cost_per_unit(h1) + cost_per_unit(h2))
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return c_mid * dist

def distance_only_cost(x1, y1, x2, y2):
    """Pure Euclidean distance cost."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# ======================================
# Phase 1: Build a Distance-Only RRT*
# ======================================
class DistanceNode:
    """Node class for distance-based RRT* (no battery constraints)."""
    def __init__(self, x, y, theta, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.cost = cost  # total distance from start node

def distance(n1, n2):
    return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

def sample_random_state():
    x = np.random.uniform(0.25, 9.75)
    y = np.random.uniform(0.25, 9.75)
    theta = np.random.uniform(0, 2*np.pi)
    return x, y, theta

def nearest_node(nodes, x, y, theta):
    dummy = DistanceNode(x, y, theta)
    dists = [distance(dummy, n) for n in nodes]
    idx = np.argmin(dists)
    return nodes[idx]

def feasible_distance(x, y):
    """Checks feasibility ignoring battery, only collision + in-bounds."""
    if not in_bounds(x, y):
        return False
    if check_collision(x, y):
        return False
    return True

def extend_distance(node_from, x_to, y_to, theta_to, step_size=0.3):
    """
    Extend using pure distance cost. 
    Battery constraints are ignored here.
    """
    dx = x_to - node_from.x
    dy = y_to - node_from.y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1e-9:
        return None

    # determine how many steps
    if dist < step_size:
        step_size = dist
    steps = int(math.ceil(dist / step_size))

    x_cur, y_cur = node_from.x, node_from.y
    total_incremental_cost = 0.0

    for _ in range(steps):
        angle = math.atan2(dy, dx)
        x_next = x_cur + step_size * math.cos(angle)
        y_next = y_cur + step_size * math.sin(angle)
        if not feasible_distance(x_next, y_next):
            return None
        # accumulate distance cost
        cost_step = distance_only_cost(x_cur, y_cur, x_next, y_next)
        x_cur, y_cur = x_next, y_next
        total_incremental_cost += cost_step
    
    new_cost = node_from.cost + total_incremental_cost
    new_node = DistanceNode(x_cur, y_cur, angle, parent=node_from, cost=new_cost)
    return new_node

def get_near_nodes(nodes, new_node, radius=NEAR_RADIUS):
    near_nodes = []
    for nd in nodes:
        if distance(nd, new_node) <= radius:
            near_nodes.append(nd)
    return near_nodes

def check_path_feasibility_distance(n_from, x_to, y_to, step_size=0.1):
    """Check collision feasibility for the path from n_from to (x_to,y_to)."""
    dx = x_to - n_from.x
    dy = y_to - n_from.y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < 1e-9:
        return None

    if dist < step_size:
        step_size = dist
    steps = int(math.ceil(dist / step_size))
    x_cur, y_cur = n_from.x, n_from.y
    total_cost_inc = 0.0

    for _ in range(steps):
        angle = math.atan2(dy, dx)
        x_next = x_cur + step_size * math.cos(angle)
        y_next = y_cur + step_size * math.sin(angle)
        if not feasible_distance(x_next, y_next):
            return None
        cost_step = distance_only_cost(x_cur, y_cur, x_next, y_next)
        x_cur, y_cur = x_next, y_next
        total_cost_inc += cost_step
    
    final_cost = n_from.cost + total_cost_inc
    final_angle = math.atan2(y_cur - n_from.y, x_cur - n_from.x)
    return (x_cur, y_cur, final_angle, final_cost)

def rewire_distance(nodes, new_node):
    """RRT* rewire step using distance-only cost."""
    near_nodes = get_near_nodes(nodes, new_node, radius=NEAR_RADIUS)
    for nnear in near_nodes:
        feasible_check = check_path_feasibility_distance(new_node, nnear.x, nnear.y)
        if feasible_check is not None:
            x_cur, y_cur, theta_cur, new_cost = feasible_check
            if new_cost < nnear.cost:
                nnear.parent = new_node
                nnear.cost = new_cost
                nnear.x, nnear.y, nnear.theta = x_cur, y_cur, theta_cur

# Build the distance-only tree
distance_nodes = []
start_distance_node = DistanceNode(start_pos[0], start_pos[1], 0.0, parent=None, cost=0.0)
distance_nodes.append(start_distance_node)

best_distance = float('inf')
best_dist_node = None

for i in range(max_iterations):
    x_rand, y_rand, theta_rand = sample_random_state()
    n_nearest = nearest_node(distance_nodes, x_rand, y_rand, theta_rand)
    new_node = extend_distance(n_nearest, x_rand, y_rand, theta_rand)
    if new_node is not None:
        distance_nodes.append(new_node)
        rewire_distance(distance_nodes, new_node)
        
        # check goal
        d_goal = math.sqrt((new_node.x - goal_pos[0])**2 + (new_node.y - goal_pos[1])**2)
        if d_goal < goal_threshold and new_node.cost < best_distance:
            best_distance = new_node.cost
            best_dist_node = new_node

if best_dist_node:
    print(f"[PHASE 1] Goal reached (distance-based) with total distance: {best_distance:.4f}")
else:
    print("[PHASE 1] Goal not reached with distance-only RRT*")

# ======================================
# Visualize the Distance-Only RRT* Tree
# ======================================
fig2, ax2 = plt.subplots(figsize=(10,10))

for i in range(11):
    ax2.axhline(y=i, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(x=i, color='gray', linestyle='-', alpha=0.5)

for i in range(10):
    for j in range(10):
        ax2.text(j + 0.5, i + 0.5, values[i, j], 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 color='red', fontsize=12, fontweight='bold')

for (ox, oy, r) in obstacles:
    circle = Circle((ox, oy), r, fill=True, alpha=0.2, color='red', edgecolor='red', linewidth=2)
    ax2.add_patch(circle)

start_circle = Circle(start_pos, 0.25, fill=True, color='red')
goal_circle  = Circle(goal_pos,  0.25, fill=True, color='green')
ax2.add_patch(start_circle)
ax2.add_patch(goal_circle)

# draw the RRT* tree
for nd in distance_nodes:
    if nd.parent is not None:
        ax2.plot([nd.x, nd.parent.x], [nd.y, nd.parent.y], '-k', alpha=0.5)

# highlight best path in blue
if best_dist_node:
    path = []
    cur = best_dist_node
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path = path[::-1]
    for i in range(len(path)-1):
        ax2.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], '-b', linewidth=2.5)

ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
ax2.set_aspect('equal')
ax2.set_title('Distance-Only RRT* Tree & Path')
plt.grid(True)
plt.show()

# ======================================
# Phase 2: Compute Battery Usage of That Path
# ======================================
if best_dist_node:
    # Reconstruct the distance-only path
    path_coords = []
    cur = best_dist_node
    while cur is not None:
        path_coords.append((cur.x, cur.y))
        cur = cur.parent
    path_coords = path_coords[::-1]  # from start to goal

    # Now compute the battery usage along that path using your energy_cost() function
    total_battery_cost = 0.0
    for i in range(len(path_coords)-1):
        x1, y1 = path_coords[i]
        x2, y2 = path_coords[i+1]
        total_battery_cost += energy_cost(x1, y1, x2, y2)

    print(f"[PHASE 2] Battery cost for the distance-based path = {total_battery_cost:.4f}")
else:
    print("[PHASE 2] No path found to compute battery usage.")
