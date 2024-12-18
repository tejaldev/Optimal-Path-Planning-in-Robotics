import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='RRT* Path Planning with Battery Constraints')

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

# Parse arguments
args = parser.parse_args()

# Use the parsed arguments
np.random.seed(args.seed)
max_iterations = args.max_iterations
goal_threshold = args.goal_threshold
choice = args.terrain_type
start_pos = (args.start_x, args.start_y)
goal_pos = (args.goal_x, args.goal_y)

# ========================
# Parameters & Constants
# ========================
X_MIN, X_MAX = 0, 10
Y_MIN, Y_MAX = 0, 10
ROBOT_RADIUS = 0.25
B_MAX = 100
b_factor = 1.0  # Baseline energy factor

# For rewiring, define a search radius
NEAR_RADIUS = 1.0  # You can adjust this based on environment size, number of nodes, etc.

# Define transitions for the energy cost function

def cost_per_unit(terrain_label):
    if terrain_label == '1':
        return 0  # e.g. b
    elif terrain_label == '2':
        return 2*b_factor
    elif terrain_label == '3':
        return 3*b_factor
    else:
        # default or error case
        raise ValueError(f"Invalid terrain label: {terrain_label}")

# ========================
# Generate Terrain & Obstacles
# ========================

fig, ax = plt.subplots(figsize=(10, 10))
# MAKE THIS PSEUDO RANDOM, USER SHOULD BE ABLE TO OVERWRITE THE TERRAIN
# choice = 'all uphill'
choice = 'random'
# choice = 'all downhill'
#values = np.random.choice(['+', '-', '0'], size=(10, 10))
values = np.random.choice(['3', '1', '2'], size=(10, 10))
if choice == 'type1':
    values = np.array([['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1'],
                       ['2', '2', '2', '2', '2', '2', '2', '1', '1', '1']])
elif choice == 'all T1':
    values = np.array([['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1']])
elif choice == 'all T3':
    values = np.array([['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3'],
                       ['3', '3', '3', '3', '3', '3', '3', '3', '3', '3']])
elif choice == 'crater':
    values = np.array([['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '1', '1', '1', '1', '1'],
                       ['1', '1', '1', '1', '1', '2', '2', '2', '2', '2'],
                       ['1', '1', '1', '1', '1', '2', '3', '3', '3', '3'],
                       ['1', '1', '1', '1', '1', '2', '3', '3', '3', '3'],
                       ['1', '1', '1', '1', '1', '2', '3', '3', '3', '3'],
                       ['1', '1', '1', '1', '1', '2', '3', '3', '3', '3']])

for i in range(11):
    ax.axhline(y=i, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(x=i, color='gray', linestyle='-', alpha=0.5)

for i in range(10):
    for j in range(10):
        ax.text(j + 0.5, i + 0.5, values[i, j], 
                horizontalalignment='center', 
                verticalalignment='center',
                color='red',
                fontsize=12,
                fontweight='bold')

num_obstacles = 7
obstacles = []
for _ in range(num_obstacles):
    x = np.random.uniform(0, 10)
    y = np.random.uniform(0, 10)
    radius = np.random.choice([0.5, 1.0])
    obstacles.append((x, y, radius))
    circle = Circle((x, y), radius, fill=True, alpha=0.2, color='red', edgecolor='red', linewidth=2)
    ax.add_patch(circle)

# Start and Goal
#start_pos = (0.5, 0.5)
#goal_pos = (9.5, 9.5)
start_node_circle = Circle(start_pos, 0.25, fill=True, color='red')
goal_node_circle = Circle(goal_pos, 0.25, fill=True, color='green')
ax.add_patch(start_node_circle)
ax.add_patch(goal_node_circle)

ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title('10x10 Configuration Space')
plt.grid(True)
plt.show()

# ========================
# Utility Functions
# ========================
def in_bounds(x, y):
    return (0.25 <= x <= 9.75) and (0.25 <= y <= 9.75)

def check_collision(x, y):
    for (ox, oy, r) in obstacles:
        dist = math.sqrt((x - ox)**2 + (y - oy)**2)
        if dist <= (r + ROBOT_RADIUS):
            return True
    return False

def terrain_category(x, y):
    i = int(y)
    j = int(x)
    if i < 0: i = 0
    if i > 9: i = 9
    if j < 0: j = 0
    if j > 9: j = 9
    return values[i, j]

def energy_cost(x1, y1, x2, y2):
    # terrain label at end
    # h2 = terrain_category(x2, y2)  # returns '1','2','3'
    # c = cost_per_unit(h2)
    # dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    # return c * dist
    h1 = terrain_category(x1, y1)
    h2 = terrain_category(x2, y2)
    c_mid = 0.5 * (cost_per_unit(h1) + cost_per_unit(h2)) # taking average of the two terrain categories
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return c_mid * dist

def feasible(x, y, b):
    if not in_bounds(x, y):
        return False
    if check_collision(x, y):
        return False
    if b <= 0:
        return False
    return True

# Node class for RRT*
class Node:
    def __init__(self, x, y, theta, b, parent=None, cost=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.b = b
        self.parent = parent
        self.cost = cost  # total battery used from start node

def distance(n1, n2):
    return math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)

def sample_random_state():
    x = np.random.uniform(0.25, 9.75)
    y = np.random.uniform(0.25, 9.75)
    theta = np.random.uniform(0, 2*np.pi)
    return x, y, theta

def nearest_node(nodes, x, y, theta):
    dummy = Node(x,y,theta,B_MAX)
    dists = [distance(dummy, n) for n in nodes]
    idx = np.argmin(dists)
    return nodes[idx]

def path_cost(parent_node, x_new, y_new):
    # Compute the incremental cost from parent_node to new point
    # We discretize to a small step if needed
    # For simplicity, just a single step:
    cost_step = energy_cost(parent_node.x, parent_node.y, x_new, y_new)
    return cost_step

def distance_only_cost(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def extend(node_from, x_to, y_to, theta_to, step_size=0.3):
    dx = x_to - node_from.x
    dy = y_to - node_from.y
    dist = math.sqrt(dx*dx + dy*dy)
    if dist < step_size:
        step_size = dist
    steps = int(math.ceil(dist / step_size))
    x_cur, y_cur, theta_cur, b_cur = node_from.x, node_from.y, node_from.theta, node_from.b
    total_incremental_cost = 0.0
    
    for _ in range(steps):
        angle = math.atan2(dy, dx)
        x_next = x_cur + step_size * math.cos(angle)
        y_next = y_cur + step_size * math.sin(angle)
        cost_step = energy_cost(x_cur, y_cur, x_next, y_next)
        b_next = b_cur - cost_step
        if not feasible(x_next, y_next, b_next):
            return None
        # update
        x_cur, y_cur, theta_cur, b_cur = x_next, y_next, angle, b_next
        total_incremental_cost += cost_step

    new_cost = node_from.cost + total_incremental_cost
    return Node(x_cur, y_cur, theta_cur, b_cur, parent=node_from, cost=new_cost)


def get_near_nodes(nodes, new_node, radius=NEAR_RADIUS):
    near_nodes = []
    for n in nodes:
        if distance(n, new_node) <= radius:
            near_nodes.append(n)
    return near_nodes


def check_path_feasibility(n_from, x_to, y_to, step_size=0.1):
    # Check feasibility without actually creating a node first
    dx = x_to - n_from.x
    dy = y_to - n_from.y
    dist = math.sqrt(dx*dx + dy*dy)
    
    # If points are effectively the same, return None
    if dist < 1e-10:
        return None
        
    if dist < step_size:
        step_size = dist
    steps = int(math.ceil(dist / step_size))
    x_cur, y_cur, b_cur = n_from.x, n_from.y, n_from.b
    
    total_cost_increment = 0.0
    
    for _ in range(steps):
        angle = math.atan2(dy, dx)
        x_next = x_cur + step_size * math.cos(angle)
        y_next = y_cur + step_size * math.sin(angle)
        
        # Compute step cost
        cost_step = energy_cost(x_cur, y_cur, x_next, y_next)
        
        b_next = b_cur - cost_step
        if not feasible(x_next, y_next, b_next):
            return None
        
        # Accumulate cost and update state
        total_cost_increment += cost_step
        x_cur, y_cur, b_cur = x_next, y_next, b_next
    
    # final_cost should add the newly used energy to n_from.cost
    # Since total_cost_increment = (n_from.b - b_cur),
    # we have final_cost = n_from.cost + total_cost_increment
    final_cost = n_from.cost + total_cost_increment
    
    # Also compute the final angle (optional if needed)
    final_angle = math.atan2(y_cur - n_from.y, x_cur - n_from.x)
    
    return (x_cur, y_cur, final_angle, b_cur, final_cost)


# def adaptive_radius(num_nodes):
#     # Example: gamma = 2.0 (tweak as needed)
#     gamma = 2.0
#     d = 2  # dimension: 2D (x,y)
#     r = gamma * ((math.log(num_nodes) / num_nodes) ** (1.0 / d))
#     return r

def rewire(nodes, new_node):
    near_nodes = get_near_nodes(nodes, new_node, radius=NEAR_RADIUS)
    # radius = adaptive_radius(len(nodes))
    # near_nodes = get_near_nodes(nodes, new_node, radius=radius)
    for nnear in near_nodes:
        # Check if going from new_node to nnear improves cost
        feasible_check = check_path_feasibility(new_node, nnear.x, nnear.y)
        if feasible_check is not None:
            x_cur, y_cur, theta_cur, b_cur, new_cost = feasible_check
            # if we got a feasible path and cost is lower:
            if new_cost < nnear.cost:
                # Update nnear's parent and cost
                nnear.parent = new_node
                nnear.cost = new_cost
                nnear.x, nnear.y, nnear.theta, nnear.b = x_cur, y_cur, theta_cur, b_cur

# ========================
# Basic RRT* Loop
# ========================
nodes = []
start_node = Node(start_pos[0],start_pos[1],0.0,B_MAX,parent=None,cost=0.0)
nodes.append(start_node)

best_cost = float('inf')
best_node = None

for i in range(max_iterations):
    x_rand, y_rand, theta_rand = sample_random_state()
    n_nearest = nearest_node(nodes, x_rand, y_rand, theta_rand)
    new_node = extend(n_nearest, x_rand, y_rand, theta_rand)
    if new_node is not None:
        # Rewire
        nodes.append(new_node)
        rewire(nodes, new_node)
        
        # Check if close to goal
        d_goal = math.sqrt((new_node.x - goal_pos[0])**2 + (new_node.y - goal_pos[1])**2)
        if d_goal < goal_threshold and new_node.cost < best_cost:
            best_cost = new_node.cost
            best_node = new_node

if best_node:
    print(f"Goal reached with BATTERY cost: {best_cost}")
    # Reconstruct the path (from best_node back to start)

    path_coords = []
    cur = best_node
    while cur is not None:
        path_coords.append((cur.x, cur.y))
        cur = cur.parent
    path_coords.reverse()  # Now path_coords goes from start -> goal

    # Compute total Euclidean distance of the battery-optimal path
    total_distance = 0.0
    for i in range(len(path_coords) - 1):
        x1, y1 = path_coords[i]
        x2, y2 = path_coords[i+1]
        segment_dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        total_distance += segment_dist

    print(f"Battery-optimal path distance: {total_distance:.4f}")

else:
    print("Goal not reached.")


# ========================
# Visualizing the Result
# ========================
fig2, ax2 = plt.subplots(figsize=(10,10))

# draw grid
for i in range(11):
    ax2.axhline(y=i, color='gray', linestyle='-', alpha=0.5)
    ax2.axvline(x=i, color='gray', linestyle='-', alpha=0.5)

# Draw terrain categories
for i in range(10):
    for j in range(10):
        ax2.text(j + 0.5, i + 0.5, values[i, j], 
                 horizontalalignment='center', 
                 verticalalignment='center',
                 color='red',
                 fontsize=12,
                 fontweight='bold')

# Draw obstacles
for (ox, oy, r) in obstacles:
    circle = Circle((ox, oy), r, fill=True, alpha=0.2, color='red', edgecolor='red', linewidth=2)
    ax2.add_patch(circle)

# Draw start and goal
start_circle = Circle(start_pos, 0.25, fill=True, color='red')
goal_circle = Circle(goal_pos, 0.25, fill=True, color='green')
ax2.add_patch(start_circle)
ax2.add_patch(goal_circle)

# Draw the entire tree in black
for n in nodes:
    if n.parent is not None:
        ax2.plot([n.x, n.parent.x], [n.y, n.parent.y], '-k', alpha=0.5)

# If best path found, highlight it in blue
if best_node is not None:
    path = []
    cur = best_node
    while cur is not None:
        path.append(cur)
        cur = cur.parent
    path = path[::-1]  # from start to goal

    for i in range(len(path)-1):
        ax2.plot([path[i].x, path[i+1].x], [path[i].y, path[i+1].y], '-b', linewidth=2.5)

ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
ax2.set_aspect('equal')
ax2.set_title('RRT* Tree with Battery-Optimal Path')
plt.grid(True)
plt.show()
