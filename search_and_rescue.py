import streamlit as st
import numpy as np
import heapq
import matplotlib.pyplot as plt

# --- CONFIG ---
GRID_SIZE = 10
NUM_SURVIVORS = 5
SENSOR_RANGE = 2

# --- ENVIRONMENT SETUP ---
np.random.seed(42)

def create_environment(size, num_survivors):
    grid = np.zeros((size, size))
    survivors = []
    for _ in range(num_survivors):
        x, y = np.random.randint(0, size, 2)
        while grid[x, y] != 0:
            x, y = np.random.randint(0, size, 2)
        grid[x, y] = 2  # Survivor
        survivors.append((x, y))
    return grid, survivors

# --- A* PATHFINDING ---
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def a_star(grid, start, goal):
    neighbors = [(0,1),(1,0),(-1,0),(0,-1)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        _, current = heapq.heappop(oheap)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        close_set.add(current)
        for dx, dy in neighbors:
            neighbor = (current[0]+dx, current[1]+dy)
            tentative_g_score = gscore[current] + 1
            if 0 <= neighbor[0] < grid.shape[0]:
                if 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0]][neighbor[1]] == 1:  # obstacle
                        continue
                else:
                    continue
            else:
                continue

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return []

# --- SENSING ---
def detect_survivors(position, grid):
    x, y = position
    found = []
    for i in range(x - SENSOR_RANGE, x + SENSOR_RANGE + 1):
        for j in range(y - SENSOR_RANGE, y + SENSOR_RANGE + 1):
            if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
                if grid[i, j] == 2:
                    found.append((i, j))
    return found

# --- STREAMLIT INTERFACE ---
def main():
    st.title("🦾 Search-and-Rescue Robot Simulator")
    grid, survivors = create_environment(GRID_SIZE, NUM_SURVIVORS)
    st.write(f"🔍 Total Survivors to Find: {NUM_SURVIVORS}")
    robot_position = (0, 0)
    found_survivors = []
    explored = set()
    path_log = []

    while len(found_survivors) < NUM_SURVIVORS:
        detected = detect_survivors(robot_position, grid)
        for d in detected:
            if d not in found_survivors:
                found_survivors.append(d)

        unexplored = [
            (i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE)
            if (i, j) not in explored
        ]

        if detected:
            goal = detected[0]
        else:
            goal = unexplored[np.random.randint(0, len(unexplored))]

        path = a_star(grid, robot_position, goal)
        if not path:
            st.warning("No path found. Environment may be blocked.")
            break

        for step in path:
            robot_position = step
            explored.add(step)
            path_log.append(step)
            detected = detect_survivors(robot_position, grid)
            for d in detected:
                if d not in found_survivors:
                    found_survivors.append(d)

    st.success("✅ All survivors rescued!")
    st.write(f"📍 Robot Path Length: {len(path_log)}")

    # --- PLOT ---
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')
    for (x, y) in path_log:
        ax.plot(y, x, "bo", markersize=3)
    for (x, y) in survivors:
        ax.plot(y, x, "rx", markersize=8)
    ax.plot(0, 0, "go", label="Start")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
