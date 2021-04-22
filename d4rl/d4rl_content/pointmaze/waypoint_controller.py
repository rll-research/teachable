import numpy as np
from d4rl_content.pointmaze import q_iteration
from d4rl_content.pointmaze.gridcraft import grid_env
from d4rl_content.pointmaze.gridcraft import grid_spec
from d4rl_content.pointmaze.gridcraft.grid_spec import WALL

ZEROS = np.zeros((2,), dtype=np.float32)
ONES = np.zeros((2,), dtype=np.float32)


class WaypointController(object):
    def __init__(self, maze_str, solve_thresh=0.1, p_gain=10.0, d_gain=-1.0, offset_mapping=np.array([0, 0])):
        self.maze_str = maze_str
        self._target = -1000 * ONES

        self.p_gain = p_gain
        self.d_gain = d_gain
        self.offset_mapping = offset_mapping
        self.solve_thresh = solve_thresh
        self.vel_thresh = 0.1

        self._waypoint_idx = 0
        self.waypoints = []
        self._waypoint_prev_loc = ZEROS
        if type(maze_str) is str:
            self.env = grid_env.GridEnv(grid_spec.spec_from_string(maze_str))
        elif type(maze_str) is list:
            self.env = grid_env.GridEnv(grid_spec.spec_from_array(maze_str))
        else:
            raise NotImplementedError(f'Unexpected maze str type {type(maze_str)}')

    def current_waypoint(self):
        return self.waypoints[self._waypoint_idx]

    def get_action(self, location, velocity, target):
        self.new_target(location, target)

        dist = np.linalg.norm(location - self._target)
        vel = self._waypoint_prev_loc - location
        vel_norm = np.linalg.norm(vel)
        task_not_solved = (dist >= self.solve_thresh) or (vel_norm >= self.vel_thresh)

        if task_not_solved and self._waypoint_idx < len(self.waypoints):
            next_wpnt = self.waypoints[self._waypoint_idx]
        else:
            next_wpnt = self._target

        # Compute control
        prop = next_wpnt - location
        action = self.p_gain * prop + self.d_gain * velocity

        dist_next_wpnt = np.linalg.norm(location - next_wpnt)
        if task_not_solved and (dist_next_wpnt < self.solve_thresh) and (vel_norm < self.vel_thresh):
            self._waypoint_idx += 1
            if self._waypoint_idx == len(self.waypoints)-1:
                assert np.linalg.norm(self.waypoints[self._waypoint_idx] - self._target) <= self.solve_thresh

        self._waypoint_prev_loc = location
        action = np.clip(action, -1.0, 1.0)
        return action, (not task_not_solved)

    def gridify_state(self, state):
        return (int(round(state[0])), int(round(state[1])))

    def new_target(self, start, target):
        raw_start = start
        start = self.gridify_state(start)
        raw_target = target
        target = self.gridify_state(target)

        # Get states
        grid = self.env.gs
        waypoints = self._breadth_first_search(np.array(start), np.array(target) + 1, grid)
        # Replace end waypoint with the true goal
        waypoints = waypoints[:-1] + [raw_target]
        self.waypoints = waypoints
        self._waypoint_prev_loc = start
        self._target = raw_target


    def _breadth_first_search(self, initial_state, goal, grid):

        # Add the offset mapping, which changes from world coordinates to grid coordinates
        initial_state = initial_state + self.offset_mapping
        goal = goal + self.offset_mapping

        queue = [(initial_state, None)]
        previous_pos = dict()

        while True:
            state, prev_pos = queue[0]
            queue = queue[1:]
            i, j = state

            if (i, j) in previous_pos:
                continue

            previous_pos[(i, j)] = prev_pos

            # If we reached a position satisfying the acceptance condition
            if np.array_equal(np.array([i, j]), goal):
                path = []
                pos = (i, j)
                # Subtract the offset mapping to change back into world coordinates
                while pos:
                    path.append(np.array(pos) - self.offset_mapping)
                    pos = previous_pos[pos]
                return path[::-1]

            if grid[i, j] == WALL:
                continue

            # Adjacent cells; prioritized by dist to goal
            adj_cells = []
            for k, l in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                next_pos = (i + k, j + l)
                adj_cells.append(next_pos)
            adj_cells.sort(key=lambda x: np.linalg.norm(np.array(x) - goal))
            queue += [(next_pos, (i, j)) for next_pos in adj_cells]


if __name__ == "__main__":
    print(q_iteration.__file__)
    TEST_MAZE = \
            "######\\"+\
            "#OOOO#\\"+\
            "#O##O#\\"+\
            "#OOOO#\\"+\
            "######"
    controller = WaypointController(TEST_MAZE)
    start = np.array((0, 3), dtype=np.float32)
    target = np.array((4,3), dtype=np.float32)
    act, done = controller.get_action(start, np.array([0,0]), target)
    for i in range(40):
        start = np.array((1 + i / 10, 1), dtype=np.float32)
        target = np.array((4, 3), dtype=np.float32)
        act, done = controller.get_action(start, np.array([0, 0]), target)
        print("WAYPOINT", start, controller.current_waypoint())


    # print('wpt:', controller.waypoints)
    # controller = WaypointController(TEST_MAZE)
    # start = np.array((2, 1), dtype=np.float32)
    # target = np.array((4, 3), dtype=np.float32)
    # act, done = controller.get_action(start, np.array([0, 0]), target)
    # print('wpt:', controller.waypoints)
    # print(act, done)
    import pdb; pdb.set_trace()
    pass
