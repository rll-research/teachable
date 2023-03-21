#   Pytorch probably has a way to do this
#   Want to think about if converter class should have a loading init or if the comparison class should have that

from algos import utils
from torch import nn
import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
from algos.utils import WaypointToDirectional as wtd

batch_size = 2000
num_batches = 5000
advice_interval = 5


def get_direction(offset_waypoint, velocity):
    action = 10 * offset_waypoint + -1 * velocity

    # Inverse tanh with clipping to match the output of the directional controller
    eps = 0.001
    # eps = 0
    action = np.clip(action, -1.0 + eps, 1.0 - eps)
    action = np.arctanh(action)
    return action


def breadth_first_search(initial_state, goal, grid):

    # Add the offset mapping, which changes from world coordinates to grid coordinates
    initial_state = initial_state.astype(np.int32)
    goal = goal.astype(np.int32)

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
                path.append(np.array(pos))
                pos = previous_pos[pos]
            return path[0]

        if grid[i, j] == 1:
            continue

        # Adjacent cells; prioritized by dist to goal
        adj_cells = []
        for k, l in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            next_pos = (i + k, j + l)
            adj_cells.append(next_pos)
        adj_cells.sort(key=lambda x: np.linalg.norm(np.array(x) - goal))
        queue += [(next_pos, (i, j)) for next_pos in adj_cells]


def generate_advice_batch(advice_embedding):
    # Get position, random.linear within bounds of max grid size (15x15)? Run render to find out)
    # Get velocity, random.linear from some reasonable value
    # Get waypoints. Should be able to based on max distance from starting point to waypoint without regard for actual environment
    #   because that's what the actual direction instruction is taken from
    #   Max distance based on gridify state is just up to 1 in each direction, so can do 2 random.linear(0, 1)s
    # Get corresponding observations (see how d4rlenv handles observation information (if walls and stuff included, assume none?)
    #   or make random? Need to think about this.
    # Calculate offset waypoint and direction advice with the corresponding teachers
    # Store everything in a big batch and return

    advice_input = np.zeros((batch_size*advice_interval, 383), np.float32)
    advice_output = np.zeros((batch_size*advice_interval, 2), np.float32)

    for j in range(batch_size):
        grid = np.repeat([np.repeat(1, 15)], 15, axis=0)
        grid[1:14, 1:14] = np.zeros((13, 13))

        position = np.random.uniform(1, 13, 2)
        velocity = np.random.normal(0, 0.1, 2)
        target = np.random.uniform(1, 13, 2)
        grid[target.astype(np.int32)[0], target.astype(np.int32)[0]] = -1

        # offset_waypoint = np.random.uniform(-1, 1, 2).astype('f')
        # advice_input[j, :2], advice_input[j, 2:4], advice_input[j, 4:] = offset_waypoint, position, velocity

        waypoint = breadth_first_search(position, target, grid).astype('f')
        offset_waypoint = (waypoint - position).astype('f')
        advice_output[j*advice_interval:(j+1)*advice_interval] = get_direction(offset_waypoint, velocity)
        advice_input[j*advice_interval:(j+1)*advice_interval, 255:] = np.array(advice_embedding(torch.tensor(offset_waypoint)).detach())

        for k in range(advice_interval):
            position = position + np.random.normal(0, 0.5, 2)
            velocity = velocity + np.random.normal(0, 0.1, 2)
            advice_input[j*advice_interval+k, :255] = utils.get_obs(position, velocity, target)

    return advice_input, advice_output


# Parameters?
if __name__ == '__main__':
    start_time = time.time()
    input_converter = wtd()
    loss_trace = np.zeros(num_batches)
    advice_embedding = utils.mlp(2, None, 128, 0, output_mod=nn.Sigmoid())
    if(os.path.isfile('advice embedder.pth')):
        advice_embedding.load_state_dict(torch.load('advice embedder.pth'))
    torch.save(advice_embedding.state_dict(), 'advice embedder.pth')
    for b in range(num_batches):
        batch_input, batch_labels = generate_advice_batch(advice_embedding)
        batch_loss = input_converter.train(batch_input, batch_labels)
        loss_trace[b] = batch_loss
        if b % 100 == 0:
            torch.save(input_converter.trunk.state_dict(), 'trained_input_converter.pth')
            np.savetxt('mlp_loss.txt', loss_trace, fmt='%.10f', delimiter='\n')
        print("batch number ", b, ", loss: ", batch_loss.item())

    torch.save(input_converter.trunk.state_dict(), 'trained_input_converter.pth')

    print("time to train advice converter: ", time.time() - start_time)
    np.savetxt('mlp_loss.txt', loss_trace, fmt='%.10f', delimiter='\n')
    
    #plt.plot(range(num_batches), loss_trace)
    #plt.show()
