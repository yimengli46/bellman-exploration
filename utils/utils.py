import numpy as np
import os
import math
import torch
import torch.nn.functional as F
import quaternion

def depth_to_3D(depth_obs, img_size, xs, ys, inv_K):

    depth = depth_obs[...,0].reshape(1, img_size[0], img_size[1])

    # Unproject
    # negate depth as the camera looks along -Z
    # SPEEDUP - create ones in constructor
    xys = torch.vstack((torch.mul(xs, depth), torch.mul(ys, depth), -depth, torch.ones(depth.shape, device='cuda'))) # 4 x 128 x 128
    xys = xys.reshape(4, -1)
    xy_c0 = torch.matmul(inv_K, xys)

    # SPEEDUP - don't allocate new memory, manipulate existing shapes
    local3D = torch.zeros((xy_c0.shape[1], 3), dtype=torch.float32, device='cuda')
    local3D[:,0] = xy_c0[0,:]
    local3D[:,1] = xy_c0[1,:]
    local3D[:,2] = xy_c0[2,:]

    return local3D

def get_sim_location(agent_state):
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    height = agent_state.position[1]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    pose = x, y, o
    return pose, height

def get_rel_pose(pos2, pos1):
    x1, y1, o1 = pos1
    if len(pos2)==2: # if pos2 has no rotation
        x2, y2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy
    else:
        x2, y2, o2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        do = o2 - o1
        if do < -math.pi:
            do += 2 * math.pi
        if do > math.pi:
            do -= 2 * math.pi
        return dx, dy, do

# Taken from: https://github.com/pytorch/pytorch/issues/35674
def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)