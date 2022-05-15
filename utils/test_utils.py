import numpy as np
import torch
import os
import utils.m_utils as mutils
import utils.utils as utils

def get_latest_model(save_dir):
    checkpoint_list = []
    for dirpath, _, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
    return latest_checkpoint

def load_model(model, checkpoint_file):
    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['models']['predictor_model'])

    #return model

def get_coord_pose(sg, rel_pose, init_pose, grid_dim, cell_size, device):
    # Create a grid where the starting location is always at the center looking upwards (like the ground-projected grids)
    # Then use the spatial transformer to move that location at the right place
    if isinstance(init_pose, list) or isinstance(init_pose, tuple):
        init_pose = torch.tensor(init_pose).unsqueeze(0)
    else:
        init_pose = init_pose.unsqueeze(0)
    init_pose = init_pose.to(device)

    zero_pose = torch.tensor([[0., 0., 0.]]).to(device)

    zero_coords = mutils.discretize_coords(x=zero_pose[:,0], z=zero_pose[:,1], grid_dim=(grid_dim, grid_dim), cell_size=cell_size)

    pose_grid = torch.zeros((1, 1, grid_dim, grid_dim), dtype=torch.float32).to(device)
    pose_grid[0,0,zero_coords[0,0], zero_coords[0,1]] = 1

    pose_grid_transf = sg.spatialTransformer(grid=pose_grid, pose=rel_pose, abs_pose=init_pose)
    
    pose_grid_transf = pose_grid_transf.squeeze(0).squeeze(0)
    inds = utils.unravel_index(pose_grid_transf.argmax(), pose_grid_transf.shape)

    pose_coord = torch.zeros((1, 1, 2), dtype=torch.int64).to(device)
    pose_coord[0,0,0] = inds[1] # inds is y,x
    pose_coord[0,0,1] = inds[0]
    return pose_coord
