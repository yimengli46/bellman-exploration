import numpy as np
import os
import torch

def est_occ_from_depth(local3D, grid_dim, cell_size, device, occupancy_height_thresh=-0.9):

    ego_grid_occ = torch.zeros((len(local3D), 3, grid_dim[0], grid_dim[1]), dtype=torch.float32, device=device)

    for k in range(len(local3D)):

        local3D_step = local3D[k]

        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        local3D_step = local3D_step[(z < 3) & (z > 0.5) & (y < 1), :]

        # initialize all locations as unknown (void)
        occ_lbl = torch.zeros((local3D_step.shape[0], 1), dtype=torch.float32, device=device)

        # threshold height to get occupancy and free labels
        thresh = occupancy_height_thresh
        y = local3D_step[:,1]
        occ_lbl[y>thresh,:] = 1
        occ_lbl[y<=thresh,:] = 2

        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)

        ## Replicate label pooling
        grid = torch.empty(3, grid_dim[0], grid_dim[1], device=device)
        grid[:] = 1 / 3

        # If the robot does not project any values on the grid, then return the empty grid
        if map_coords.shape[0]==0:
            ego_grid_occ[k,:,:,:] = grid.unsqueeze(0)
            continue

        concatenated = torch.cat([map_coords, occ_lbl.long()], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        ego_grid_occ[k,:,:,:] = grid / grid.sum(dim=0)

    return ego_grid_occ

def discretize_coords(x, z, grid_dim, cell_size, translation=0):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    # If translation=0, assumes the agent is at the center
    # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
    map_coords = torch.zeros((len(x), 2), device='cuda')
    xb = torch.floor(x[:]/cell_size) + (grid_dim[0]-1)/2.0
    zb = torch.floor(z[:]/cell_size) + (grid_dim[1]-1)/2.0 + translation
    xb = xb.int()
    zb = zb.int()
    map_coords[:,0] = xb
    map_coords[:,1] = zb
    # keep bin coords within dimensions
    map_coords[map_coords>grid_dim[0]-1] = grid_dim[0]-1
    map_coords[map_coords<0] = 0
    return map_coords.long()

def crop_grid(grid, crop_size):
    # Assume input grid is already transformed such that agent is at the center looking upwards
    grid_dim_h, grid_dim_w = grid.shape[2], grid.shape[3]
    cx, cy = int(grid_dim_w/2.0), int(grid_dim_h/2.0)
    rx, ry = int(crop_size[0]/2.0), int(crop_size[1]/2.0)
    top, bottom, left, right = cx-rx, cx+rx, cy-ry, cy+ry
    return grid[:, :, top:bottom, left:right]
