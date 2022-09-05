import numpy as np 
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import read_occ_map_npy
from core import cfg

import skimage.measure

import modeling.utils.frontier_utils as fr_utils

from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx

import scipy.ndimage
from math import sqrt
import torch
import torch.nn.functional as F
from timeit import default_timer as timer
import random
import math
from scipy import ndimage

def crop_map(h, x, crop_size, mode="bilinear"):
    """
    Crops a tensor h centered around location x with size crop_size
    Inputs:
        h - (bs, F, H, W)
        x - (bs, 2) --- (x, y) locations
        crop_size - scalar integer
    Conventions for x:
        The origin is at the top-left, X is rightward, and Y is downward.
    """

    bs, _, H, W = h.size()
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H // 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W // 2
    start = -(crop_size - 1) / 2 if crop_size % 2 == 1 else -(crop_size // 2)
    end = start + crop_size - 1
    x_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(0)
        .expand(crop_size, -1)
        .contiguous()
        .float()
    )
    y_grid = (
        torch.arange(start, end + 1, step=1)
        .unsqueeze(1)
        .expand(-1, crop_size)
        .contiguous()
        .float()
    )
    center_grid = torch.stack([x_grid, y_grid], dim=2).to(
        h.device
    )  # (crop_size, crop_size, 2)

    x_pos = x[:, 0] - Wby2  # (bs, )
    y_pos = x[:, 1] - Hby2  # (bs, )

    crop_grid = center_grid.unsqueeze(0).expand(
        bs, -1, -1, -1
    )  # (bs, crop_size, crop_size, 2)
    crop_grid = crop_grid.contiguous()

    # Convert the grid to (-1, 1) range
    crop_grid[:, :, :, 0] = (
        crop_grid[:, :, :, 0] + x_pos.unsqueeze(1).unsqueeze(2)
    ) / Wby2
    crop_grid[:, :, :, 1] = (
        crop_grid[:, :, :, 1] + y_pos.unsqueeze(1).unsqueeze(2)
    ) / Hby2

    h_cropped = F.grid_sample(h, crop_grid, mode=mode, align_corners=False)

    return h_cropped

def spatial_transform_map(p, x, invert=True, mode="bilinear"):
    """
    Inputs:
        p     - (bs, f, H, W) Tensor
        x     - (bs, 3) Tensor (x, y, theta) transforms to perform
    Outputs:
        p_trans - (bs, f, H, W) Tensor
    Conventions:
        Shift in X is rightward, and shift in Y is downward. Rotation is clockwise.
    Note: These denote transforms in an agent's position. Not the image directly.
    For example, if an agent is moving upward, then the map will be moving downward.
    To disable this behavior, set invert=False.
    """
    device = p.device
    H, W = p.shape[2:]

    trans_x = x[:, 0]
    trans_y = x[:, 1]
    # Convert translations to -1.0 to 1.0 range
    Hby2 = (H - 1) / 2 if H % 2 == 1 else H / 2
    Wby2 = (W - 1) / 2 if W % 2 == 1 else W / 2

    trans_x = trans_x / Wby2
    trans_y = trans_y / Hby2
    rot_t = x[:, 2]

    sin_t = torch.sin(rot_t)
    cos_t = torch.cos(rot_t)

    # This R convention means Y axis is downwards.
    A = torch.zeros(p.size(0), 3, 3).to(device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = -sin_t
    A[:, 1, 0] = sin_t
    A[:, 1, 1] = cos_t
    A[:, 0, 2] = trans_x
    A[:, 1, 2] = trans_y
    A[:, 2, 2] = 1

    # Since this is a source to target mapping, and F.affine_grid expects
    # target to source mapping, we have to invert this for normal behavior.
    Ainv = torch.inverse(A)

    # If target to source mapping is required, invert is enabled and we invert
    # it again.
    if invert:
        Ainv = torch.inverse(Ainv)

    Ainv = Ainv[:, :2]
    grid = F.affine_grid(Ainv, p.size(), align_corners=False)
    p_trans = F.grid_sample(p, grid, mode=mode, align_corners=False)

    return p_trans


def get_frontiers(occupancy_grid, gt_occupancy_grid, observed_area_flag, skeleton_G):
	filtered_grid = scipy.ndimage.maximum_filter(occupancy_grid == cfg.FE.UNOBSERVED_VAL, size=3)
	frontier_point_mask = np.logical_and(filtered_grid, occupancy_grid == cfg.FE.FREE_VAL)

	if cfg.FE.GROUP_INFLATION_RADIUS < 1:
		inflated_frontier_mask = frontier_point_mask
	else:
		inflated_frontier_mask = gridmap.utils.inflate_grid(frontier_point_mask,
			inflation_radius=cfg.FE.GROUP_INFLATION_RADIUS, obstacle_threshold=0.5,
			collision_val=1.0) > 0.5

	# Group the frontier points into connected components
	labels, nb = scipy.ndimage.label(inflated_frontier_mask)

	# Extract the frontiers
	frontiers = set()
	for ii in range(nb):
		raw_frontier_indices = np.where(np.logical_and(labels == (ii + 1), frontier_point_mask))
		frontiers.add(
			fr_utils.Frontier(
				np.concatenate((raw_frontier_indices[0][None, :],
								raw_frontier_indices[1][None, :]),
							   axis=0)))

	# Compute potential
	if cfg.NAVI.PERCEPTION == 'Potential':
		free_but_unobserved_flag = np.logical_and(gt_occupancy_grid == cfg.FE.FREE_VAL, observed_area_flag == False)
		free_but_unobserved_flag = scipy.ndimage.maximum_filter(free_but_unobserved_flag, size=3)

		labels, nb = scipy.ndimage.label(free_but_unobserved_flag)

		for ii in range(nb):
			component = (labels == (ii+1))
			for f in frontiers:
				if component[int(f.centroid[0]), int(f.centroid[1])]:
					f.R = np.sum(component)
					#f.D = round(sqrt(f.R), 2)
					#try:
					#cost_dall, cost_din, cost_dout, skeleton, skeleton_graph = skeletonize_map(component)#, display=True)
					#cost_dall, cost_din, cost_dout, skeleton_component = skeletonize_frontier(component, skeleton)
					#t1 = timer()
					#cost_dall, cost_din, cost_dout, component_G = skeletonize_frontier_on_the_graph(component, skeleton)
					#t2 = timer()
					#print(f't2 - t1 = {t2-t1}')
					'''
					except:
						cost_dall = round(sqrt(f.R), 2)
						cost_din = cost_dall
						cost_dout = cost_dall
					'''
					#print(f'cost_dall = {cost_dall}, cost_din = {cost_din}, cost_dout = {cost_dout}')

					if False:
						fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
						ax[0].imshow(occupancy_grid, cmap='gray')
						ax[0].scatter(f.points[1], f.points[0], c='yellow', zorder=2)
						ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
						ax[0].get_xaxis().set_visible(False)
						ax[0].get_yaxis().set_visible(False)
						ax[0].set_title('explored occupancy map')

						ax[1].imshow(component, cmap='gray')
						ax[1].get_xaxis().set_visible(False)
						ax[1].get_yaxis().set_visible(False)
						ax[1].set_title(f'area potential, component {ii}')


						fig.tight_layout()
						#plt.title(f'component {ii}')
						plt.show()

	return frontiers

scene_name = 'yqstnuAEVhm_0'
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/test/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

cp_gt_occ_map = gt_occ_map.copy()

LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

gt_occ_map = np.where(gt_occ_map==1, cfg.FE.FREE_VAL, gt_occ_map) # free cell
gt_occ_map = np.where(gt_occ_map==0, cfg.FE.COLLISION_VAL, gt_occ_map) # occupied cell

occupancy_map = gt_occ_map.copy()

H, W = occupancy_map.shape
BLOCK = 40
observed_area_flag = np.zeros((H, W), dtype=bool)
observed_area_flag[BLOCK:H-BLOCK, BLOCK:W-BLOCK] = True

occupancy_map[~observed_area_flag] = cfg.FE.UNOBSERVED_VAL


agent_map_pose = (91, 111)


# Expand to add batch dim
t1 = timer()
Wby2, Hby2 = W // 2, H // 2
tform_trans = torch.Tensor([[agent_map_pose[0] - Wby2, agent_map_pose[1] - Hby2, 0]])
tensor_occupancy_map = torch.tensor(occupancy_map).float().unsqueeze(0).unsqueeze(1)

# Crop a large-enough map around agent
_, N, H, W = tensor_occupancy_map.shape
crop_center = torch.Tensor([[W / 2.0, H / 2.0]]) + tform_trans[:, :2]
map_size = int(2 * 24 / 0.05)
tensor_occupancy_map = crop_map(tensor_occupancy_map, crop_center, map_size)

rot = random.uniform(-math.pi, math.pi)
rot = 3.14/2
tform_rot = torch.Tensor([[0, 0, rot]])
tensor_occupancy_map = spatial_transform_map(tensor_occupancy_map, tform_rot)
# Crop out the appropriate size of the map
_, N, H, W = tensor_occupancy_map.shape
map_center = torch.Tensor([[W / 2.0, H / 2.0]])
map_size = int(24 / 0.05)
tensor_occupancy_map = crop_map(tensor_occupancy_map, map_center, map_size, 'nearest')

# change back to numpy
tensor_occupancy_map = tensor_occupancy_map.squeeze(0).squeeze(0).numpy()

t2 = timer()
print(f't2 - t1 = {t2-t1}')

temp_map = np.stack((tensor_occupancy_map, tensor_occupancy_map), axis=2)
rot_map = ndimage.rotate(temp_map, 45, reshape=False)

'''
# fill the boundary
result_map = np.zeros((480, 480), dtype=np.int64)

# shift the image so the agent is at the center
Wby2, Hby2 = W // 2, H // 2
trans = np.array([Hby2 - agent_map_pose[1], Wby2 - agent_map_pose[0]])

result_shift = scipy.ndimage.shift(occupancy_map, trans, mode='nearest')
'''
assert 1==2



frontiers = get_frontiers(occupancy_map, gt_occ_map, observed_area_flag, None)

frontiers = LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, occupancy_map)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
ax.imshow(occupancy_map)
for f in frontiers:
	ax.scatter(f.points[1], f.points[0], c='white', zorder=2)
	ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
ax.scatter(agent_map_pose[0], agent_map_pose[1], marker='s', s=50, c='red', zorder=5)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()

