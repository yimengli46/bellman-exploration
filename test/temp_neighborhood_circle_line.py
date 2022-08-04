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

from skimage.draw import line, circle_perimeter
from timeit import default_timer as timer

def get_frontiers(occupancy_grid, gt_occupancy_grid, observed_area_flag):
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
					f.D = round(sqrt(f.R), 2)

					if False:
						fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))
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

gt_occ_map = np.where(gt_occ_map==1, cfg.FE.FREE_VAL, gt_occ_map) # free cell
gt_occ_map = np.where(gt_occ_map==0, cfg.FE.COLLISION_VAL, gt_occ_map) # occupied cell

occupancy_map = gt_occ_map.copy()

H, W = occupancy_map.shape
BLOCK = 10
observed_area_flag = np.zeros((H, W), dtype=bool)
observed_area_flag[BLOCK:H-BLOCK, BLOCK:W-BLOCK] = True

occupancy_map[~observed_area_flag] = cfg.FE.UNOBSERVED_VAL

agent_map_pose = (30, 100) # (81, 111)
occupancy_map[agent_map_pose[1]-20:agent_map_pose[1]+20, agent_map_pose[0]-20:agent_map_pose[0]+20] = cfg.FE.UNOBSERVED_VAL
#occupancy_map[agent_map_pose[1]-5:agent_map_pose[1]+5, agent_map_pose[0]-5:agent_map_pose[0]+5] = cfg.FE.FREE_VAL


# draw circle
#'''
t1 = timer()
rr_cir, cc_cir = circle_perimeter(agent_map_pose[1], agent_map_pose[0], 10, method='andres')
rr_full_cir = np.array([], dtype='int64')
cc_full_cir = np.array([], dtype='int64')
for idx in range(len(rr_cir)):
	rr_line, cc_line = line(agent_map_pose[1], agent_map_pose[0], rr_cir[idx], cc_cir[idx])

	mask_line = np.logical_and(np.logical_and(rr_line >= 0, rr_line < H), 
		np.logical_and(cc_line >= 0, cc_line < W))
	rr_line = rr_line[mask_line]
	cc_line = cc_line[mask_line]
	
	first_unknown = np.nonzero(occupancy_map[rr_line, cc_line] == cfg.FE.UNOBSERVED_VAL)[0]
	first_collision = np.nonzero(occupancy_map[rr_line, cc_line] == cfg.FE.COLLISION_VAL)[0]
	idx_first_unknown = 0 if len(first_unknown) == 0 else first_unknown[0]
	idx_first_collision = len(rr_line) if len(first_collision) == 0 else first_collision[0]
	rr_full_cir = np.concatenate((rr_full_cir, rr_line[idx_first_unknown:idx_first_collision]))
	cc_full_cir = np.concatenate((cc_full_cir, cc_line[idx_first_unknown:idx_first_collision]))

mask_complement = np.zeros(occupancy_map.shape, dtype='bool')
mask_complement[rr_full_cir, cc_full_cir] = True
occupancy_map[np.logical_and(mask_complement, occupancy_map == cfg.FE.UNOBSERVED_VAL)] = cfg.FE.FREE_VAL
t2 = timer()
print(f'time = {t2 - t1}')
#'''




frontiers = get_frontiers(occupancy_map, gt_occ_map, observed_area_flag)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
ax.imshow(occupancy_map)
'''
for f in frontiers:
	ax.scatter(f.points[1], f.points[0], c='white', zorder=2)
	ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
'''
ax.scatter(agent_map_pose[0], agent_map_pose[1], marker='s', s=50, c='red', zorder=5)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()


