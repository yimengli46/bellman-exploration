import numpy as np 
import matplotlib.pyplot as plt
from baseline_utils import read_occ_map_npy
from core import cfg

import skimage.measure

import frontier_utils as fr_utils

from localNavigator_Astar import localNav_Astar
import networkx as nx

import scipy.ndimage
from math import sqrt

from topo_map.img_to_skeleton import img2skeleton
from topo_map.skeleton_to_topoMap import skeleton2topoMap
from topo_map.utils import drawtoposkele_with_VE,  build_VE_from_graph

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
						fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
						ax[0].imshow(occupancy_grid)
						ax[0].scatter(f.points[1], f.points[0], c='white', zorder=2)
						ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
						ax[0].get_xaxis().set_visible(False)
						ax[0].get_yaxis().set_visible(False)
						ax[0].set_title('explored occupancy map')

						ax[1].imshow(component)
						ax[1].get_xaxis().set_visible(False)
						ax[1].get_yaxis().set_visible(False)
						ax[1].set_title('area potential')

						ax[2].imshow(gt_occupancy_grid)
						ax[2].get_xaxis().set_visible(False)
						ax[2].get_yaxis().set_visible(False)
						ax[2].set_title('gt occupancy map')

						fig.tight_layout()
						plt.title(f'component {ii}')
						plt.show()

	return frontiers

scene_name = 'yqstnuAEVhm_0'
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

cp_gt_occ_map = gt_occ_map.copy()

LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

gt_occ_map = np.where(gt_occ_map==1, cfg.FE.FREE_VAL, gt_occ_map) # free cell
gt_occ_map = np.where(gt_occ_map==0, cfg.FE.COLLISION_VAL, gt_occ_map) # occupied cell

occupancy_map = gt_occ_map.copy()

H, W = occupancy_map.shape
BLOCK = 20
observed_area_flag = np.zeros((H, W), dtype=bool)
observed_area_flag[BLOCK:H-BLOCK, BLOCK:W-BLOCK] = True

occupancy_map[~observed_area_flag] = cfg.FE.UNOBSERVED_VAL

#plt.imshow(occupancy_map)
#plt.show()

agent_map_pose = (31, 111)
frontiers = get_frontiers(occupancy_map, gt_occ_map, observed_area_flag)

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


'''
gray1, gau1, skeleton = img2skeleton(cp_gt_occ_map)
graph = skeleton2topoMap(skeleton)

v_lst, e_lst = build_VE_from_graph(graph, skeleton, vertex_dist=10)

fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
drawtoposkele_with_VE(graph, skeleton + (1 - gray1[:, :] / 255) * 2, v_lst, e_lst, ax=ax)
plt.show()

result = {}
result['vertices'] = v_lst
result['edges'] = e_lst
np.save(f'v_and_e.npy',result)
'''

topo_V_E = np.load(f'v_and_e.npy', allow_pickle=True).item()
v_lst, e_lst = topo_V_E['vertices'], topo_V_E['edges']

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))
ax[0].imshow(occupancy_map, cmap='gray')
for f in frontiers:
	ax[0].scatter(f.points[1], f.points[0], c='green', zorder=2)
	ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
ax[0].scatter(agent_map_pose[0], agent_map_pose[1], marker='s', s=50, c='red', zorder=5)

ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title('improved observed_occ_map + frontiers')

x, y = [], []
for ed in e_lst:
	v1 = v_lst[ed[0]]/10.
	v2 = v_lst[ed[1]]/10.
	y.append(v1[1])
	x.append(v1[0])
	y.append(v2[1])
	x.append(v2[0])
	ax[0].plot([v1[0], v2[0]], [v1[1], v2[1]], 
            'y-', lw=1)
ax[0].scatter(x=x, y=y, c='blue', s=2)


ax[1].imshow(gt_occ_map, cmap='gray')
ax[1].scatter(agent_map_pose[0], agent_map_pose[1], marker='s', s=50, c='red', zorder=5)

ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title('improved observed_occ_map + frontiers')

x, y = [], []
for ed in e_lst:
	v1 = v_lst[ed[0]]/10.
	v2 = v_lst[ed[1]]/10.
	y.append(v1[1])
	x.append(v1[0])
	y.append(v2[1])
	x.append(v2[0])
	ax[1].plot([v1[0], v2[0]], [v1[1], v2[1]], 
            'y-', lw=1)
ax[1].scatter(x=x, y=y, c='blue', s=2)
fig.tight_layout()
plt.title('observed area')
plt.show()
