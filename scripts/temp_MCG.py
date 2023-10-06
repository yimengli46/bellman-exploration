import numpy as np
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import read_occ_map_npy
from core import cfg
import modeling.utils.frontier_utils as fr_utils

from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx

import scipy.ndimage
from math import sqrt

from skimage.graph import MCP_Geometric as MCPG


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
                        # plt.title(f'component {ii}')
                        plt.show()

    return frontiers


scene_name = 'yqstnuAEVhm_0'
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/test/{scene_name}/BEV_occupancy_map.npy',
                      allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

cp_gt_occ_map = gt_occ_map.copy()


LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

gt_occ_map = np.where(gt_occ_map == 1, cfg.FE.FREE_VAL, gt_occ_map)  # free cell
gt_occ_map = np.where(gt_occ_map == 0, cfg.FE.COLLISION_VAL, gt_occ_map)  # occupied cell

occupancy_map = gt_occ_map.copy()

H, W = occupancy_map.shape
BLOCK = 50
observed_area_flag = np.zeros((H, W), dtype=bool)
observed_area_flag[BLOCK:H-BLOCK, BLOCK:W-BLOCK] = True

occupancy_map[~observed_area_flag] = cfg.FE.UNOBSERVED_VAL

# plt.imshow(occupancy_map)
# plt.show()

agent_map_pose = (81, 111)
frontiers = get_frontiers(occupancy_map, gt_occ_map, observed_area_flag)

# frontiers = LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, occupancy_map)


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

# connected component through scipy_label
local_occupancy_map = occupancy_map.copy()
local_occupancy_map[local_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL
local_occupancy_map[local_occupancy_map == cfg.FE.COLLISION_VAL] = 0

labels, nb = scipy.ndimage.label(local_occupancy_map, structure=np.ones((3, 3)))
agent_label = labels[agent_map_pose[1], agent_map_pose[0]]

filtered_frontiers = set()
for fron in frontiers:
    fron_centroid_coords = (int(fron.centroid[1]),
                            int(fron.centroid[0]))
    fron_label = labels[fron_centroid_coords[1], fron_centroid_coords[0]]
    if fron_label == agent_label:
        filtered_frontiers.add(fron)

frontiers = filtered_frontiers

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

# computed distance to each frontier through skimage.graph.MCG
local_occupancy_map[local_occupancy_map != 0] = 1
local_occupancy_map[local_occupancy_map == 0] = 1000
m = MCPG(local_occupancy_map)
starts = [[agent_map_pose[1], agent_map_pose[0]]]
ends = []
for fron in frontiers:
    ends.append([fron.centroid[0], fron.centroid[1]])

cost_array, tracebacks_array = m.find_costs(starts, ends)

# Transpose `ends` so can be used to index in NumPy
ends_idx = tuple(np.array(ends).T.tolist())
costs = cost_array[ends_idx]

tracebacks = [m.traceback(end) for end in ends]

# visualize the tracebacks
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
ax.imshow(occupancy_map)
for f in frontiers:
    ax.scatter(f.points[1], f.points[0], c='white', zorder=2)
    ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
ax.scatter(agent_map_pose[0], agent_map_pose[1], marker='s', s=50, c='red', zorder=5)
nodes = tracebacks[0]
for path in tracebacks[1:]:
    nodes += path
nodes = np.array(nodes)
ax.scatter(nodes[:, 1], nodes[:, 0], c='blue', zorder=2)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()
