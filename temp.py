import numpy as np 
import matplotlib.pyplot as plt
from baseline_utils import read_occ_map_npy
from core import cfg

import skimage.measure

import frontier_utils as fr_utils

scene_name = '2t7WUuJeko7_0'
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)

occupancy_map = gt_occ_map.copy()
occupancy_map = np.where(occupancy_map==1, cfg.FE.FREE_VAL, occupancy_map) # free cell
occupancy_map = np.where(occupancy_map==0, cfg.FE.COLLISION_VAL, occupancy_map) # occupied cell

H, W = occupancy_map.shape
BLOCK = 40
occupancy_map[0:BLOCK, :] = cfg.FE.UNOBSERVED_VAL
occupancy_map[H-BLOCK:H, :] = cfg.FE.UNOBSERVED_VAL
occupancy_map[:, 0:BLOCK] = cfg.FE.UNOBSERVED_VAL
occupancy_map[:, W-BLOCK:W] = cfg.FE.UNOBSERVED_VAL


plt.imshow(occupancy_map)
plt.show()


frontiers2 = fr_utils.get_frontiers(occupancy_map)

#chosen_frontier = fr_utils.get_frontier_with_maximum_area(frontiers2, [], gt_occ_map)


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
ax.imshow(occupancy_map)
for f in frontiers2:
	ax.scatter(f.points[1], f.points[0], c='white', zorder=2)
	ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()