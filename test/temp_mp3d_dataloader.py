import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda')

env_scene = '17DRP5sb8fy'
floor_id = 0
scene_name = '17DRP5sb8fy_0'

def get_region(robot_pos, H, W, size=2):
	y, x = robot_pos
	y1 = max(0, y-size)
	y2 = min(H-1, y+size)
	x1 = max(0, x-size)
	x2 = min(W-1, x+size)

	return (y1, x1, y2, x2)

#======================================== generate gt map M_c ============================================
sem_map_npy = np.load(f'{cfg.SAVE.SEMANTIC_MAP_PATH}/{cfg.MAIN.SPLIT}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
gt_sem_map, pose_range, coords_range, WH = read_map_npy(sem_map_npy)
occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{cfg.MAIN.SPLIT}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, _, _, _ = read_occ_map_npy(occ_map_npy)

gt_occupancy_map = gt_occ_map.copy()
gt_occupancy_map = np.where(gt_occupancy_map == 1, cfg.FE.FREE_VAL, gt_occupancy_map)  # free cell
gt_occupancy_map = np.where(gt_occupancy_map == 0, cfg.FE.COLLISION_VAL, gt_occupancy_map)  # occupied cell

M_c = np.stack((gt_occupancy_map, gt_sem_map))
H, W = gt_sem_map.shape

#====================================== generate (start, goal) locs, compute path P==========================
LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

G = LN.get_G_from_map(gt_occupancy_map)
largest_cc = list(max(nx.connected_components(G), key=len))
start_loc, goal_loc = random.choices(largest_cc, k=2)
path = nx.shortest_path(G,
						source=start_loc,
						target=goal_loc)

M_p = np.zeros(M_c.shape, dtype=np.int16)
observed_area_flag = np.zeros((H, W), dtype=bool)
for i_loc, robot_loc in enumerate(path):
	print(f'robot_loc = {robot_loc}')
	#=================================== generate partial map M_p ==================================
	roi = get_region(robot_loc, H, W, size=cfg.PRED.NEIGHBOR_SIZE)
	M_p[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1] = M_c[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1]
	observed_area_flag[roi[0]:roi[2]+1, roi[1]:roi[3]+1] = True

	#================================= compute area at frontier points ========================
	U_a = np.zeros((H, W), dtype=np.float32)
	observed_occupancy_map = M_p[0]
	frontiers = fr_utils.get_frontiers(observed_occupancy_map, gt_occupancy_map, observed_area_flag)
	agent_map_pose = (robot_loc[1], robot_loc[0])
	frontiers = LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, observed_occupancy_map)

	for fron in frontiers:
		points = fron.points.transpose() # N x 2
		R = min(1. * fron.R / cfg.PRED.MAX_AREA, 1.0)
		U_a[points[:, 0], points[:, 1]] = R

	#======= visualize M_p=========
	if cfg.PRED.FLAG_VISUALIZE_PRED_LABELS:
		occ_map_Mp = M_p[0]
		sem_map_Mp = M_p[1]
		color_sem_map_Mp = apply_color_to_map(sem_map_Mp)

		fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
		ax[0][0].imshow(occ_map_Mp, cmap='gray')
		ax[0][0].get_xaxis().set_visible(False)
		ax[0][0].get_yaxis().set_visible(False)
		ax[0][0].set_title('input: occupancy_map_Mp')
		ax[0][1].imshow(color_sem_map_Mp)
		ax[0][1].get_xaxis().set_visible(False)
		ax[0][1].get_yaxis().set_visible(False)
		ax[0][1].set_title('input: semantic_map_Mp')

		ax[1][0].imshow(occ_map_Mp, cmap='gray')
		x_coord_lst = [path[i][1] for i in range(i_loc+1)]
		z_coord_lst = [path[i][0] for i in range(i_loc+1)]
		ax[1][0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
		for f in frontiers:
			ax[1][0].scatter(f.points[1], f.points[0], c='yellow', zorder=2)
			ax[1][0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
		ax[1][0].get_xaxis().set_visible(False)
		ax[1][0].get_yaxis().set_visible(False)
		ax[1][0].set_title('observed_occ_map + frontiers')

		ax[1][1].imshow(U_a, vmin=0.0, vmax=1.0)
		ax[1][1].get_xaxis().set_visible(False)
		ax[1][1].get_yaxis().set_visible(False)
		ax[1][1].set_title('output: U_a')

		fig.tight_layout()
		plt.show()