import numpy as np 
import matplotlib.pyplot as plt
from modeling.utils.baseline_utils import read_occ_map_npy, crop_map
from core import cfg
import skimage.measure
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
from ANS_modeling.model import RL_Policy
import torch.nn as nn

from skimage.draw import line


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


agent_map_coords = (43, 111)


device = torch.device('cuda:0')
g_policy = RL_Policy((8, 240, 240), 
	base_kwargs={'recurrent': 0,
	'hidden_size': 256,
	'downscaling': 2
	}).to(device)

state_dict = torch.load('trained_weights/model_best.global', map_location=lambda storage, loc: storage)
g_policy.load_state_dict(state_dict)

#======================= build input ================================

M_p = np.zeros((1, 4, H, W), dtype='float32')
M_p[0, 0, :, :] = (occupancy_map == cfg.FE.COLLISION_VAL) # first channel obstacle
M_p[0, 1, :, :] = (occupancy_map != cfg.FE.UNOBSERVED_VAL) # second channel explored
M_p[0, 2, agent_map_coords[1]-1:agent_map_coords[1]+2, agent_map_coords[0]-1:agent_map_coords[0]+2] = 1 # third channel current location
M_p[0, 3, agent_map_coords[1]-1:agent_map_coords[1]+2, agent_map_coords[0]-1: agent_map_coords[0]+2] = 1 # fourth channel visited places
tensor_M_p = torch.tensor(M_p).float()
#print(f'tensor_M_p.shape = {tensor_M_p.shape}')

#================== crop out the map centered at the agent ==========================
_, _, H, W = M_p.shape
Wby2, Hby2 = W // 2, H // 2
tform_trans = torch.Tensor([[agent_map_coords[0] - Wby2, agent_map_coords[1] - Hby2, 0]])
crop_center = torch.Tensor([[W / 2.0, H / 2.0]]) + tform_trans[:, :2]
# Crop out the appropriate size of the map
local_map_size = int(240)
tensor_local_M_p = crop_map(tensor_M_p, crop_center, local_map_size, 'nearest')
global_map_size = int(480)
tensor_global_M_p = crop_map(tensor_M_p, crop_center, global_map_size, 'nearest')





local_w = 240
local_h = 240
num_scenes = 1
global_downscaling = 2


global_input = torch.zeros(num_scenes, 8, local_w, local_h)
global_orientation = torch.zeros(num_scenes, 1).long()
global_orientation[0] = int((90.0 + 180.0) / 5.)

global_input[:, 0:4, :, :] = tensor_local_M_p
global_input[:, 4:, :, :] = nn.MaxPool2d(global_downscaling)(tensor_global_M_p)

global_input = global_input.to(device)
global_orientation = global_orientation.to(device)

# Run Global Policy (global_goals = Long-Term Goal)
_, g_action, g_action_log_prob, _ = \
	g_policy.act(
		global_input,
		None,
		None,
		extras=global_orientation,
		deterministic=False
	)

cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
				for action in cpu_actions]
global_goals_coords = (agent_map_coords[0]+global_goals[0][0], agent_map_coords[1]+global_goals[0][1]) 

# find reachable global_goals
rr_line, cc_line = line(agent_map_coords[1], agent_map_coords[0], global_goals_coords[1], global_goals_coords[0])
binary_occupancy_map = occupancy_map.copy()
binary_occupancy_map[binary_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL
binary_occupancy_map[binary_occupancy_map == cfg.FE.COLLISION_VAL] = 0
labels, nb = scipy.ndimage.label(binary_occupancy_map, structure=np.ones((3,3)))
agent_label = labels[agent_map_coords[1], agent_map_coords[0]]
for idx in list(reversed(range(len(rr_line)))):
	if labels[rr_line[idx], cc_line[idx]] == agent_label:
		reachable_global_goals_coords = (cc_line[idx], rr_line[idx])
		break

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 15))
ax.imshow(occupancy_map)
ax.scatter(agent_map_coords[0], agent_map_coords[1], marker='s', s=50, c='red', zorder=5)
ax.scatter(reachable_global_goals_coords[0], reachable_global_goals_coords[1], marker='s', s=50, c='black', zorder=5)
ax.scatter(global_goals_coords[0], global_goals_coords[1], marker='s', s=50, c='blue', zorder=5)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()

