import numpy as np 
import matplotlib.pyplot as plt 
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn
from core import cfg

scene_name = 'yqstnuAEVhm_0'
#============================== load the map ======================
occ_map_npy = np.load(f'output/semantic_map/test/yqstnuAEVhm_0/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
gt_occ_map = np.where(gt_occ_map==1, cfg.FE.FREE_VAL, gt_occ_map) # free cell
gt_occ_map = np.where(gt_occ_map==0, cfg.FE.COLLISION_VAL, gt_occ_map) # occupied cell

#=============================== load the running files =======================
# FME
FME_file = f'output/TESTING_RESULTS_360degree_FME_NAVMESH_MAP_1STEP_500STEPS/results_{scene_name}.npy'
FME_npy = np.load(FME_file, allow_pickle=True).item()

# Greedy
Greedy_file = f'output/TESTING_RESULTS_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_500STEPS/results_{scene_name}.npy'
Greedy_npy = np.load(Greedy_file, allow_pickle=True).item()

# DP
DP_file = f'output/TESTING_RESULTS_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS/results_{scene_name}.npy'
DP_npy = np.load(DP_file, allow_pickle=True).item()

map_flag = DP_npy[0]['observed_area_flag']
gt_occ_map[map_flag < 1] = 0

#=============================== draw the figure ==========================
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
ax.imshow(gt_occ_map, cmap='gray')

#================ draw FME ==================
traverse_lst = FME_npy[0]['trajectory']
x_coord_lst, z_coord_lst, theta_lst = [], [], []
for cur_pose in traverse_lst:
	x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
	x_coord_lst.append(x_coord)
	z_coord_lst.append(z_coord)			
	theta_lst.append(cur_pose[2])
marker, scale = gen_arrow_head_marker(theta_lst[-1])
ax.scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='cyan', zorder=5, alpha=0.2)
ax.plot(x_coord_lst, z_coord_lst, lw=5, c='cyan', zorder=3, alpha=0.2)

#================ draw Greedy ==================
traverse_lst = Greedy_npy[0]['trajectory']
x_coord_lst, z_coord_lst, theta_lst = [], [], []
for cur_pose in traverse_lst:
	x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
	x_coord_lst.append(x_coord)
	z_coord_lst.append(z_coord)			
	theta_lst.append(cur_pose[2])
marker, scale = gen_arrow_head_marker(theta_lst[-1])
ax.scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='green', zorder=5, alpha=0.2)
ax.plot(x_coord_lst, z_coord_lst, lw=5, c='green', zorder=3, alpha=0.2)

#================ draw DP ==================
traverse_lst = DP_npy[0]['trajectory']
x_coord_lst, z_coord_lst, theta_lst = [], [], []
for cur_pose in traverse_lst:
	x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
	x_coord_lst.append(x_coord)
	z_coord_lst.append(z_coord)			
	theta_lst.append(cur_pose[2])
marker, scale = gen_arrow_head_marker(theta_lst[-1])
ax.scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=6)
ax.scatter(x_coord_lst, 
			   z_coord_lst, 
			   c=range(len(x_coord_lst)), 
			   cmap='viridis', 
			   s=np.linspace(5, 2, num=len(x_coord_lst))**2, 
			   zorder=4)
ax.scatter(x_coord_lst[0],
					  z_coord_lst[0],
					  marker='s',
					  s=50,
					  c='red',
					  zorder=5)

ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
#ax.set_title('improved observed_occ_map + frontiers')

fig.tight_layout()
plt.title('observed area')
plt.show()