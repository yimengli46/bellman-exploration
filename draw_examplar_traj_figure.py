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
FME_file = f'output/TESTING_RESULTS_360degree_FME_NAVMESH_MAP_1STEP_1000STEPS/results_{scene_name}_step_cov_pairs.npy'
FME_npy = np.load(FME_file, allow_pickle=True).item()

# Greedy
Greedy_file = f'output/TESTING_RESULTS_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_500STEPS/results_{scene_name}.npy'
Greedy_npy = np.load(Greedy_file, allow_pickle=True).item()

# DP
DP_file = f'output/TESTING_RESULTS_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS/results_{scene_name}.npy'
DP_npy = np.load(DP_file, allow_pickle=True).item()
