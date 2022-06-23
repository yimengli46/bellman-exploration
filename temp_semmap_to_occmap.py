from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn
import numpy as np 
import matplotlib.pyplot as plt 
from core import cfg

split = 'test'
scene_name =  'yqstnuAEVhm_0'  #'17DRP5sb8fy_0'
semantic_map_folder = f'output/semantic_map/{split}'

sem_map_npy = np.load(f'{semantic_map_folder}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
sem_map, _, _, _ = read_map_npy(sem_map_npy)

occ_map = np.ones(sem_map.shape, dtype=np.int8) * cfg.FE.UNOBSERVED_VAL
mask_occupied = np.logical_and(sem_map > 0, sem_map != 2)
occ_map = np.where(mask_occupied, cfg.FE.COLLISION_VAL, occ_map)
mask_free = (sem_map == 2)
occ_map = np.where(mask_free, cfg.FE.FREE_VAL, occ_map)

plt.imshow(occ_map, cmap='gray')
plt.show() 