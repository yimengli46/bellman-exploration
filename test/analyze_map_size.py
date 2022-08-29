import numpy as np 
from core import cfg
from modeling.utils.baseline_utils import read_map_npy
import matplotlib.pyplot as plt

scene_list = cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST
split = 'test'

scene_floor_dict = np.load(
	f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/large_scale_{split}_scene_floor_dict.npy', allow_pickle=True).item()

h_lst, w_lst = [], []
for env_scene in scene_list:
	scene_dict = scene_floor_dict[env_scene]
	for floor_id in list(scene_dict.keys()):
		split = 'test'
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'
	
		#========================== load the scene map===========================
		sem_map_npy = np.load(
			f'output/large_scale_semantic_map/{scene_name}/BEV_semantic_map.npy',
			allow_pickle=True).item()
		semantic_map, pose_range, coords_range, WH = read_map_npy(sem_map_npy)

		h, w = semantic_map.shape

		h_lst.append(h)
		w_lst.append(w)

