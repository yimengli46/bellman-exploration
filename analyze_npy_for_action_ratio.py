import numpy as np 
from core import cfg
import pandas as pd
from modeling.utils.baseline_utils import read_map_npy
import matplotlib.pyplot as plt

scene_list = cfg.MAIN.TEST_SCENE_LIST
output_folder = 'output' #cfg.SAVE.TESTING_RESULTS_FOLDER
result_folder = 'TESTING_RESULTS_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_500STEPS'
scene_list = ['yqstnuAEVhm_0']

avg_percent_list = []
avg_step_list = []
thresh_percent = .01

#for scene_name in scene_list:
for scene_name in scene_list:
	try:
		#========================== load the scene map===========================
		sem_map_npy = np.load(
			f'output/semantic_map/test/{scene_name}/BEV_semantic_map.npy',
			allow_pickle=True).item()
		semantic_map, pose_range, coords_range, WH = read_map_npy(sem_map_npy)
		occ_map = semantic_map > 0
		area = np.sum(occ_map) * .0025

		results_npy = np.load(f'{output_folder}/{result_folder}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test = len(results_npy.keys())

		percent_list = []
		step_list = []

		for i in range(num_test):
			result = results_npy[i]
			print(f'result = {result}')
			
	except:
		print(f'failed to process scene {scene_name}.')

