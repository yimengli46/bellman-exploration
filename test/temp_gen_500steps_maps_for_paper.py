import numpy as np 
from core import cfg
import pandas as pd
from modeling.utils.baseline_utils import read_map_npy
import matplotlib.pyplot as plt



#output_folder = 'output/TESTING_RESULTS_360degree_ANS_NAVMESH_MAP_1000STEPS'
#output_folder = 'output/TESTING_RESULTS_360degree_FME_NAVMESH_MAP_1STEP_1000STEPS'
#output_folder = 'output/TESTING_RESULTS_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS'
#output_folder = 'output/TESTING_RESULTS_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS'
output_folder = 'output/TESTING_RESULTS_360degree_DP_NAVMESH_MAP_View_Potential_D_Skeleton_Dall_1STEP_500STEPS'
NUM_STEPS = 500


thresh_percent = 0#.3
thresh_steps = 0#150


result_path = f'output/step_cov_curves/temp_{output_folder[7:]}_{NUM_STEPS}STEPS'

all_epi_step_cov = np.zeros((55, NUM_STEPS))

#=============================== basic setup =======================================
split = 'test'
scene_floor_dict = np.load(
	f'output/scene_height_distribution/{split}_scene_floor_dict.npy', allow_pickle=True).item()

count_epi = 0
for env_scene in ['2t7WUuJeko7', '5ZKStnWn8Zo', 'ARNzJeq3xxb', 'RPmz2sHmrrY', 'UwV83HsGsw3', 'Vt2qJdWjCF2', 'WYY7iVyf5p8', 'YFuZgdQ5vWj', 'YVUC4YcDtcY', 'fzynW3qQPVF', 'gYvKGZ5eRqb', 'gxdoqLR6rwA', 'jtcxE69GiFV', 'pa4otMbVnkk', 'q9vSo1VnCiC', 'rqfALeAoiTq', 'wc2JMjhGNzB', 'yqstnuAEVhm']:
	if env_scene == 'WYY7iVyf5p8':
		continue
	scene_dict = scene_floor_dict[env_scene]
	for floor_id in list(scene_dict.keys()):
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'
		#try: 

		#results_step_npy = np.load(f'{output_folder}/results_{scene_name}_step_cov_pairs.npy', allow_pickle=True).item()
		results_step_npy = np.load(f'{output_folder}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test = len(results_step_npy.keys())

		cov_list = []
		

		for i in range(num_test):
			step_cov_pairs = results_step_npy[i]['step_cov_pairs']
			if step_cov_pairs is not None:
				num_steps = step_cov_pairs.shape[0]

				if num_steps > thresh_steps and step_cov_pairs[-1][1] > thresh_percent:
					# compute coverage
					if num_steps < NUM_STEPS:
						all_epi_step_cov[count_epi, :] = step_cov_pairs[num_steps-1][1]
						for j in range(num_steps):
							all_epi_step_cov[count_epi, j] = step_cov_pairs[j][1]
					else:
						for j in range(NUM_STEPS):
							all_epi_step_cov[count_epi, j] = step_cov_pairs[j][1]

					count_epi += 1

			

			#print(f'scene_name = {scene_name}, avg_percent = {avg_percent}, avg_step = {avg_step}')


		#except:
		#	print(f'failed to process scene {scene_name}.')

print('=========================================================================================')

all_epi_step_cov = all_epi_step_cov[:count_epi]

avg_epi_step_cov = np.mean(all_epi_step_cov, axis=0)

np.save(f'{result_path}.npy', avg_epi_step_cov)
