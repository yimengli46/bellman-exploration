import numpy as np 
from core import cfg
import pandas as pd
from modeling.utils.baseline_utils import read_map_npy
import matplotlib.pyplot as plt

#yaml_file = 'exp_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_1000STEPS.yaml'
#yaml_file = 'exp_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS.yaml'
#yaml_file = 'exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
yaml_file = 'exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_1000STEPS.yaml'
#yaml_file = 'exp_90degree_FME_NAVMESH_MAP_1STEP_1000STEPS.yaml'
#yaml_file = 'exp_90degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS.yaml'
#yaml_file = 'exp_90degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
#yaml_file = 'exp_90degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_1000STEPS.yaml'
cfg.merge_from_file(f'configs/{yaml_file}')
cfg.freeze()

avg_percent_list = []
avg_step_list = []
thresh_percent = .0
thresh_steps = 0

df = pd.DataFrame(columns=['Scene', 'Run', 'Num_steps', 'Coverage_500steps', 'Coverage_1000steps', 'MaxSteps', 'Scene_Area'])
df['Num_steps'] = df['Num_steps'].astype(int)
df['Coverage_500steps'] = df['Coverage_500steps'].astype(float)
df['Coverage_1000steps'] = df['Coverage_1000steps'].astype(float)
df['MaxSteps'] = df['MaxSteps'].astype(int)

result_folder = cfg.SAVE.TESTING_RESULTS_FOLDER[7:]

#=============================== basic setup =======================================
split = 'test'
if cfg.EVAL.SIZE == 'small':
	scene_floor_dict = np.load(
		f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
		allow_pickle=True).item()
elif cfg.EVAL.SIZE == 'large':
	scene_floor_dict = np.load(
		f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/large_scale_{split}_scene_floor_dict.npy',
		allow_pickle=True).item()

if cfg.EVAL.SIZE == 'small':
	output_folder = cfg.SAVE.TESTING_RESULTS_FOLDER
elif cfg.EVAL.SIZE == 'large':
	output_folder = cfg.SAVE.LARGE_TESTING_RESULTS_FOLDER

for env_scene in cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST:
	if env_scene == 'WYY7iVyf5p8':
		continue
	scene_dict = scene_floor_dict[env_scene]
	for floor_id in list(scene_dict.keys()):
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'
		try: 
			#========================== load the scene map===========================
			if cfg.EVAL.SIZE == 'small':
				sem_map_npy = np.load(
					f'output/semantic_map/test/{scene_name}/BEV_semantic_map.npy',
					allow_pickle=True).item()
			elif cfg.EVAL.SIZE == 'large':
				sem_map_npy = np.load(
					f'output/large_scale_semantic_map/{scene_name}/BEV_semantic_map.npy',
					allow_pickle=True).item()
			semantic_map, pose_range, coords_range, WH = read_map_npy(sem_map_npy)
			occ_map = semantic_map > 0
			entire_area = np.sum(occ_map) * .0025
			#results_npy = np.load(f'{output_folder}/results_{scene_name}_step_cov_pairs.npy', allow_pickle=True).item()
			results_step_npy = np.load(f'{output_folder}/results_{scene_name}.npy', allow_pickle=True).item()
			num_test = len(results_step_npy.keys())

			cov_list = []
			maxStep_list = []

			for i in range(num_test):
				step_cov_pairs = results_step_npy[i]['step_cov_pairs']
				if step_cov_pairs is not None:
					num_steps = step_cov_pairs.shape[0]
					
					# compute coverage
					if num_steps <= 500:
						cov = step_cov_pairs[num_steps-1, 1]
					else:
						cov = step_cov_pairs[500, 1]

					cov_final = step_cov_pairs[-1, 1]

					# compute MaxSteps
					if step_cov_pairs[-1, 1] >= 0.95:
						percent_arr = step_cov_pairs[:, 1]
						max_steps = np.argmax(percent_arr >= 0.95)
					else:
						max_steps = 1000

					# compute area
					'''
					last_percent = step_cov_pairs[-1, 1]
					last_area = step_cov_pairs[-1, 2]
					entire_area = last_area * 1. / last_percent
					'''

					#if num_steps < 1000 and cov_final < 0.95:
					#	pass
					#else:
					df = df.append({'Scene': scene_name, 'Run': i, 'Num_steps': num_steps, 'Coverage_500steps': cov, 'Coverage_1000steps': cov_final, 'MaxSteps': max_steps, 'Scene_Area': entire_area}, 
							ignore_index=True)
				#else:
				#	df = df.append({'Scene': scene_name, 'Run': i, 'Num_steps': np.nan, 'Coverage': np.nan, 'Scene_Area': area}, 
				#		ignore_index=True)


				#print(f'scene_name = {scene_name}, avg_percent = {avg_percent}, avg_step = {avg_step}')


		except:
			print(f'failed to process scene {scene_name}.')

print('=========================================================================================')
#avg_percent_list = np.array(avg_percent_list)
#avg_step_list = np.array(avg_step_list)
#print(f'avg percent = {np.mean(avg_percent_list)}, avg step = {np.mean(avg_step_list)}')


#=================================== write df to html ======================================
html = df.to_html()
  
# write html to file
html_f = open(f'output/EVAL_results/{result_folder}.html', "w")
html_f.write(f'<h5>All data</h5>')
html_f.write(html)

#==================================== clean up df ===========================
df2 = df.dropna()
# ignore coverage lower than 0.2
filt1 = df2['Coverage_500steps'] >= thresh_percent
df2 = df2[filt1]
filt2 = df2['Num_steps'] >= thresh_steps
df2 = df2[filt2]

#============================= compute data by scene ===============================
scene_grp = df2.groupby(['Scene'])
scene_step = scene_grp['Num_steps'].mean()
scene_cov = scene_grp['Coverage_500steps'].mean()
scene_cov_final = scene_grp['Coverage_1000steps'].mean()
scene_maxsteps = scene_grp['MaxSteps'].mean()

scene_info = pd.concat([scene_step, scene_cov, scene_maxsteps], axis='columns', sort=False)
scene_info.rename(columns={'Num_steps': 'Avg_Num_steps', 'Coverage_500steps': 'Avg_Coverage_500steps', 'Coverage_1000steps': 'Avg_Coverage_1000steps', 'MaxSteps': 'Avg_MaxSteps'}, inplace=True)

#================================ write df to html ==========================================
html = scene_info.to_html()
html_f.write(f'<h5>Group by each scene</h5>')
html_f.write(html)


#=========================== group results by area size ===================================
df2['Area_Type'] = 'small'
df2.loc[df2['Scene_Area'] > 200, 'Area_Type'] = 'medium'
df2.loc[df2['Scene_Area'] > 500, 'Area_Type'] = 'large'

scene_grp = df2.groupby(['Area_Type'])
scene_step = scene_grp['Num_steps'].mean()
scene_cov = scene_grp['Coverage_500steps'].mean()
scene_cov_final = scene_grp['Coverage_1000steps'].mean()
scene_maxsteps = scene_grp['MaxSteps'].mean()

scene_info = pd.concat([scene_step, scene_cov, scene_maxsteps], axis='columns', sort=False)
scene_info.rename(columns={'Num_steps': 'Avg_Num_steps', 'Coverage_500steps': 'Avg_Coverage_500steps', 'Coverage_1000steps': 'Avg_Coverage_1000steps', 'MaxSteps': 'Avg_MaxSteps'}, inplace=True)

#================================ write df to html ==========================================
html = scene_info.to_html()
html_f.write(f'<h5>Group by the area</h5>')
html_f.write(html)

#=========================== group results by area size ===================================
df2['Area_Type'] = 'all'

scene_grp = df2.groupby(['Area_Type'])
scene_step = scene_grp['Num_steps'].mean()
scene_cov = scene_grp['Coverage_500steps'].mean()
scene_cov_final = scene_grp['Coverage_1000steps'].mean()
scene_maxsteps = scene_grp['MaxSteps'].mean()

scene_info = pd.concat([scene_step, scene_cov, scene_maxsteps], axis='columns', sort=False)
scene_info.rename(columns={'Num_steps': 'Avg_Num_steps', 'Coverage_500steps': 'Avg_Coverage_500steps', 'Coverage_1000steps': 'Avg_Coverage_1000steps', 'MaxSteps': 'Avg_MaxSteps'}, inplace=True)

#================================ write df to html ==========================================
html = scene_info.to_html()
html_f.write(f'<h5>Group by all</h5>')
html_f.write(html)
html_f.close()