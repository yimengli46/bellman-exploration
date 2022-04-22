import numpy as np 
from core import cfg

scene_list = ['yqstnuAEVhm_0']

avg_percent_list = []
avg_step_list = []

for scene_name in scene_list:
	try:
		output_folder = 'output/TESTING_RESULTS_360degree_Greedy_Frontier_Occupancy'

		results_npy = np.load(f'{output_folder}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test = len(results_npy.keys())

		percent_list = []
		step_list = []

		for i in range(num_test):
			result = results_npy[i]
			print(f'result = {result}')
			flag_suc = result['flag']
			if flag_suc:
				percent = result['covered_area']
				percent_list.append(percent)
				step = result['steps']
				step_list.append(step)

		percent_list = np.array(percent_list)
		print(f'percent_list = {percent_list}')
		avg_percent = np.sum(percent_list) / percent_list.shape[0]

		step_list = np.array(step_list)
		print(f'step_list = {step_list}')
		avg_step = np.sum(step_list) / step_list.shape[0]

		print(f'scene_name = {scene_name}, avg_percent = {avg_percent}, avg_step = {avg_step}')

		if avg_percent > 0:
			avg_percent_list.append(avg_percent)
			avg_step_list.append(avg_step)
	except:
		print(f'failed to process scene {scene_name}.')

print('=========================================================================================')
avg_percent_list = np.array(avg_percent_list)
avg_step_list = np.array(avg_step_list)
print(f'avg percent = {np.mean(avg_percent_list)}, avg step = {np.mean(avg_step_list)}')