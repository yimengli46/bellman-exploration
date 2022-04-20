import numpy as np 
from core import cfg

scene_list = cfg.MAIN.SCENE_LIST
for scene_name in scene_list:
	try:
		output_folder = cfg.SAVE.TESTING_RESULTS_FOLDER

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
				percent_list.append(flag_suc)
				step = result['steps']
				step_list.append(step)


		
		percent_list = np.array(percent_list)
		print(f'percent_list = {percent_list}')
		avg_percent = np.sum(percent_list) / percent_list.shape[0]

		print(f'scene_name = {scene_name}, avg_percent = {avg_percent}')
	except:
		print(f'failed to process scene {scene_name}.')


