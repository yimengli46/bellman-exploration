import numpy as np 
from core import cfg
import pandas as pd

scene_list = cfg.MAIN.TEST_SCENE_LIST
output_folder = 'output' #cfg.SAVE.TESTING_RESULTS_FOLDER
result_folder = 'TESTING_RESULTS_360degree_Greedy_GT_Potential_1STEP_500STEPS'
#scene_list = ['2t7WUuJeko7_0', '5ZKStnWn8Zo_0', 'ARNzJeq3xxb_0', 'RPmz2sHmrrY_0', 'Vt2qJdWjCF2_0', 'WYY7iVyf5p8_0', 'YFuZgdQ5vWj_0', 'YVUC4YcDtcY_0', 'fzynW3qQPVF_0', 'gYvKGZ5eRqb_0', 'gxdoqLR6rwA_0', 'q9vSo1VnCiC_0', 'rqfALeAoiTq_0', 'wc2JMjhGNzB_0', 'yqstnuAEVhm_0']


avg_percent_list = []
avg_step_list = []
thresh_percent = .2


df = pd.DataFrame(columns=['Scene', 'Run', 'Num_steps', 'Coverage'])
df['Num_steps'] = df['Num_steps'].astype(int)
df['Coverage'] = df['Coverage'].astype(float)

for scene_name in scene_list:
	try:

		results_npy = np.load(f'{output_folder}/{result_folder}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test = len(results_npy.keys())

		percent_list = []
		step_list = []

		for i in range(num_test):
			result = results_npy[i]
			print(f'result = {result}')
			flag_suc = result['flag']
			if flag_suc:
				percent = result['covered_area']
				step = result['steps']
				if percent > thresh_percent: # to deal with the bad start points
					percent_list.append(percent)
					step_list.append(step)

				df = df.append({'Scene': scene_name, 'Run': i, 'Num_steps': step, 'Coverage': percent}, 
					ignore_index=True)
			else:
				df = df.append({'Scene': scene_name, 'Run': i, 'Num_steps': np.nan, 'Coverage': np.nan}, 
					ignore_index=True)

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


#=================================== write df to html ======================================
html = df.to_html()
  
# write html to file
html_f = open(f'{output_folder}/pandas_results/{result_folder}.html', "w")
html_f.write(f'<h5>All data</h5>')
html_f.write(html)

#==================================== clean up df ===========================
df2 = df.dropna()
# ignore coverage lower than 0.2
filt = df2['Coverage'] >= thresh_percent
df2 = df2[filt]

#============================= compute data by scene ===============================
scene_grp = df2.groupby(['Scene'])
scene_step = scene_grp['Num_steps'].mean()
scene_cov = scene_grp['Coverage'].mean()

scene_info = pd.concat([scene_step, scene_cov], axis='columns', sort=False)
scene_info.rename(columns={'Num_steps': 'Avg_Num_steps', 'Coverage': 'Avg_Coverage'}, inplace=True)

#================================ write df to html ==========================================
html = scene_info.to_html()
html_f.write(f'<h5>Description by each scene</h5>')
html_f.write(html)
html_f.close()