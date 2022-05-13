from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
import pandas as pd
from core import cfg

scene_list = cfg.MAIN.SCENE_LIST
thresh_percent = .2
data = []

for scene_name in scene_list:
	#try:
		output_folder1 = 'output/TESTING_RESULTS_360degree_Greedy_Potential_10STEP_600STEPS'
		results_npy1 = np.load(f'{output_folder1}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test1 = len(results_npy1.keys())

		output_folder2 = 'output/TESTING_RESULTS_360degree_DP_Potential_10STEP_600STEPS'
		results_npy2 = np.load(f'{output_folder2}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test2 = len(results_npy2.keys())

		assert num_test1 == num_test2

		percent_list = []
		step_list = []

		for i in range(num_test1):
			result1 = results_npy1[i]
			result2 = results_npy2[i]

			flag_suc = result1['flag']
			if flag_suc:
				percent1 = result1['covered_area']
				percent2 = result2['covered_area']
				if percent1 > thresh_percent: # to deal with the bad start points
					step1 = result1['steps']
					step2 = result2['steps']
					data.append([percent1 * 1000. / step1, percent2 * 1000. / step2, f'{scene_name}_{i}'])
					
	#except:
	#	print(f'failed to process scene {scene_name}.')

data = pd.DataFrame(data, columns=['greedy', 'dp', 'name'])

xy = np.vstack([data['greedy'], data['dp']])

z = scipy.stats.gaussian_kde(xy)(xy)
data['zs'] = z
data = data.sort_values(by=['zs'])
z = data['zs']
colors = cm.get_cmap('hot')((z - z.min()) / (z.max() - z.min()))

fig = plt.figure(figsize=(5, 5))
fig.gca()
ax = plt.subplot(111)
ax.scatter(data['greedy'], data['dp'], c='b')
ax.set_aspect('equal', adjustable='box')
cb = min(max(data['greedy']), max(data['dp']))
ax.plot([0, cb], [0, cb], 'k')
xmin = data['greedy'].min()
xmax = data['greedy'].max()
ymin = data['dp'].min()
ymax = data['dp'].max()
ax.set_xlim([0, 8])
ax.set_ylim([0, 8])
ax.set_xlabel('Greedy')
ax.set_ylabel('DP')