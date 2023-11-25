import numpy as np 
import matplotlib.pyplot as plt

output_folder = 'output/step_cov_curves'

# load ANS
arr_ANS = np.load(f'{output_folder}/temp_TESTING_RESULTS_360degree_ANS_NAVMESH_MAP_1000STEPS_500STEPS.npy', allow_pickle=True)
arr_ANS = np.insert(arr_ANS, 0, 0)

# load FME
arr_FME = np.load(f'{output_folder}/temp_TESTING_RESULTS_360degree_FME_NAVMESH_MAP_1STEP_1000STEPS_500STEPS.npy', allow_pickle=True)
arr_FME = np.insert(arr_FME, 0, 0)

# load Greedy
arr_Greedy = np.load(f'{output_folder}/temp_TESTING_RESULTS_360degree_DP_NAVMESH_MAP_View_Potential_D_Skeleton_Dall_1STEP_500STEPS_500STEPS.npy', allow_pickle=True)
arr_Greedy = np.insert(arr_Greedy, 0, 0)

# load UNet
arr_UNet = np.load(f'{output_folder}/temp_TESTING_RESULTS_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS_500STEPS.npy', allow_pickle=True)
arr_UNet = np.insert(arr_UNet, 0, 0)

# load View
arr_View = np.load(f'{output_folder}/temp_TESTING_RESULTS_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS_500STEPS.npy', allow_pickle=True)
arr_View = np.insert(arr_View, 0, 0)

t = np.arange(0, 500, 1)
s1 = arr_ANS[:500]
s2 = arr_FME[:500]
s3 = arr_Greedy[:500]
s4 = arr_UNet[:500]
s5 = arr_View[:500]

fig, ax = plt.subplots()
ax.plot(t, s1)
ax.plot(t, s2)
ax.plot(t, s3)
ax.plot(t, s4)
ax.plot(t, s5)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

#fig.savefig("test.png")
plt.show()
