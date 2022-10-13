import os

folder = 'output/TESTING_RESULTS_Frontier/DP_large'
num_files = 500


for i in range(num_files):
	os.rename(f"{folder}/step_{i}_semmap.png", f"{folder}/step_{i:03}_semmap.png")
	os.rename(f"{folder}/step_{i}_panor.png", f"{folder}/step_{i:03}_panor.png")