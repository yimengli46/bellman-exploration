import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx
from random import Random
from timeit import default_timer as timer
from itertools import islice
import os
from timeit import default_timer as timer
import multiprocessing
import pickle

def get_region(robot_pos, H, W, size=2):
	y, x = robot_pos
	y1 = max(0, y-size)
	y2 = min(H-1, y+size)
	x1 = max(0, x-size)
	x2 = min(W-1, x+size)

	return (y1, x1, y2, x2)

class Data_Gen_MP3D:
	'''
		generate partial map training data for each MP3D scene.
	'''
	def __init__(self, split, scene_name, saved_dir=''):
		self.split = split
		self.scene_name = scene_name
		self.random = Random(cfg.GENERAL.RANDOM_SEED)

		#============= create scene folder =============
		scene_folder = f'{saved_dir}/{scene_name}'
		if not os.path.exists(scene_folder):
			os.mkdir(scene_folder)
		self.scene_folder = scene_folder
	
		self.init_scene()
		
	def init_scene(self):
		scene_name = self.scene_name
		print(f'init new scene: {scene_name}')

		#================================= read in pre-built occupancy and semantic map =============================
		sem_map_npy = np.load(f'{cfg.SAVE.SEMANTIC_MAP_PATH}/{self.split}/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
		self.gt_sem_map, self.pose_range, self.coords_range, self.WH = read_map_npy(sem_map_npy)
		occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{self.split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
		gt_occ_map, _, _, _ = read_occ_map_npy(occ_map_npy)

		gt_occupancy_map = gt_occ_map.copy()
		gt_occupancy_map = np.where(gt_occupancy_map == 1, cfg.FE.FREE_VAL, gt_occupancy_map)  # free cell
		self.gt_occupancy_map = np.where(gt_occupancy_map == 0, cfg.FE.COLLISION_VAL, gt_occupancy_map)  # occupied cell

		self.M_c = np.stack((self.gt_occupancy_map, self.gt_sem_map))
		self.H, self.W = self.gt_sem_map.shape

		# initialize path planner
		self.LN = localNav_Astar(self.pose_range, self.coords_range, self.WH, scene_name)

		# find the largest connected component on the map
		self.G = self.LN.get_G_from_map(gt_occupancy_map)
		self.largest_cc = list(max(nx.connected_components(self.G), key=len))

	def write_to_file(self, num_samples=100):
		count_sample = 0
		#=========================== process each episode
		for idx_epi in range(num_samples):
			print(f'idx_epi = {idx_epi}')

			#====================================== generate (start, goal) locs, compute path P==========================
			start_loc, goal_loc = self.random.choices(self.largest_cc, k=2)
			path = nx.shortest_path(self.G,
									source=start_loc,
									target=goal_loc)

			M_p = np.zeros(self.M_c.shape, dtype=np.int16)
			observed_area_flag = np.zeros((self.H, self.W), dtype=bool)
			i_loc = 0

			while i_loc < len(path):
				robot_loc = path[i_loc]

				t0 = timer()
				#=================================== generate partial map M_p ==================================
				roi = get_region(robot_loc, self.H, self.W, size=cfg.PRED.PARTIAL_MAP.NEIGHBOR_SIZE)
				M_p[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1] = self.M_c[:, roi[0]:roi[2]+1, roi[1]:roi[3]+1]
				observed_area_flag[roi[0]:roi[2]+1, roi[1]:roi[3]+1] = True
				t1 = timer()
				print(f't1 - t0 = {t1 - t0}')

				if i_loc % cfg.PRED.PARTIAL_MAP.STEP_GAP == 0:
					#================================= compute area at frontier points ========================
					U_a = np.zeros((self.H, self.W), dtype=np.float32)
					observed_occupancy_map = M_p[0]
					frontiers = fr_utils.get_frontiers(observed_occupancy_map, self.gt_occupancy_map, observed_area_flag, None)
					agent_map_pose = (robot_loc[1], robot_loc[0])
					frontiers = self.LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, observed_occupancy_map)

					for fron in frontiers:
						points = fron.points.transpose() # N x 2
						R = 1. * fron.R / cfg.PRED.PARTIAL_MAP.DIVIDE_AREA
						U_a[points[:, 0], points[:, 1]] = R
					t2 = timer()
					print(f't2 - t1 = {t2 - t1}')

					#=================================== visualize M_p =========================================
					if cfg.PRED.PARTIAL_MAP.FLAG_VISUALIZE_PRED_LABELS:
						occ_map_Mp = M_p[0]
						sem_map_Mp = M_p[1]
						color_sem_map_Mp = apply_color_to_map(sem_map_Mp)

						fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
						ax[0][0].imshow(occ_map_Mp, cmap='gray')
						ax[0][0].get_xaxis().set_visible(False)
						ax[0][0].get_yaxis().set_visible(False)
						ax[0][0].set_title('input: occupancy_map_Mp')
						ax[0][1].imshow(color_sem_map_Mp)
						ax[0][1].get_xaxis().set_visible(False)
						ax[0][1].get_yaxis().set_visible(False)
						ax[0][1].set_title('input: semantic_map_Mp')

						ax[1][0].imshow(occ_map_Mp, cmap='gray')
						x_coord_lst = [path[i][1] for i in range(i_loc+1)]
						z_coord_lst = [path[i][0] for i in range(i_loc+1)]
						ax[1][0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
						for f in frontiers:
							ax[1][0].scatter(f.points[1], f.points[0], c='yellow', zorder=2)
							ax[1][0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
						ax[1][0].get_xaxis().set_visible(False)
						ax[1][0].get_yaxis().set_visible(False)
						ax[1][0].set_title('observed_occ_map + frontiers')

						ax[1][1].imshow(U_a, vmin=0.0)
						ax[1][1].get_xaxis().set_visible(False)
						ax[1][1].get_yaxis().set_visible(False)
						ax[1][1].set_title('output: U_a')

						fig.tight_layout()
						plt.show()

					# =========================== save data =========================
					eps_data = {}
					eps_data['Mp'] = M_p.copy()
					eps_data['Ua'] = U_a.copy()

					sample_name = str(count_sample).zfill(len(str(num_samples)))
					np.save(f'{self.scene_folder}/{sample_name}.npy', eps_data)
					with open(f'{self.scene_folder}/{sample_name}.pkl', 'wb') as pk_file:
						pickle.dump(obj=frontiers, file=pk_file)
					
					#===================================================================
					count_sample += 1

					if count_sample == num_samples:
						return

					t3 = timer()
					print(f't3 - t2 = {t3 - t2}')

				i_loc += 1

def multi_run_wrapper(args):
	""" wrapper for multiprocessor """
	gen = Data_Gen_MP3D(args[0], args[1], saved_dir=args[2])
	gen.write_to_file(num_samples=cfg.PRED.PARTIAL_MAP.NUM_GENERATED_SAMPLES_PER_SCENE)


if __name__ == "__main__":
	cfg.merge_from_file('configs/train_input_partial_map.yaml')
	cfg.freeze()

	SEED = cfg.GENERAL.RANDOM_SEED
	random.seed(SEED)
	np.random.seed(SEED)

	split = 'train'
	if split == 'train':
		scene_list = cfg.MAIN.TRAIN_SCENE_LIST
	elif split == 'val':
		scene_list = cfg.MAIN.VAL_SCENE_LIST
	elif split == 'test':
		scene_list = cfg.MAIN.TEST_SCENE_LIST
		
	output_folder = 'output/training_data_input_partial_map'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	split_folder = f'{output_folder}/{split}'
	if not os.path.exists(split_folder):
		os.mkdir(split_folder)

	if False: # single process
		for scene in scene_list: 
			gen = Data_Gen_MP3D(split, scene, saved_dir=split_folder)
			gen.write_to_file(num_samples=cfg.PRED.PARTIAL_MAP.NUM_GENERATED_SAMPLES_PER_SCENE)
	else:
		with multiprocessing.Pool(processes=4) as pool:
			args0 = [split for _ in range(len(scene_list))]
			args1 = [scene for scene in scene_list]
			args2 = [split_folder for _ in range(len(scene_list))]
			pool.map(multi_run_wrapper, list(zip(args0, args1, args2)))
			pool.close()