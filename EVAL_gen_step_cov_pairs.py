import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor, degrees
import random
from modeling.utils.navigation_utils import change_brightness, SimpleRLEnv, get_obs_and_pose, get_obs_and_pose_by_action
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn
from modeling.utils.map_utils_pcd_height import SemanticMap
from modeling.localNavigator_Astar import localNav_Astar
import habitat
import habitat_sim
import random
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from timeit import default_timer as timer
import argparse
import multiprocessing
import os

def build_env(env_scene, device_id=0):
	#================================ load habitat env============================================
	config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
	config.defrost()
	#config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH
	config.SIMULATOR.SCENE = f'{cfg.GENERAL.HABITAT_SCENE_DATA_PATH}/mp3d/{env_scene}/{env_scene}.glb'
	config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
	config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = device_id
	config.freeze()
	env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
	return env

def compute_step_cov_pairs(split, env, scene_name, scene_height, traverse_lst):
	#============================ get scene ins to cat dict
	scene = env.semantic_annotations()
	ins2cat_dict = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

	if cfg.NAVI.GT_OCC_MAP_TYPE == 'NAV_MESH':
		if cfg.EVAL.SIZE == 'small':
			occ_map_npy = np.load(
				f'output/semantic_map/{split}/{scene_name}/BEV_occupancy_map.npy',
				allow_pickle=True).item()
		elif cfg.EVAL.SIZE == 'large':
			occ_map_npy = np.load(
				f'output/large_scale_semantic_map/{scene_name}/BEV_occupancy_map.npy',
				allow_pickle=True).item()
	
	gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
	H, W = gt_occ_map.shape[:2]

	LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

	# ================= get the area connected with start pose ==============
	gt_reached_area = LN.get_start_pose_connected_component(traverse_lst[0], gt_occ_map)
	#plt.imshow(gt_reached_area)
	#plt.show()

	semMap_module = SemanticMap(split, scene_name, pose_range, coords_range, WH, ins2cat_dict) # build the observed sem map
	
	step_cov_pairs = []

	for idx_step, agent_map_pose in enumerate(traverse_lst):
		agent_env_pose = (agent_map_pose[0], -agent_map_pose[1], -agent_map_pose[2])
		agent_pos = np.array([agent_env_pose[0], scene_height, agent_env_pose[1]])

		if cfg.NAVI.HFOV == 90:
			obs_list, pose_list = [], []
			heading_angle = agent_env_pose[2]
			obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)
			obs_list.append(obs)
			pose_list.append(pose)
		elif cfg.NAVI.HFOV == 360:
			obs_list, pose_list = [], []
			for rot in [90, 180, 270, 0]:
				heading_angle = rot / 180 * np.pi
				heading_angle = plus_theta_fn(heading_angle, agent_env_pose[2])
				obs, pose = get_obs_and_pose(env, agent_pos, heading_angle)
				obs_list.append(obs)
				pose_list.append(pose)

		semMap_module.build_semantic_map(obs_list, pose_list)
		
		observed_occupancy_map, gt_occupancy_map, observed_area_flag, built_semantic_map = semMap_module.get_observed_occupancy_map(agent_map_pose)

		if False:
			#============================ visualize built map ========================
			x_coord_lst, z_coord_lst, theta_lst = [], [], []
			for cur_pose in traverse_lst[0:idx_step+1]:
				x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
				x_coord_lst.append(x_coord)
				z_coord_lst.append(z_coord)			
				theta_lst.append(cur_pose[2])

			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
			ax[0].imshow(observed_occupancy_map, cmap='gray')
			marker, scale = gen_arrow_head_marker(theta_lst[-1])
			ax[0].scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=5)
			ax[0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			#ax.set_title('improved observed_occ_map + frontiers')

			color_built_semantic_map = apply_color_to_map(built_semantic_map)
			ax[1].imshow(color_built_semantic_map)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)

			fig.tight_layout()
			plt.title('observed area')
			plt.show()

		#=================== compute percent and explored area ===============================
		# percent is computed on the free space
		explored_free_space = np.logical_and(gt_reached_area, observed_area_flag)
		percent = 1. * np.sum(explored_free_space) / np.sum(gt_reached_area)

		# explored area
		area = np.sum(observed_area_flag) * .0025
		print(f'step = {idx_step}, percent = {percent}, area = {area} meter^2')
		step_cov_pairs.append((idx_step, percent, area))

	step_cov_pairs = np.array(step_cov_pairs).astype('float32')

	return step_cov_pairs

def eval_scene(env_scene, output_folder, scene_floor_dict):
	#============================ get a gpu
	device_id = gpu_Q.get()

	#================ initialize habitat env =================
	env = build_env(env_scene, device_id=device_id)
	env.reset()
	cfg.merge_from_file(f'configs/exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml')
	cfg.freeze()

	scene_dict = scene_floor_dict[env_scene]

	#=============================== traverse each floor ===========================
	for floor_id in list(scene_dict.keys()):
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'
		
		print(f'**********scene_name = {scene_name}***********')

		#=================== load npy file ==========================
		results_npy = np.load(f'{output_folder}/results_{scene_name}.npy', allow_pickle=True).item()
		num_test = len(results_npy.keys())

		results = {}
		for idx in range(num_test):
			result = results_npy[idx]
			trajectory = result['trajectory']
			eps_id = result['eps_id']
			step_cov_pairs = compute_step_cov_pairs(split, env, scene_name, height, trajectory)
			
			new_result = {}
			new_result['eps_id'] = eps_id
			new_result['step_cov_pairs'] = step_cov_pairs

			results[idx] = new_result

		np.save(f'{output_folder}/results_{scene_name}_step_cov_pairs.npy', results)

	env.close()

	#================================ release the gpu============================
	gpu_Q.put(device_id)


def multi_run_wrapper(args):
	""" wrapper for multiprocessor """
	eval_scene(args[0], args[1], args[2])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config',
						type=str,
						required=False,
						default='exp_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_500STEPS.yaml')
	args = parser.parse_args()

	cfg.merge_from_file(f'configs/{args.config}')
	cfg.freeze()

	#====================== get the available GPU devices ============================
	visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
	devices = [int(dev) for dev in visible_devices]

	for device_id in devices:
		for _ in range(1):
			gpu_Q.put(device_id)

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
	create_folder(output_folder)

	args0 = cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST
	with multiprocessing.Pool(processes=len(args0)) as pool:
		args1 = [output_folder for _ in range(len(args0))]
		args2 = [scene_floor_dict for _ in range(len(args0))]
		pool.map(multi_run_wrapper, list(zip(args0, args1, args2)))
		pool.close()
		pool.join()

if __name__ == "__main__":
	gpu_Q = multiprocessing.Queue()
	main()