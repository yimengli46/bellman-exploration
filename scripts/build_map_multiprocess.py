import numpy as np
from modeling.utils.baseline_utils import create_folder
import habitat
import habitat_sim
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg
import argparse
import multiprocessing
import os
import torch
from build_semantic_BEV_map_large_scale import build_sem_map
from build_occ_map_from_densely_checking_cells_large_scale import build_occ_map

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

def build_floor(env_scene, output_folder, scene_floor_dict):
	#============================ get a gpu
	device_id = gpu_Q.get()

	#================ initialize habitat env =================
	env = build_env(env_scene, device_id=device_id)
	env.reset()

	scene_dict = scene_floor_dict[env_scene]
	for floor_id in list(scene_dict.keys()):
		split = 'test'
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'

		scene_output_folder = f'{output_folder}/{scene_name}'
		create_folder(scene_output_folder)

		#build_sem_map(env, scene_output_folder, height)
		build_occ_map(env, scene_output_folder, height, scene_name)

	env.close()

	#================================ release the gpu============================
	gpu_Q.put(device_id)

def multi_run_wrapper(args):
	""" wrapper for multiprocessor """
	build_floor(args[0], args[1], args[2])

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
		for _ in range(cfg.MP.PROC_PER_GPU):
			gpu_Q.put(device_id)

	#=============================== basic setup =======================================
	split = 'test'
	scene_floor_dict = np.load(
		f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/large_scale_{split}_scene_floor_dict.npy',
		allow_pickle=True).item()

	output_folder = 'output/large_scale_semantic_map'
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
