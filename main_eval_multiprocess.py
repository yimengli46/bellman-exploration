import numpy as np
from modeling.frontier_explore import nav
from modeling.utils.baseline_utils import create_folder
import habitat
import habitat_sim
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg
import argparse
import multiprocessing

def build_env(env_scene):
	#================================ load habitat env============================================
	config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
	config.defrost()
	#config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH
	config.SIMULATOR.SCENE = f'{cfg.GENERAL.HABITAT_SCENE_DATA_PATH}/mp3d/{env_scene}/{env_scene}.glb'
	config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
	config.freeze()
	env = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)
	return env

def nav_test(env_scene, output_folder, scene_floor_dict):
	floor_id = 0
	split = 'test'
	scene_dict = scene_floor_dict[env_scene]
	height = scene_dict[0]['y']
	scene_name = f'{env_scene}_{floor_id}'

	print(f'**********scene_name = {scene_name}***********')
	scene_output_folder = f'{output_folder}/{scene_name}'
	create_folder(scene_output_folder)

	#================ initialize habitat env =================
	env = build_env(env_scene)
	env.reset()

	#================ load testing data ==================
	testing_data = scene_dict[floor_id]['start_pose']
	if not cfg.EVAL.USE_ALL_START_POINTS:
		if len(testing_data) > 3:
			testing_data = testing_data[:3]

	results = {}
	for idx, data in enumerate(testing_data):
		#for idx in range(1, 2):
		data = testing_data[idx]
		print(
			f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
		)
		start_pose = data
		print(f'start_pose = {start_pose}')
		saved_folder = f'{scene_output_folder}/eps_{idx}'
		create_folder(saved_folder, clean_up=False)
		flag = False
		steps = 0
		covered_area_percent = 0
		#try:
		flag, covered_area_percent, steps = nav(split, env, idx, scene_name, height, start_pose, saved_folder)
			#print(f'EEEEEEEEEEEEEEEEEEE {scene_name} EPS {idx} FFFFFFFFFFFFFFFF')
		#except:
		#	print(f'CCCCCCCCCCCCCC failed {scene_name} EPS {idx} DDDDDDDDDDDDDDD')

		result = {}
		result['eps_id'] = idx
		result['steps'] = steps
		result['covered_area'] = covered_area_percent
		result['flag'] = flag

		results[idx] = result

	np.save(f'{output_folder}/results_{scene_name}.npy', results)

	env.close()

def multi_run_wrapper(args):
	""" wrapper for multiprocessor """
	nav_test(args[0], args[1], args[2])

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--config',
						type=str,
						required=False,
						default='exp_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_500STEPS.yaml')
	args = parser.parse_args()

	cfg.merge_from_file(f'configs/{args.config}')
	cfg.freeze()

	#=============================== basic setup =======================================
	split = 'test'
	scene_floor_dict = np.load(
		f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
		allow_pickle=True).item()

	output_folder = cfg.SAVE.LARGE_TESTING_RESULTS_FOLDER
	create_folder(output_folder)

	with multiprocessing.Pool(processes=cfg.MP.GPU_CAPACITY) as pool:
		args0 = cfg.MAIN.TEST_SCENE_NO_FLOOR_LIST
		args1 = [output_folder for _ in range(len(args0))]
		args2 = [scene_floor_dict for _ in range(len(args0))]
		pool.map(multi_run_wrapper, list(zip(args0, args1, args2)))
		pool.close()

if __name__ == "__main__":
	main()
