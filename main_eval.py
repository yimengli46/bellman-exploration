import numpy as np
from modeling.frontier_explore import nav
from modeling.utils.baseline_utils import create_folder
import habitat
import habitat_sim
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name
from core import cfg

split = 'test'
scene_floor_dict = np.load(
	f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy',
	allow_pickle=True).item()

cfg.merge_from_file('configs/exp_360degree_Greedy_Potential_500STEPS.yaml')
cfg.freeze()

#================================ load habitat env============================================
config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
config.defrost()
config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()
env = SimpleRLEnv(config=config)

for episode_id in range(17):
	env.reset()
	print('episode_id = {}'.format(episode_id))
	print('env.current_episode = {}'.format(env.current_episode))

	env_scene = get_scene_name(env.current_episode)
	scene_dict = scene_floor_dict[env_scene]

	#=============================== traverse each floor ===========================
	for floor_id in list(scene_dict.keys()):
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'

		if scene_name in cfg.MAIN.TEST_SCENE_LIST:
			print(f'**********scene_name = {scene_name}***********')

			output_folder = cfg.SAVE.TESTING_RESULTS_FOLDER
			create_folder(output_folder)
			scene_output_folder = f'{output_folder}/{scene_name}'
			create_folder(scene_output_folder)

			testing_data = scene_dict[floor_id]['start_pose']
			if not cfg.EVAL.USE_ALL_START_POINTS:
				if len(testing_data) > 3:
					testing_data = testing_data[:3]

			#'''
			results = {}
			for idx, data in enumerate(testing_data):
				#for idx in range(1, 2):
				data = testing_data[idx]
				print(
					f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA EPS {idx} BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB'
				)
				start_pose = data
				saved_folder = f'{scene_output_folder}/eps_{idx}'
				create_folder(saved_folder, clean_up=True)
				flag = False
				steps = 0
				covered_area_percent = 0
				try:
					flag, covered_area_percent, steps = nav(split, env, idx, scene_name, height, start_pose, saved_folder)
				except:
					print(f'CCCCCCCCCCCCCC failed EPS {idx} DDDDDDDDDDDDDDD')

				result = {}
				result['eps_id'] = idx
				result['steps'] = steps
				result['covered_area'] = covered_area_percent
				result['flag'] = flag

				results[idx] = result

			np.save(f'{output_folder}/results_{scene_name}.npy', results)
			#'''

env.close()
