import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn, convertInsSegToSSeg
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx
import torch.utils.data as data
import torch
import torch.nn.functional as F
from random import Random
from timeit import default_timer as timer
from itertools import islice
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name, get_obs_and_pose
from modeling.utils.map_utils_for_training import SemanticMap
import habitat

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

class MP3D_View_Generator:

	def __init__(self, split, scene_names):

		self.split = split
		self.scene_names = scene_names

		#============================= initialize habitat env===================================
		self.scene_floor_dict = np.load(
			f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{self.split}_scene_floor_dict.npy',
			allow_pickle=True).item()

		#================================ load habitat env============================================
		config = habitat.get_config(config_paths=cfg.GENERAL.DATALOADER_CONFIG_PATH)
		config.defrost()
		if self.split == 'train':
			config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TRAIN_EPISODE_DATA_PATH 
		elif self.split == 'val':
			config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_VAL_EPISODE_DATA_PATH

		config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
		config.freeze()
		self.env = SimpleRLEnv(config=config)
		
	def generate(self):
		for episode_id in range(len(self.scene_names)):
			self.env.reset()
			print('episode_id = {}'.format(episode_id))
			print('env.current_episode = {}'.format(self.env.current_episode))

			env_scene = get_scene_name(self.env.current_episode)
			scene_dict = self.scene_floor_dict[env_scene]

			for floor_id in list(scene_dict.keys()):
				height = scene_dict[floor_id]['y']
				scene_name = f'{env_scene}_{floor_id}'

				if scene_name in self.scene_names:
					scene_height = height

					#============================ get scene ins to cat dict =====================
					scene = self.env.habitat_env.sim.semantic_annotations()
					ins2cat_dict = {
						int(obj.id.split("_")[-1]): obj.category.index()
						for obj in scene.objects
					}

					#======================================== get occ map ============================================
					occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{self.split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
					gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
					gt_occupancy_map = gt_occ_map.copy()
					gt_occupancy_map = np.where(gt_occupancy_map == 1, cfg.FE.FREE_VAL, gt_occupancy_map)  # free cell
					gt_occupancy_map = np.where(gt_occupancy_map == 0, cfg.FE.COLLISION_VAL, gt_occupancy_map)  # occupied cell

					LN = localNav_Astar(pose_range, coords_range, WH, scene_name)
					G = LN.get_G_from_map(gt_occupancy_map)
					largest_cc = list(max(nx.connected_components(G), key=len))

					#======================================= randomly generate episodes ======================
					for idx_epi in range(cfg.PRED.NUM_GENERATE_EPISODES_PER_SCENE):
						start_loc, goal_loc = random.choices(largest_cc, k=2)
						path = nx.shortest_path(G,
												source=start_loc,
												target=goal_loc)
				
						semMap_module = SemanticMap(self.split, scene_name, pose_range, coords_range, WH, ins2cat_dict)  # build the observed sem map
						poses = LN.convert_path_to_pose(path)

						#============================== start navigation ==========================
						current_frontiers = set()
						for i_pose in range(len(poses)):

							next_pose = poses[i_pose]
							print(f'next_pose = {next_pose}')

							#=============================== get obs ========================================
							print(f'height = {scene_height}')
							agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
							# output rot is negative of the input angle
							if cfg.NAVI.HFOV == 90:
								obs_list, pose_list = [], []
								heading_angle = -next_pose[2]
								obs, pose = get_obs_and_pose(self.env, agent_pos, heading_angle)
								obs_list.append(obs)
								pose_list.append(pose)
							elif cfg.NAVI.HFOV == 360:
								obs_list, pose_list = [], []
								for rot in [90, 180, 270, 0]:
									heading_angle = rot / 180 * np.pi
									heading_angle = plus_theta_fn(heading_angle, -next_pose[2])
									obs, pose = get_obs_and_pose(self.env, agent_pos, heading_angle)
									obs_list.append(obs)
									pose_list.append(pose)

							# get panorama observation
							rgb_lst, depth_lst, sseg_lst = [], [], []
							for idx, obs in enumerate(obs_list):
								# load rgb image, depth and sseg
								rgb_img = obs['rgb']
								depth_img = 5. * obs['depth']
								depth_img = cv2.blur(depth_img, (3, 3))
								InsSeg_img = obs["semantic"]
								sseg_img = convertInsSegToSSeg(InsSeg_img, ins2cat_dict)
								rgb_lst.append(rgb_img)
								depth_lst.append(depth_img)
								sseg_lst.append(sseg_img)

								if False:
									print(f'idx = {idx}')
									fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
									ax[0].imshow(rgb_img)
									ax[0].get_xaxis().set_visible(False)
									ax[0].get_yaxis().set_visible(False)
									ax[0].set_title("rgb")
									ax[1].imshow(apply_color_to_map(sseg_img))
									ax[1].get_xaxis().set_visible(False)
									ax[1].get_yaxis().set_visible(False)
									ax[1].set_title("sseg")
									ax[2].imshow(depth_img)
									ax[2].get_xaxis().set_visible(False)
									ax[2].get_yaxis().set_visible(False)
									ax[2].set_title("depth")
									fig.tight_layout()
									plt.show()

							panorama_rgb = np.concatenate(rgb_lst, axis=1)
							panorama_depth = np.concatenate(depth_lst, axis=1)
							panorama_sseg = np.concatenate(sseg_lst, axis=1)

							if True:
								fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 6))
								ax[0].imshow(panorama_rgb)
								ax[0].get_xaxis().set_visible(False)
								ax[0].get_yaxis().set_visible(False)
								ax[0].set_title("rgb")
								ax[1].imshow(apply_color_to_map(panorama_sseg))
								ax[1].get_xaxis().set_visible(False)
								ax[1].get_yaxis().set_visible(False)
								ax[1].set_title("sseg")
								ax[2].imshow(panorama_depth)
								ax[2].get_xaxis().set_visible(False)
								ax[2].get_yaxis().set_visible(False)
								ax[2].set_title("depth")
								fig.tight_layout()
								plt.show()
							#assert 1==2

							#=============================== update map =================================
							semMap_module.build_semantic_map(obs_list, pose_list)
							observed_occupancy_map, gt_occupancy_map, observed_area_flag, _ = semMap_module.get_observed_occupancy_map()

							plt.imshow(observed_occupancy_map)
							plt.show()
							#============================= find new frontiers ==============================
							frontiers = fr_utils.get_frontiers(observed_occupancy_map, gt_occupancy_map, observed_area_flag, None)
							pose = pose_list[-1]
							agent_map_pose = (pose[0], -pose[1], -pose[2])

							frontiers = self.LN.filter_unreachable_frontiers_temp(frontiers, agent_map_pose, observed_occupancy_map)

							assert 1==2



#'''
device = torch.device('cuda')

scene_name = '17DRP5sb8fy_0'

data = MP3D_View_Generator(split='train', scene_names=[scene_name])
data.generate()
#'''