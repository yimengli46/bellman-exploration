import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from modeling.utils.baseline_utils import project_pixels_to_world_coords, convertInsSegToSSeg, apply_color_to_map, create_folder
import habitat
import habitat_sim
from modeling.utils.build_map_utils import SemanticMap
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import random
from core import cfg

#=========================================== fix the habitat scene shuffle ===============================
SEED = 5
random.seed(SEED)
np.random.seed(SEED)

output_folder = f'output/semantic_map/{cfg.MAIN.SPLIT}'
# after testing, using 8 angles is most efficient
theta_lst = [0, pi/4, pi/2, pi*3./4, pi, pi*5./4, pi*3./2, pi*7./4]
#theta_lst = [0]
str_theta_lst = ['000', '090', '180', '270']

#============================ load scene heights ===================================
scene_floor_dict = np.load(f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{cfg.MAIN.SPLIT}_scene_floor_dict.npy', allow_pickle=True).item()

#============================= build a grid =========================================
x = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, 0.3)
z = np.arange(-cfg.SEM_MAP.WORLD_SIZE, cfg.SEM_MAP.WORLD_SIZE, 0.3)
xv, zv = np.meshgrid(x, z)
#xv = xv.flatten()
#zv = zv.flatten()
grid_H, grid_W = zv.shape

#============================ traverse each scene =============================
class SimpleRLEnv(habitat.RLEnv):
	def get_reward_range(self):
		return [-1, 1]

	def get_reward(self, observations):
		return 0

	def get_done(self, observations):
		return self.habitat_env.episode_over

	def get_info(self, observations):
		return self.habitat_env.get_metrics()

def get_scene_name(episode):
	idx_right_most_slash = episode.scene_id.rfind('/')
	return episode.scene_id[idx_right_most_slash+1:-4]

def verify_img(img):
    sum_img = np.sum((img[:,:,0] > 0))
    h, w = img.shape[:2]
    return sum_img > h*w*0.75

def convert_depth_to_uint16(depth_img):
	a = depth_img*256.
	return a.astype(np.uint16)


config = habitat.get_config(config_paths=cfg.GENERAL.BUILD_MAP_CONFIG_PATH)
config.defrost()
if cfg.MAIN.SPLIT == 'train':
	config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TRAIN_EPISODE_DATA_PATH 
elif cfg.MAIN.SPLIT == 'val':
	config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_VAL_EPISODE_DATA_PATH
elif cfg.MAIN.SPLIT == 'test':
	config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()
env = SimpleRLEnv(config=config)

for episode_id in range(cfg.MAIN.NUM_SCENES):
	env.reset()
	print('episode_id = {}'.format(episode_id))
	print('env.current_episode = {}'.format(env.current_episode))

	env_scene = get_scene_name(env.current_episode)
	
	scene_dict = scene_floor_dict[env_scene]

	#=============================== traverse each floor ===========================
	for floor_id in list(scene_dict.keys()):
		height = scene_dict[floor_id]['y']
		scene_name = f'{env_scene}_{floor_id}'

		print(f'*****scene_name = {scene_name}***********')

		saved_folder = f'{output_folder}/{scene_name}'
		create_folder(saved_folder, clean_up=False)

		#============================ get scene ins to cat dict
		scene = env.habitat_env.sim.semantic_annotations()
		ins2cat_dict = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

		#================================ Building a map ===============================
		SemMap = SemanticMap(saved_folder)

		count_ = 0
		#========================= generate observations ===========================
		for grid_z in range(grid_H):
			for grid_x in range(grid_W):
				x = xv[grid_z, grid_x]
				z = zv[grid_z, grid_x]
				y = height
				agent_pos = np.array([x, y, z])

				flag_nav = env.habitat_env.sim.is_navigable(agent_pos)
				#print(f'after teleportation, flag_nav = {flag_nav}')

				if flag_nav:
					#==================== traverse theta ======================
					for idx_theta, theta in enumerate(theta_lst):
						agent_rot = habitat_sim.utils.common.quat_from_angle_axis(theta, habitat_sim.geo.GRAVITY)
						observations = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)
						rgb_img = observations["rgb"]
						depth_img = observations['depth'][:, :, 0]
						InsSeg_img = observations["semantic"]
						sseg_img = convertInsSegToSSeg(InsSeg_img, ins2cat_dict)
						#print(f'rgb_img.shape = {rgb_img.shape}')

						'''
						fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(100, 300))

						ax[0].imshow(rgb_img)
						ax[1].imshow(depth_img)
						ax[2].imshow(sseg_img)

						fig.tight_layout()
						plt.show()
						'''

						#=============================== get agent global pose on habitat env ========================#
						agent_pos = env.habitat_env.sim.get_agent_state().position
						agent_rot = env.habitat_env.sim.get_agent_state().rotation
						heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
						phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
						angle = phi
						print(f'agent position = {agent_pos}, angle = {angle}')
						pose = (agent_pos[0], agent_pos[2], angle)

						SemMap.build_semantic_map(rgb_img, depth_img, sseg_img, pose, count_)
						count_ += 1

		SemMap.save_final_map()
