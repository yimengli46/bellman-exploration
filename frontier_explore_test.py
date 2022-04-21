import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor, degrees
import random
from navigation_utils import change_brightness, SimpleRLEnv
from baseline_utils import apply_color_to_map, pose_to_coords, gen_arrow_head_marker, read_map_npy, read_occ_map_npy
from map_utils import SemanticMap
from localNavigator_Astar import localNav_Astar
import habitat
import habitat_sim
from habitat.tasks.utils import cartesian_to_polar, quaternion_rotate_vector
import random
from core import cfg
import frontier_utils as fr_utils
from timeit import default_timer as timer

split = 'test'
env_scene = 'yqstnuAEVhm'
floor_id = 0
scene_name = 'yqstnuAEVhm_0'

scene_floor_dict = np.load(f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{split}_scene_floor_dict.npy', allow_pickle=True).item()

#================================ load habitat env============================================
config = habitat.get_config(config_paths=cfg.GENERAL.HABITAT_CONFIG_PATH)
config.defrost()
config.DATASET.DATA_PATH = cfg.GENERAL.HABITAT_EPISODE_DATA_PATH
config.DATASET.SCENES_DIR = cfg.GENERAL.HABITAT_SCENE_DATA_PATH
config.freeze()
env = SimpleRLEnv(config=config)

scene_height = scene_floor_dict[env_scene][floor_id]['y']
start_pose = (0.03828, -8.55, 0.296)
saved_folder = f'output/TESTING_RESULTS_Frontier'

#============================ get scene ins to cat dict
scene = env.habitat_env.sim.semantic_annotations()
ins2cat_dict = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}

#=================================== start original navigation code ========================
np.random.seed(cfg.GENERAL.RANDOM_SEED)
random.seed(cfg.GENERAL.RANDOM_SEED)

if cfg.NAVI.FLAG_GT_OCC_MAP:
	occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
gt_occ_map, pose_range, coords_range, WH = read_occ_map_npy(occ_map_npy)
H, W = gt_occ_map.shape[:2]

LN = localNav_Astar(pose_range, coords_range, WH, scene_name)

semMap_module = SemanticMap(scene_name, pose_range, coords_range, WH, ins2cat_dict) # build the observed sem map
traverse_lst = []

#===================================== setup the start location ===============================#

agent_pos = np.array([start_pose[0], scene_height, start_pose[1]]) # (6.6, -6.9), (3.6, -4.5)
agent_rot = habitat_sim.utils.common.quat_from_angle_axis(start_pose[2], habitat_sim.geo.GRAVITY)
# check if the start point is navigable
if not env.habitat_env.sim.is_navigable(agent_pos):
	print(f'start pose is not navigable ...')
	assert 1==2
obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

step = 0
subgoal_coords = None
subgoal_pose = None 
MODE_FIND_SUBGOAL = True
explore_steps = 0
MODE_FIND_GOAL = False
visited_frontier = []
chosen_frontier = None

while step < cfg.NAVI.NUM_STEPS:
	print(f'step = {step}')

	#=============================== get agent global pose on habitat env ========================#
	agent_pos = env.habitat_env.sim.get_agent_state().position
	agent_rot = env.habitat_env.sim.get_agent_state().rotation
	heading_vector = quaternion_rotate_vector(agent_rot.inverse(), np.array([0, 0, -1]))
	phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
	angle = phi
	print(f'agent position = {agent_pos}, angle = {angle}')
	pose = (agent_pos[0], agent_pos[2], angle)
	agent_map_pose = (pose[0], -pose[1], -pose[2])
	traverse_lst.append(agent_map_pose)

	# add the observed area
	t0 = timer()
	semMap_module.build_semantic_map(obs, pose, step=step, saved_folder=saved_folder)
	t1 = timer()
	print(f'build map time = {t1 - t0}')

	if MODE_FIND_SUBGOAL:
		t1 = timer()
		observed_occupancy_map, gt_occupancy_map, observed_area_flag = semMap_module.get_observed_occupancy_map()
		t2 = timer()
		print(f't2- t1 = {t2 - t1}')
		improved_observed_occupancy_map = fr_utils.remove_isolated_points(observed_occupancy_map)
		t3 = timer()
		print(f't3- t2 = {t3 - t2}')
		frontiers = fr_utils.get_frontiers(improved_observed_occupancy_map)
		t4 = timer()
		print(f't4- t3 = {t4 - t3}')
		frontiers = LN.filter_unreachable_frontiers(frontiers, agent_map_pose, observed_occupancy_map)
		t5 = timer()
		print(f't5- t4 = {t5 - t4}')
		chosen_frontier = fr_utils.get_frontier_with_maximum_area(frontiers, visited_frontier, gt_occupancy_map)
		t6 = timer()
		print(f't6- t5 = {t6 - t5}')
		#============================================= visualize semantic map ===========================================#
		if True:
			#==================================== visualize the path on the map ==============================
			built_semantic_map, observed_area_flag, _ = semMap_module.get_semantic_map()

			#built_semantic_map = built_semantic_map[coords_range[1]:coords_range[3]+1, coords_range[0]:coords_range[2]+1]
			color_built_semantic_map = apply_color_to_map(built_semantic_map)
			color_built_semantic_map = change_brightness(color_built_semantic_map, observed_area_flag, value=60)

			#=================================== visualize the agent pose as red nodes =======================
			x_coord_lst, z_coord_lst, theta_lst = [], [], []
			for cur_pose in traverse_lst:
				x_coord, z_coord = pose_to_coords((cur_pose[0], cur_pose[1]), pose_range, coords_range, WH)
				x_coord_lst.append(x_coord)
				z_coord_lst.append(z_coord)			
				theta_lst.append(cur_pose[2])

			#'''
			fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
			ax[0].imshow(improved_observed_occupancy_map)
			marker, scale = gen_arrow_head_marker(theta_lst[-1])
			ax[0].scatter(x_coord_lst[-1], z_coord_lst[-1], marker=marker, s=(30*scale)**2, c='red', zorder=5)
			ax[0].plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
			for f in frontiers:
				ax[0].scatter(f.points[1], f.points[0], c='white', zorder=2)
				ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
			if chosen_frontier is not None:
				ax[0].scatter(chosen_frontier.points[1], chosen_frontier.points[0], c='green', zorder=4)
				ax[0].scatter(chosen_frontier.centroid[1], chosen_frontier.centroid[0], c='red', zorder=4)
			ax[0].get_xaxis().set_visible(False)
			ax[0].get_yaxis().set_visible(False)
			ax[0].set_title('improved observed_occ_map + frontiers')

			ax[1].imshow(color_built_semantic_map)
			ax[1].get_xaxis().set_visible(False)
			ax[1].get_yaxis().set_visible(False)
			ax[1].set_title('built semantic map')

			fig.tight_layout()
			plt.title('observed area')
			plt.show()
			#fig.savefig(f'{saved_folder}/step_{step}_semmap.jpg')
			#plt.close()
			#assert 1==2
			#'''

	#===================================== check if exploration is done ========================
	if chosen_frontier is None:
		print('There are no more frontiers to explore. Stop navigation.')
		break

	#==================================== update particle filter =============================
	if MODE_FIND_SUBGOAL:
		MODE_FIND_SUBGOAL = False
		explore_steps = 0
		t7 = timer()
		flag_plan, subgoal_coords, subgoal_pose = LN.plan_to_reach_frontier(chosen_frontier, agent_map_pose, observed_occupancy_map, step, saved_folder)
		t8 = timer()
		print(f't8 - t7 = {t8 - t7}')
		if not flag_plan:
			print(f'local planning reach the frontier failed.')
			assert 1==2
		print(f'subgoal_coords = {subgoal_coords}')
		
	#====================================== take next action ================================
	action, next_pose = LN.next_action(env, scene_height)
	print(f'action = {action}')
	if action == "collision":
		step += 1
		explore_steps += 1
		#assert next_pose is None
		# input next_pose is environment pose, not sem_map pose
		semMap_module.add_occupied_cell_pose(next_pose)
		# redo the planning
		print(f'redo planning')
		observed_occupancy_map, gt_occupancy_map, observed_area_flag = semMap_module.get_observed_occupancy_map()
		'''
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(100, 100))
		ax.imshow(occupancy_map, vmax=5)
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		plt.title('collision occupancy_map')
		plt.show()
		'''
		
		flag_plan, subgoal_coords, subgoal_pose = LN.plan_to_reach_frontier(chosen_frontier, agent_map_pose, observed_occupancy_map, step, saved_folder)

		# do not take any actions
	elif action == "": # finished navigating to the subgoal
		print(f'reached the subgoal')
		MODE_FIND_SUBGOAL = True
		visited_frontier.append(chosen_frontier)
	else:
		step += 1
		explore_steps += 1
		print(f'next_pose = {next_pose}')
		agent_pos = np.array([next_pose[0], scene_height, next_pose[1]])
		# output rot is negative of the input angle
		agent_rot = habitat_sim.utils.common.quat_from_angle_axis(-next_pose[2], habitat_sim.geo.GRAVITY)
		obs = env.habitat_env.sim.get_observations_at(agent_pos, agent_rot, keep_agent_at_new_pose=True)

