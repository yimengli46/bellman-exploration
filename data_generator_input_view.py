import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import random
from modeling.utils.baseline_utils import apply_color_to_map, pose_to_coords, pxl_coords_to_pose, gen_arrow_head_marker, read_map_npy, read_occ_map_npy, plus_theta_fn, minus_theta_fn, convertInsSegToSSeg
from core import cfg
import modeling.utils.frontier_utils as fr_utils
from modeling.localNavigator_Astar import localNav_Astar
import networkx as nx
from random import Random
from timeit import default_timer as timer
from itertools import islice
from modeling.utils.navigation_utils import SimpleRLEnv, get_scene_name, get_obs_and_pose, get_obs_and_pose_by_action
from modeling.utils.map_utils_pcd_height import SemanticMap
import habitat
import os
from skimage.morphology import skeletonize
from modeling.localNavigator_slam import localNav_slam
import math
import bz2
import _pickle as cPickle
import argparse
import multiprocessing

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


class Data_Gen_View:

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
		#============================ get a gpu
		device_id = gpu_Q.get()

		scene_name = self.scene_name
		print(f'init new scene: {scene_name}')
		env_scene = scene_name[:-2]

		#============================= initialize habitat env===================================
		self.scene_floor_dict = np.load(
			f'{cfg.GENERAL.SCENE_HEIGHTS_DICT_PATH}/{self.split}_scene_floor_dict.npy',
			allow_pickle=True).item()
		self.height = self.scene_floor_dict[env_scene][0]['y']

		#================================ load habitat env============================================
		self.env = build_env(env_scene, device_id=device_id)
		self.env.reset()

		scene = self.env.semantic_annotations()
		self.ins2cat_dict = {
			int(obj.id.split("_")[-1]): obj.category.index()
			for obj in scene.objects
		}

		#================================= read in pre-built occupancy and semantic map =============================
		occ_map_npy = np.load(f'{cfg.SAVE.OCCUPANCY_MAP_PATH}/{self.split}/{scene_name}/BEV_occupancy_map.npy', allow_pickle=True).item()
		gt_occ_map, self.pose_range, self.coords_range, self.WH = read_occ_map_npy(occ_map_npy)

		if cfg.NAVI.D_type == 'Skeleton':
			self.skeleton = skeletonize(gt_occ_map)
			if cfg.NAVI.PRUNE_SKELETON:
				self.skeleton = fr_utils.prune_skeleton(gt_occ_map, self.skeleton)

		# initialize path planner
		self.LN = localNav_Astar(self.pose_range, self.coords_range, self.WH)

		self.LS = localNav_slam(self.pose_range, self.coords_range, self.WH, mark_locs=True, close_small_openings=False, recover_on_collision=False, 
			fix_thrashing=False, point_cnt=2)
		self.LS.reset(gt_occ_map)

		# find the largest connected component on the map
		gt_occupancy_map = gt_occ_map.copy()
		gt_occupancy_map = np.where(gt_occupancy_map == 1, cfg.FE.FREE_VAL, gt_occupancy_map)  # free cell
		self.gt_occupancy_map = np.where(gt_occupancy_map == 0, cfg.FE.COLLISION_VAL, gt_occupancy_map)  # occupied cell
		self.G = self.LN.get_G_from_map(gt_occupancy_map)
		self.largest_cc = list(max(nx.connected_components(self.G), key=len))

		self.act_dict = {-1: 'Done', 0: 'stop', 1: 'forward', 2: 'left', 3:'right'}
		
	def write_to_file(self, num_samples=100):
		count_sample = 0
		#=========================== process each episode
		#for idx_epi in range(num_samples):
		while True:
			#print(f'idx_epi = {idx_epi}')

			#====================================== generate (start, goal) locs, compute path P==========================
			start_loc = self.random.choices(self.largest_cc, k=1)[0]
			print(f'===============> start_loc = {start_loc}')

			semMap_module = SemanticMap(self.split, self.scene_name, self.pose_range, self.coords_range, self.WH,
								self.ins2cat_dict)  # build the observed sem map


			#=====================================start exploration ===============================
			traverse_lst = []
			action_lst = []

			#===================================== setup the start location ===============================#
			start_pose = pxl_coords_to_pose(start_loc, self.pose_range,
								  self.coords_range, self.WH)
			start_pose = (start_pose[0], -start_pose[1])
			agent_pos = np.array([start_pose[0], self.height,
								  start_pose[1]])  # (6.6, -6.9), (3.6, -4.5)
			# check if the start point is navigable
			if not self.env.is_navigable(agent_pos):
				print(f'start pose is not navigable ...')
				assert 1 == 2

			if cfg.NAVI.HFOV == 90:
				obs_list, pose_list = [], []
				heading_angle = 0
				obs, pose = get_obs_and_pose(self.env, agent_pos, heading_angle)
				obs_list.append(obs)
				pose_list.append(pose)
			elif cfg.NAVI.HFOV == 360:
				obs_list, pose_list = [], []
				for rot in [90, 180, 270, 0]:
					heading_angle = rot / 180 * np.pi
					heading_angle = plus_theta_fn(heading_angle, 0)
					obs, pose = get_obs_and_pose(self.env, agent_pos, heading_angle)
					obs_list.append(obs)
					pose_list.append(pose)

			step = 0
			subgoal_coords = None
			subgoal_pose = None
			MODE_FIND_SUBGOAL = True
			explore_steps = 0
			MODE_FIND_GOAL = False
			visited_frontier = set()
			chosen_frontier = None
			old_frontiers = None
			frontiers = None

			while step < cfg.NAVI.NUM_STEPS:
				print(f'step = {step}')

				#=============================== get agent global pose on habitat env ========================#
				pose = pose_list[-1]
				print(f'agent position = {pose[:2]}, angle = {pose[2]}')
				agent_map_pose = (pose[0], -pose[1], -pose[2])
				agent_map_coords = pose_to_coords(agent_map_pose, self.pose_range, self.coords_range, self.WH)
				traverse_lst.append(agent_map_pose)

				# add the observed area
				semMap_module.build_semantic_map(obs_list,
												 pose_list,
												 step=step,
												 saved_folder='')

				if MODE_FIND_SUBGOAL:
					observed_occupancy_map, gt_occupancy_map, observed_area_flag, built_semantic_map = semMap_module.get_observed_occupancy_map(agent_map_pose
					)

					if frontiers is not None:
						old_frontiers = frontiers

					frontiers = fr_utils.get_frontiers(observed_occupancy_map)
					frontiers = frontiers - visited_frontier

					frontiers, dist_occupancy_map = self.LN.filter_unreachable_frontiers(
						frontiers, agent_map_pose, observed_occupancy_map)

					#if cfg.NAVI.PERCEPTION == 'View_Potential':
					# find connections between frontiers and panorama
					#====================== get the panorama image ===============
					rgb_lst, depth_lst, sseg_lst = [], [], []
					for i_obs in [2, 3, 0, 1, 2, 3]:
						obs = obs_list[i_obs]
						# load rgb image, depth and sseg
						rgb_img = obs['rgb']
						depth_img = obs['depth'][:,:,0]
						#print(f'depth_img.shape = {depth_img.shape}')
						InsSeg_img = obs["semantic"]
						sseg_img = convertInsSegToSSeg(InsSeg_img, self.ins2cat_dict)
						rgb_lst.append(rgb_img)
						depth_lst.append(depth_img)
						sseg_lst.append(sseg_img)
					panorama_rgb = np.concatenate(rgb_lst, axis=1)
					panorama_depth = np.concatenate(depth_lst, axis=1)
					panorama_sseg = np.concatenate(sseg_lst, axis=1)
					print(f'panorama_depth.shape = {panorama_depth.shape}')

					if False:
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

					#========================= compute the angle between frontier and agent
					for fron in frontiers:
						fron_centroid_coords = (int(fron.centroid[1]), int(fron.centroid[0]))
						#print(f'fron_coords = {fron_centroid_coords}, agent_map_coords = {agent_map_coords}')
						angle_fron_agent = math.atan2(agent_map_coords[1] - fron_centroid_coords[1],
							fron_centroid_coords[0] - agent_map_coords[0])
						#print(f'angle_fron_agent = {math.degrees(angle_fron_agent)}')
						#print(f'rot in drawing is {math.degrees(rot)}, rotate_rot is {math.degrees(rotate_rot)}')
						angle_agent = -(agent_map_pose[2] - .5 * math.pi)
						#print(f'angle_agent = {math.degrees(angle_agent)}')
						deg = math.degrees(minus_theta_fn(angle_agent, angle_fron_agent))
						#print(f'angle difference is {deg}')
						deg = -deg + 135
						if deg < 45:
							deg += 360
						assert deg >= 45
						#print(f'final deg = {deg}')
						bin_from_deg = int(256 / 90 * deg)

						fron_rgb   = panorama_rgb[:, bin_from_deg-128:bin_from_deg+128]
						fron_depth = panorama_depth[:, bin_from_deg-128:bin_from_deg+128]
						fron_sseg  = panorama_sseg[:, bin_from_deg-128:bin_from_deg+128]

						fron.rgb_obs = fron_rgb
						fron.depth_obs = fron_depth
						fron.sseg_obs = fron_sseg

						if False:
							fig, ax = plt.subplots(nrows=1,
											   ncols=4,
											   figsize=(25, 5))
							ax[0].imshow(observed_occupancy_map, cmap='gray')
							for f in frontiers:
								ax[0].scatter(f.points[1], f.points[0], c='yellow', zorder=2)
								ax[0].scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
							
							ax[0].scatter(fron.points[1],
									   fron.points[0],
									   c='green',
									   zorder=4)
							ax[0].scatter(fron.centroid[1],
									   fron.centroid[0],
									   c='red',
									   zorder=4)
							ax[0].get_xaxis().set_visible(False)
							ax[0].get_yaxis().set_visible(False)
							ax[0].set_title('explored occupancy map')

							ax[1].imshow(fron_rgb)
							ax[1].get_xaxis().set_visible(False)
							ax[1].get_yaxis().set_visible(False)
							ax[1].set_title('area potential')

							ax[2].imshow(fron_depth)
							ax[2].get_xaxis().set_visible(False)
							ax[2].get_yaxis().set_visible(False)
							ax[2].set_title('area potential')

							ax[3].imshow(fron_sseg)
							ax[3].get_xaxis().set_visible(False)
							ax[3].get_yaxis().set_visible(False)
							ax[3].set_title('area potential')

							fig.tight_layout()
							plt.title(f'frontier egocentric view')
							plt.show()

					#=========================== visualization ==============================

					if cfg.NAVI.D_type == 'Skeleton':
						frontiers = fr_utils.compute_frontier_potential(frontiers, observed_occupancy_map, gt_occupancy_map, 
							observed_area_flag, built_semantic_map, self.skeleton)

					#for fron in frontiers:
					#	print(f'fron.R = {fron.R}, fron.Din = {fron.Din}')

					if old_frontiers is not None:
						frontiers, added_frontiers_set = fr_utils.update_frontier_set_data_gen(old_frontiers, frontiers, max_dist=5, chosen_frontier=chosen_frontier)
					else:
						added_frontiers_set = frontiers	

					#for fron in added_frontiers_set:
					#	print(f'fron.R = {fron.R}, fron.Din = {fron.Din}')

					#============================ save the added frontiers set images =====================
					for fron in added_frontiers_set:
						eps_data = {}
						eps_data['rgb'] = fron.rgb_obs.copy()
						eps_data['depth'] = fron.depth_obs.copy()
						eps_data['sseg'] = fron.sseg_obs.copy()
						eps_data['R'] = fron.R / cfg.PRED.VIEW.DIVIDE_AREA
						eps_data['D'] = fron.D / cfg.PRED.VIEW.DIVIDE_D
						eps_data['Din'] = fron.Din / cfg.PRED.VIEW.DIVIDE_D
						eps_data['Dout'] = fron.Dout / cfg.PRED.VIEW.DIVIDE_D

						sample_name = str(count_sample).zfill(len(str(num_samples)))
						with bz2.BZ2File(f'{self.scene_folder}/{sample_name}.pbz2', 'w') as fp:
							cPickle.dump(
								eps_data,
								fp
							)

						count_sample += 1
						if count_sample == num_samples:
							self.env.close()
							#================================ release the gpu============================
							gpu_Q.put(device_id)
							return

					#============================ delete the added frontiers set images ======================
					for fron in added_frontiers_set:
						del fron.rgb_obs
						del fron.depth_obs
						del fron.sseg_obs

					
					chosen_frontier = fr_utils.get_the_nearest_frontier(frontiers, agent_map_pose, dist_occupancy_map, self.LN)

					#============================================= visualize semantic map ===========================================#
					if cfg.NAVI.FLAG_VISUALIZE_MIDDLE_TRAJ:

						#=================================== visualize the agent pose as red nodes =======================
						x_coord_lst, z_coord_lst, theta_lst = [], [], []
						for cur_pose in traverse_lst:
							x_coord, z_coord = pose_to_coords(
								(cur_pose[0], cur_pose[1]), self.pose_range, self.coords_range,
								self.WH)
							x_coord_lst.append(x_coord)
							z_coord_lst.append(z_coord)
							theta_lst.append(cur_pose[2])

						#'''
						fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
						ax.imshow(observed_occupancy_map, cmap='gray')
						marker, scale = gen_arrow_head_marker(theta_lst[-1])
						ax.scatter(x_coord_lst[-1],
								   z_coord_lst[-1],
								   marker=marker,
								   s=(30 * scale)**2,
								   c='red',
								   zorder=5)
						#ax.plot(x_coord_lst, z_coord_lst, lw=5, c='blue', zorder=3)
						ax.scatter(x_coord_lst, 
								   z_coord_lst, 
								   c=range(len(x_coord_lst)), 
								   cmap='viridis', 
								   s=np.linspace(5, 2, num=len(x_coord_lst))**2, 
								   zorder=3)
						for f in frontiers:
							ax.scatter(f.points[1], f.points[0], c='yellow', zorder=2)
							ax.scatter(f.centroid[1], f.centroid[0], c='red', zorder=2)
						if chosen_frontier is not None:
							ax.scatter(chosen_frontier.points[1],
									   chosen_frontier.points[0],
									   c='green',
									   zorder=4)
							ax.scatter(chosen_frontier.centroid[1],
									   chosen_frontier.centroid[0],
									   c='red',
									   zorder=4)
						ax.get_xaxis().set_visible(False)
						ax.get_yaxis().set_visible(False)

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

				#====================================== take next action ================================
				act, act_seq, subgoal_coords, subgoal_pose = self.LS.plan_to_reach_frontier(agent_map_pose, chosen_frontier, 
					observed_occupancy_map)
				print(f'subgoal_coords = {subgoal_coords}')
				print(f'action = {self.act_dict[act]}')
				action_lst.append(act)
				
				if act == -1 or act == 0: # finished navigating to the subgoal
					print(f'reached the subgoal')
					MODE_FIND_SUBGOAL = True
					visited_frontier.add(chosen_frontier)
				else:
					step += 1
					explore_steps += 1
					# output rot is negative of the input angle
					if cfg.NAVI.HFOV == 90:
						obs_list, pose_list = [], []
						obs, pose = get_obs_and_pose_by_action(self.env, act)
						obs_list.append(obs)
						pose_list.append(pose)
					elif cfg.NAVI.HFOV == 360:
						obs_list, pose_list = [], []
						obs, pose = get_obs_and_pose_by_action(self.env, act)
						next_pose = pose
						agent_pos = np.array([next_pose[0], self.height, next_pose[1]])
						for rot in [90, 180, 270, 0]:
							heading_angle = rot / 180 * np.pi
							heading_angle = plus_theta_fn(heading_angle, -next_pose[2])
							obs, pose = get_obs_and_pose(self.env, agent_pos, heading_angle)
							obs_list.append(obs)
							pose_list.append(pose)

				if explore_steps == cfg.NAVI.NUM_STEPS_EXPLORE:
					explore_steps = 0
					MODE_FIND_SUBGOAL = True

def multi_run_wrapper(args):
	""" wrapper for multiprocessor """
	gen = Data_Gen_View(args[0], args[1], saved_dir=args[2])
	gen.write_to_file(num_samples=cfg.PRED.VIEW.NUM_GENERATED_SAMPLES_PER_SCENE)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--j',
						type=int,
						required=False,
						default=1)
	args = parser.parse_args()
	cfg.merge_from_file('configs/exp_train_input_view_depth_and_sem.yaml')
	cfg.freeze()

	#====================== get the available GPU devices ============================
	visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
	devices = [int(dev) for dev in visible_devices]

	for device_id in devices:
		for _ in range(args.j):
			gpu_Q.put(device_id)

	SEED = cfg.GENERAL.RANDOM_SEED
	random.seed(SEED)
	np.random.seed(SEED)

	split = cfg.MAIN.SPLIT
	if split == 'train':
		scene_list = cfg.MAIN.TRAIN_SCENE_LIST
	elif split == 'val':
		scene_list = cfg.MAIN.VAL_SCENE_LIST
	elif split == 'test':
		scene_list = cfg.MAIN.TEST_SCENE_LIST
		
	output_folder = cfg.PRED.VIEW.GEN_SAMPLES_SAVED_FOLDER
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)

	split_folder = f'{output_folder}/{split}'
	if not os.path.exists(split_folder):
		os.mkdir(split_folder)

	if cfg.PRED.VIEW.multiprocessing == 'single': # single process
		for scene in scene_list: 
			gen = Data_Gen_View(split, scene, saved_dir=split_folder)
			gen.write_to_file(num_samples=cfg.PRED.VIEW.NUM_GENERATED_SAMPLES_PER_SCENE)
	elif cfg.PRED.VIEW.multiprocessing == 'mp':
		with multiprocessing.Pool(processes=cfg.PRED.VIEW.NUM_PROCESS) as pool:
			args0 = [split for _ in range(len(scene_list))]
			args1 = [scene for scene in scene_list]
			args2 = [split_folder for _ in range(len(scene_list))]
			pool.map(multi_run_wrapper, list(zip(args0, args1, args2)))
			pool.close()
	elif cfg.PRED.VIEW.multiprocessing == 'mpi4y':
		from mpi4py.futures import MPIPoolExecutor
		args0 = [split for _ in range(len(scene_list))]
		args1 = [scene for scene in scene_list]
		args2 = [split_folder for _ in range(len(scene_list))]
		executor = MPIPoolExecutor()
		prime_sets = executor.map(multi_run_wrapper, list(zip(args0, args1, args2)))
		executor.shutdown()

if __name__ == "__main__":
	gpu_Q = multiprocessing.Queue()
	main()

'''
cfg.merge_from_file('configs/exp_train_input_view_depth_and_sem.yaml')
cfg.freeze()

SEED = cfg.GENERAL.RANDOM_SEED
random.seed(SEED)
np.random.seed(SEED)

scene_name = '17DRP5sb8fy_0'
split = 'train'

output_folder = cfg.PRED.VIEW.GEN_SAMPLES_SAVED_FOLDER
if not os.path.exists(output_folder):
	os.mkdir(output_folder)

split_folder = f'{output_folder}/{split}'
if not os.path.exists(split_folder):
	os.mkdir(split_folder)

data = Data_Gen_View(split=split, scene_name=scene_name, saved_dir=split_folder)
data.write_to_file(num_samples=cfg.PRED.VIEW.NUM_GENERATED_SAMPLES_PER_SCENE)
'''