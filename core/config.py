from yacs.config import CfgNode as CN

_C = CN()

#=============================== dataset and files =============================
_C.GENERAL = CN()
_C.GENERAL.SCENE_HEIGHTS_DICT_PATH = '/home/yimeng/Datasets/habitat-lab/habitat_nav/build_avd_like_scenes/output/scene_height_distribution/scene_heights.npy'
_C.GENERAL.HABITAT_CONFIG_PATH = "configs/exploration_see_the_floor.yaml" 
_C.GENERAL.HABITAT_EPISODE_DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/objectnav/gibson/all.json.gz'
_C.GENERAL.HABITAT_SCENE_DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
_C.GENERAL.RANDOM_SEED = 5

#================================= for save =======================================
_C.SAVE = CN()
_C.SAVE.SEM_MAP_FROM_SCENE_GRAPH_PATH = 'output/gt_semantic_map_from_SceneGraph'
_C.SAVE.SEM_MAP_PATH = 'output/semantic_map' # built semantic map
_C.SAVE.OCCUPANCY_MAP_PATH = 'output/semantic_map' # built occupancy map
_C.SAVE.TESTING_DATA_FOLDER = 'output/TESTING_DATA'
_C.SAVE.TESTING_RESULTS_FOLDER = 'output/TESTING_RESULTS_USE_ROOM_TYPES'

#================================== for main_nav.py =====================
_C.MAIN = CN()
_C.MAIN.SCENE_LIST = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0'] # ['Allensville_0']


#================================== for gen_testing_data.py =====================
_C.GEN_TEST = CN()
_C.GEN_TEST.SCENE_LIST = ['Collierville_1', 'Darden_0', 'Markleeville_0', 'Wiconisco_0']
_C.GEN_TEST.NUM_EPISODES = 100
_C.GEN_TEST.ALLOWED_CATS = ['couch', 'potted_plant', 'refrigerator', 'oven', 'tv', 'chair', 'vase', 'potted plant', \
	'toilet', 'clock', 'cup', 'bottle', 'bed', 'sink']

#================================ for semantic map ===============================
_C.SEM_MAP = CN()
_C.SEM_MAP.ENLARGE_SIZE = 10
_C.SEM_MAP.IGNORED_MAP_CLASS = [0, 59]
_C.SEM_MAP.IGNORED_ROOM_CLASS = [0] # ignored class on the room type map
_C.SEM_MAP.IGNORED_SEM_CLASS = [54] # for semantic segmentation, class 54 is ceiling
_C.SEM_MAP.OBJECT_MASK_PIXEL_THRESH = 100
_C.SEM_MAP.UNDETECTED_PIXELS_CLASS = 59 # explored but semantic-unrecognized pixel
_C.SEM_MAP.CELL_SIZE = 0.1
_C.SEM_MAP.WORLD_SIZE = 30.0 # world model size in each dimension (left, right, top , bottom)
_C.SEM_MAP.GRID_Y_SIZE = 100
_C.SEM_MAP.GRID_CLASS_SIZE = 100

#=============================== for navigator ====================================
_C.NAVI = CN()
_C.NAVI.NUM_STEPS = 1250
_C.NAVI.FLAG_GT_SEM_MAP = True
_C.NAVI.NUM_STEPS_EXPLORE = 30

_C.NAVI.DETECTOR = 'PanopticSeg'
_C.NAVI.THRESH_REACH = 0.8

_C.NAVI.USE_ROOM_TYPES = True

#========================== for short-range nav ====================================
_C.LN = CN()
_C.LN.LOCAL_MAP_MARGIN = 30 

#================================ for Detectron2 ==============================
_C.DETECTRON2 = CN()

#================================ for Frontier Exploration ===========================
_C.FE = CN()
_C.FE.COLLISION_VAL = 1
_C.FE.FREE_VAL = 3
_C.FE.UNOBSERVED_VAL = 0
_C.FE.OBSTACLE_THRESHOLD = 1
_C.FE.GROUP_INFLATION_RADIUS = 0

#================================ for visualization ============================
_C.NAVI.NUM_STEPS_VIS = 50 # visualize the traj after every X steps
_C.SEM_MAP.FLAG_VISUALIZE_EGO_OBS = False
_C.PF.FLAG_VISUALIZE_INS_WEIGHTS = False
_C.PF.FLAG_VISUALIZE_PEAKS = False
_C.LN.FLAG_VISUALIZE_LOCAL_MAP = False