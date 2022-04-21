from yacs.config import CfgNode as CN

_C = CN()

#=============================== dataset and files =============================
_C.GENERAL = CN()
_C.GENERAL.SCENE_HEIGHTS_DICT_PATH = 'output/scene_height_distribution'
_C.GENERAL.HABITAT_CONFIG_PATH = 'configs/exploration_see_the_floor.yaml'
_C.GENERAL.BUILD_MAP_CONFIG_PATH = 'configs/build_map_mp3d.yaml'
_C.GENERAL.HABITAT_EPISODE_DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/datasets/pointnav/mp3d/temp/all.json.gz'
_C.GENERAL.HABITAT_SCENE_DATA_PATH = '/home/yimeng/Datasets/habitat-lab/data/scene_datasets/'
_C.GENERAL.RANDOM_SEED = 5

#================================= for save =======================================
_C.SAVE = CN()
_C.SAVE.OCCUPANCY_MAP_PATH = 'output/semantic_map' # built occupancy map
_C.SAVE.TESTING_RESULTS_FOLDER = 'output/TESTING_RESULTS_90degree_Greedy_Frontier_Occupancy'

#================================== for main_nav.py =====================
_C.MAIN = CN()
_C.MAIN.SCENE_LIST = ['2t7WUuJeko7_0', '5ZKStnWn8Zo_0', 'ARNzJeq3xxb_0', 'RPmz2sHmrrY_0', 'UwV83HsGsw3_0', 'Vt2qJdWjCF2_0', 'WYY7iVyf5p8_0', 'YFuZgdQ5vWj_0', 'YVUC4YcDtcY_0', 'fzynW3qQPVF_0', 'gYvKGZ5eRqb_0', 'gxdoqLR6rwA_0', 'jtcxE69GiFV_0', 'pa4otMbVnkk_0', 'q9vSo1VnCiC_0', 'rqfALeAoiTq_0', 'wc2JMjhGNzB_0', 'yqstnuAEVhm_0']

#================================ for semantic map ===============================
_C.SEM_MAP = CN()
_C.SEM_MAP.ENLARGE_SIZE = 10
_C.SEM_MAP.IGNORED_MAP_CLASS = [0, 59]
_C.SEM_MAP.IGNORED_ROOM_CLASS = [0] # ignored class on the room type map
_C.SEM_MAP.IGNORED_SEM_CLASS = [0, 17] # for semantic segmentation, class 17 is ceiling
_C.SEM_MAP.OBJECT_MASK_PIXEL_THRESH = 100
_C.SEM_MAP.UNDETECTED_PIXELS_CLASS = 59 # explored but semantic-unrecognized pixel
_C.SEM_MAP.CELL_SIZE = 0.1
_C.SEM_MAP.WORLD_SIZE = 50.0 # world model size in each dimension (left, right, top , bottom)
_C.SEM_MAP.GRID_Y_SIZE = 100
_C.SEM_MAP.GRID_CLASS_SIZE = 100

#=============================== for navigator ====================================
_C.NAVI = CN()
_C.NAVI.NUM_STEPS = 1250
_C.NAVI.FLAG_GT_OCC_MAP = True
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


#=============================== for Evaluation =====================================
_C.EVAL = CN()
_C.EVAL.USE_ALL_START_POINTS = True

#================================ for visualization ============================
_C.SEM_MAP.FLAG_VISUALIZE_EGO_OBS = False
_C.LN.FLAG_VISUALIZE_LOCAL_MAP = False
_C.NAVI.FLAG_VISUALIZE_FINAL_TRAJ = True
_C.NAVI.FLAG_VISUALIZE_MIDDLE_TRAJ = True