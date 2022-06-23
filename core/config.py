from yacs.config import CfgNode as CN

_C = CN()

#=============================== dataset and files =============================
_C.GENERAL = CN()
_C.GENERAL.SCENE_HEIGHTS_DICT_PATH = 'output/scene_height_distribution'
_C.GENERAL.HABITAT_CONFIG_PATH = 'configs/exploration_see_the_floor.yaml'
_C.GENERAL.BUILD_MAP_CONFIG_PATH = 'configs/build_map_mp3d.yaml'
_C.GENERAL.LEARNED_MAP_GG_CONFIG_PATH = 'configs/point_nav_mp3d_for_GG.yaml'
_C.GENERAL.DATALOADER_CONFIG_PATH = 'configs/dataloader.yaml'
_C.GENERAL.HABITAT_TRAIN_EPISODE_DATA_PATH = 'data/habitat_data/datasets/pointnav/mp3d/temp_train/all.json.gz'
_C.GENERAL.HABITAT_VAL_EPISODE_DATA_PATH = 'data/habitat_data/datasets/pointnav/mp3d/temp_val/all.json.gz'
_C.GENERAL.HABITAT_TEST_EPISODE_DATA_PATH = 'data/habitat_data/datasets/pointnav/mp3d/temp_test/all.json.gz'
_C.GENERAL.HABITAT_SCENE_DATA_PATH = 'data/habitat_data/scene_datasets/'
_C.GENERAL.RANDOM_SEED = 5

#================================= for save =======================================
_C.SAVE = CN()
_C.SAVE.SEMANTIC_MAP_PATH = 'output/semantic_map' 
_C.SAVE.OCCUPANCY_MAP_PATH = 'output/semantic_map' # built occupancy map
_C.SAVE.TESTING_RESULTS_FOLDER = 'output/TESTING_RESULTS_360degree_DP_UNet_occ_only_Predicted_Potential_10STEP_600STEPS'

#================================== for main_nav.py =====================
_C.MAIN = CN()
_C.MAIN.SPLIT = 'test' # select from 'train', 'val', 'test'
_C.MAIN.NUM_SCENES = 61 # for ('train', 'val', 'test'), num_scenes = (61, 11, 18), 90 in total.
_C.MAIN.TEST_SCENE_LIST = ['2t7WUuJeko7_0', '5ZKStnWn8Zo_0', 'ARNzJeq3xxb_0', 'RPmz2sHmrrY_0', 'UwV83HsGsw3_0', 'Vt2qJdWjCF2_0', 'WYY7iVyf5p8_0', 'YFuZgdQ5vWj_0', 'YVUC4YcDtcY_0', 'fzynW3qQPVF_0', 'gYvKGZ5eRqb_0', 'gxdoqLR6rwA_0', 'jtcxE69GiFV_0', 'pa4otMbVnkk_0', 'q9vSo1VnCiC_0', 'rqfALeAoiTq_0', 'wc2JMjhGNzB_0', 'yqstnuAEVhm_0']


_C.MAIN.TRAIN_SCENE_LIST = ['ZMojNkEp431_0', '1LXtFkjw3qL_0', 'sT4fr6TAbpF_0', 'r1Q1Z4BcV1o_0', 'cV4RVeZvu5T_0', 'EDJbREhghzL_0', 'PX4nDJXEHrG_0', 'YmJkqBEsHnH_0', 'ULsKaCPVFJR_0', '7y3sRwLe3Va_0', 'mJXqzFtmKg4_0', '759xd9YjKW5_0', '17DRP5sb8fy_0', 'ac26ZMwG7aT_0', 'Pm6F8kyY3z2_0', 'Vvot9Ly1tCj_0', 'PuKPg4mmafe_0', 'S9hNv5qa7GM_0', 'vyrNrziPKCB_0', 'SN83YJsR3w2_0', 'rPc6DW4iMge_0', 'r47D5H71a5s_0', 'qoiz87JEwZ2_0', '29hnd4uzFmX_0', '5LpN3gDmAk7_0', 'VFuaQ6m2Qom_0', 'i5noydFURQK_0', 'dhjEzFoUFzH_0', 'E9uDoFAP3SH_0', 's8pcmisQ38h_0', 'GdvgFV5R1Z5_0', '5q7pvUzZiYa_0', 'kEZ7cmS4wCh_0', 'JF19kD82Mey_0', 'pRbA3pwrgk9_0', '2n8kARJN3HM_0', 'HxpKQynjfin_0', 'VzqfbhrpDEA_0', 'Uxmj2M2itWa_0', 'V2XKFyX4ASd_0', 'JeFG25nYj2p_0', 'p5wJjkQkbXX_0', 'VLzqgDo317F_0', '8WUmhLawc2A_0', 'XcA2TqTSSAj_0', 'D7N2EKCX4Sj_0', 'aayBHfsNo7d_0', 'gZ6f7yhEvPG_0', 'D7G3Y4RVNrH_0', 'b8cTxDM8gDG_0', 'VVfe2KiqLaN_0', 'uNb9QFRL6hY_0', 'e9zR4mvMWw7_0', 'sKLMLpTHeUy_0', '82sE5b5pLXE_0', '1pXnuDYAj8r_0', 'jh4fc5c5qoQ_0', 'ur6pFq6Qu1A_0', 'B6ByNegPMKs_0', 'JmbYfDe2QKZ_0', 'gTV8FGcVJC9_0']
_C.MAIN.VAL_SCENE_LIST = ['oLBMNvg9in8_0', 'TbHJrupSAjP_0', 'QUCTc6BB5sX_0', 'EU6Fwq7SyZv_0', 'zsNo4HB9uLZ_0', 'X7HyMhZNoso_0', 'x8F5xyUWy9e_0', '8194nk5LbLH_0', '2azQ1b91cZZ_0', 'pLe4wQe7qrG_0', 'Z6MFQCViBuw_0']
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
_C.SEM_MAP.GRID_Y_SIZE = 60
_C.SEM_MAP.GRID_CLASS_SIZE = 60
_C.SEM_MAP.HABITAT_FLOOR_IDX = 2

#=============================== for navigator ====================================
_C.NAVI = CN()
_C.NAVI.NUM_STEPS = 600
_C.NAVI.FLAG_GT_OCC_MAP = True
_C.NAVI.NUM_STEPS_EXPLORE = 1

_C.NAVI.DETECTOR = 'PanopticSeg'
_C.NAVI.THRESH_REACH = 0.8

_C.NAVI.USE_ROOM_TYPES = True

_C.NAVI.HFOV = 90 # 360 means panorama, 90 means single view

_C.NAVI.PERCEPTION = 'Potential' # possible choices 'Anticipation', 'Potential', 'UNet_Potential'

_C.NAVI.STRATEGY = 'DP' # 'Greedy' vs 'DP'

#========================== for short-range nav ====================================
_C.LN = CN()
_C.LN.LOCAL_MAP_MARGIN = 30 

#================================ for Detectron2 ==============================
_C.DETECTRON2 = CN()

#================================ for Frontier Exploration ===========================
_C.FE = CN()
_C.FE.COLLISION_VAL = 1
_C.FE.FREE_VAL = 2
_C.FE.UNOBSERVED_VAL = 0
_C.FE.OBSTACLE_THRESHOLD = 1
_C.FE.GROUP_INFLATION_RADIUS = 0


#=============================== for Evaluation =====================================
_C.EVAL = CN()
_C.EVAL.USE_ALL_START_POINTS = False

#=============================== for learned occupancy map =============================
_C.MAP = CN()
_C.MAP.N_SPATIAL_CLASSES = 3 # number of categories for spatial prediction, free, unknown, occupied
_C.MAP.MAP_LOSS_SCALE = 1.0
_C.MAP.GRID_DIM = 768 # semantic grid size (grid_dim, grid_dim)
_C.MAP.CROP_SIZE = 160 # size of crop around the agent
_C.MAP.CELL_SIZE = 0.05 # Physical dimensions (meters) of each cell in the grid'
_C.MAP.IMG_SIZE = 256 
_C.MAP.OCCUPANCY_HEIGHT_THRESH = -1.0 # used when estimating occupancy from depth

#============================== for model prediction ===================================
_C.PRED = CN()
_C.PRED.NEIGHBOR_SIZE = 10
_C.PRED.DIVIDE_AREA = 1000
_C.PRED.NUM_WORKERS = 1 #4
_C.PRED.BATCH_SIZE = 4
_C.PRED.INPUT_WH = (128, 128)

_C.PRED.CHECKNAME = 'unet'
_C.PRED.LOSS_TYPE = 'CE'
_C.PRED.EPOCHS = 5
_C.PRED.LR = 0.1
_C.PRED.LR_SCHEDULER = 'poly'
_C.PRED.RESUME = ''
_C.PRED.EVAL_INTERVAL = 2
_C.PRED.NO_VAL = False
_C.PRED.DATASET = 'MP3D'

_C.PRED.NUM_ITER_TRAIN = 5000
_C.PRED.NUM_ITER_EVAL = 100

_C.PRED.RENEW_SCENE_THRESH = 0.95

_C.PRED.SAVED_FOLDER = 'output/VIS_PREDICT_occ_map'

_C.PRED.INPUT = 'occ_only' # input to the UNet. select from ['occ_and_sem', 'occ_only']
_C.PRED.INPUT_CHANNEL = 1 # number of input channels of UNet
_C.PRED.OUTPUT_CHANNEL = 1 # number of output channels of UNet

_C.PRED.DEVICE = 'cuda'

_C.PRED.NUM_GENERATE_EPISODES_PER_SCENE = 10000

#================================ for visualization ============================
_C.SEM_MAP.FLAG_VISUALIZE_EGO_OBS = True
_C.LN.FLAG_VISUALIZE_LOCAL_MAP = False
_C.NAVI.FLAG_VISUALIZE_FINAL_TRAJ = True
_C.NAVI.FLAG_VISUALIZE_MIDDLE_TRAJ = True
_C.NAVI.FLAG_VISUALIZE_FRONTIER_POTENTIAL = False
_C.PRED.FLAG_VISUALIZE_PRED_LABELS = False


