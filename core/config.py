from yacs.config import CfgNode as CN

_C = CN()

#=============================== dataset and files =============================
_C.GENERAL = CN()
_C.GENERAL.SCENE_HEIGHTS_DICT_PATH = 'output/scene_height_distribution'
_C.GENERAL.HABITAT_CONFIG_PATH = 'configs/habitat_env/exploration_see_the_floor.yaml'
_C.GENERAL.BUILD_MAP_CONFIG_PATH = 'configs/habitat_env/build_map_mp3d.yaml'
_C.GENERAL.LEARNED_MAP_GG_CONFIG_PATH = 'configs/habitat_env/point_nav_mp3d_for_GG.yaml'
_C.GENERAL.DATALOADER_CONFIG_PATH = 'configs/habitat_env/dataloader.yaml'
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

#============================== for model prediction ===================================
_C.PRED = CN()

#========================= input partial map===============
_C.PRED.PARTIAL_MAP = CN()
# observed neighborhood size of the agent
_C.PRED.PARTIAL_MAP.NEIGHBOR_SIZE = 10
# devide the real area by a constant
_C.PRED.PARTIAL_MAP.DIVIDE_AREA = 1000 
# number of workers for the dataloader
_C.PRED.PARTIAL_MAP.NUM_WORKERS = 1
# batch size
_C.PRED.PARTIAL_MAP.BATCH_SIZE = 4
# input map size in to the model (W, H)
_C.PRED.PARTIAL_MAP.INPUT_WH = (128, 128)
# model name
_C.PRED.PARTIAL_MAP.CHECKNAME = 'unet'
# loss function
_C.PRED.PARTIAL_MAP.LOSS_TYPE = 'CE'
# number of training epoches
_C.PRED.PARTIAL_MAP.EPOCHS = 5
# start learning rate
_C.PRED.PARTIAL_MAP.LR = 0.1
# scheduler
_C.PRED.PARTIAL_MAP.LR_SCHEDULER = 'poly'
# resume model trajectory
_C.PRED.PARTIAL_MAP.RESUME = ''
# between the number of interval we will evaluate the model on the validation set
_C.PRED.PARTIAL_MAP.EVAL_INTERVAL = 2
# name of the dataset
_C.PRED.PARTIAL_MAP.DATASET = 'MP3D'
# model weights saving folder
_C.PRED.PARTIAL_MAP.SAVED_FOLDER = 'output/VIS_PREDICT_occ_map'
# input type of the model, having semantic map or not
_C.PRED.PARTIAL_MAP.INPUT = 'occ_only' #select from ['occ_and_sem', 'occ_only']
# number of the input channels of UNet
_C.PRED.PARTIAL_MAP.INPUT_CHANNEL = 1 
# number of output channels of UNet
_C.PRED.PARTIAL_MAP.OUTPUT_CHANNEL = 1 
# device number
_C.PRED.PARTIAL_MAP.DEVICE = 'cuda'
# number of generated samples per scene, used for data generator
_C.PRED.PARTIAL_MAP.NUM_GENERATED_SAMPLES_PER_SCENE = 10000
# Number of the step gap when saving the map
_C.PRED.PARTIAL_MAP.STEP_GAP = 1

#========================= input partial map===============
_C.PRED.VIEW = CN()

#================================ for visualization ============================
_C.SEM_MAP.FLAG_VISUALIZE_EGO_OBS = True
_C.LN.FLAG_VISUALIZE_LOCAL_MAP = False
_C.NAVI.FLAG_VISUALIZE_FINAL_TRAJ = True
_C.NAVI.FLAG_VISUALIZE_MIDDLE_TRAJ = True
_C.NAVI.FLAG_VISUALIZE_FRONTIER_POTENTIAL = False
_C.PRED.PARTIAL_MAP.FLAG_VISUALIZE_PRED_LABELS = False


