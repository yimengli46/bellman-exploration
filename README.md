# Bellman-Exploration
This is the code release for our IROS 2023 paper:

[Learning-Augmented Model-Based Planning for Visual Exploration](https://arxiv.org/pdf/2211.07898.pdf)<br/>
Yimeng Li*, Arnab Debnath*, Gregory J. Stein, Jana Kosecka<br/>
George Mason University

[Project website](https://yimengli46.github.io/Projects/IROS2023Exploration/index.html)

<img src='Figs/iros2023_dp.gif'/>

```bibtex
@article{Li2022LearningAugmentedMP,
  title={Learning-Augmented Model-Based Planning for Visual Exploration},
  author={Yimeng Li and Arnab Debnath and Gregory J. Stein and Jana Kosecka},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2023}
}
```

### Installation
```
git clone --branch main https://github.com/yimengli46/bellman-exploration.git
cd  bellman-exploration
mkdir output
```

### Dependencies
We use `python==3.7.4`.  
We recommend using a conda environment.  
```
conda create --name bellman_explore python=3.7.4
source activate bellman_explore
```

You can install Habitat-Lab and Habitat-Sim following guidance from [here](https://github.com/facebookresearch/habitat-lab "here").  
We recommend installing Habitat-Lab and Habitat-Sim from the source code.  
We use `habitat==0.2.1` and `habitat_sim==0.2.1`.  
Use the following commands to set it up:  
```
# install habitat-lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/v0.2.1
pip install -e .

# install habitat-sim
git clone --recurse --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
sudo apt-get update || true
# These are fairly ubiquitous packages and your system likely has them already,
# but if not, let's get the essentials for EGL support:
sudo apt-get install -y --no-install-recommends \
     libjpeg-dev libglm-dev libgl1-mesa-glx libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
git checkout tags/v0.2.1
python setup.py install --with-cuda
```
You also need to install the dependencies:
```
habitat==0.2.1
habitat-sim==0.2.1
torch==1.8.0
torchvision==0.9.0
matplotlib==3.3.4
networkx==2.6.3
scikit-fmm==2022.3.26
scikit-image
sknw
tensorboardX
```

### Dataset Setup
Download *scene* dataset of **Matterport3D(MP3D)** from [here](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md "here").      
Upzip the scene data and put it under `habitat-lab/data/scene_datasets/mp3d`.  
You also need to download self-generated task episode data from [here](https://drive.google.com/drive/folders/1raUypuI9Zgig3dfFgWINv40bnKfvUadW?usp=sharing "here")  
Unzip the episode data and put it under `habitat-lab/data/datasets/pointnav/mp3d`.  
Create a soft link to the data.  
```
cd  bellman-exploration
ln -s habitat-lab/data data
```
The code requires the datasets in data folder in the following format:
```
habitat-lab/data
  └── datasets/pointnav/mp3d/v1
       └── temp_train
       └── temp_val
       └── temp_test
  └── scene_datasets/mp3d
        └── 1LXtFkjw3qL
        └── 1pXnuDYAj8r
        └── ....
```

### How to Run?
The code supports       
(a) **Exploration on MP3D test episodes.**     
All parameters are managed through the configuration file `core/config.py`.     
When initiating a new task, create a new configuration file and save it in the `configs` folder.     

##### Running the Demo
Before executing the demo, download the pre-generated `scene_maps`, `scene_floor_heights` from [here](https://drive.google.com/file/d/1P0AtKn5k2xm5rm2YP1kABuulsSTAU_X7/view?usp=sharing "here").    
Unzip the file and place the folders under `bellman-exploration/output`.

The trained learning module is available for download [here](https://drive.google.com/file/d/1SYnq1Zntk0wFSbB6EMx7wcHN3XSFMOul/view?usp=sharing).    
Unzip it and place the folders under `bellman-exploration/output`.

(b) **Large-Scale Evaluation**     
Initiating the evaluation is a straightforward process. Follow these steps:    

1. For desktop evaluation of the Greedy planner, use the following command:
```
python main_eval.py --config='exp_360degree_Greedy_GT_Potential_1STEP_500STEPS.yaml'
```
Use the following command for the proposed Bellman planner:
```
python main_eval.py --config='exp_360degree_DP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
```
2. If you're working with a server equipped with multiple GPUs, choose an alternative configuration file:
```
python main_eval_multiprocess.py --config='large_exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
```
Feel free to customize configurations to meet your evaluation requirements.      
Configuration files are provided in the `configs` folder, following this naming convention:
* Files starting with `large_exp` run complete exploration testing episodes.
* Files with only `exp` run the first three episodes of each test scene.
* `Greedy` in the title signifies running the greedy planner and `FME` signifies running frontier-based exploration while `DP` signifies running the Bellman Equation formulated exploration.
* `NAVMESH` uses an oracle mapper, and `PCDHEIGHT` builds the map on the fly.
* `UNet_OCCandSEM_Potential` means the potential is computed by UNet with inputs of both occupancy and semantic maps.
* `GT_Potential` means using ground-truth potential values.
* `500STEPS` signifies a maximum of 500 allowed steps.


### Train the learning modules
##### Generate training data
To generate training data, run the following command:   
```
python data_generator_input_partial_map.py
```
You can customize the training hyperparameters using the configuration file `exp_train_input_partial_map_occ_and_sem.yaml`.        
Here are a few key options:     
* Set the `SPLIT` parameter to either `train` or `val` to generate data for training or validation scenes.
* Adjust `PRED.PARTIAL_MAP.multiprocessing` to `single` or `mp` for single-threaded or multithreaded generation, respectively.


##### Train the learning module
Run the following command to initiate training:
```
python train_UNet_input_partial_map.py
```
Customize training hyperparameters using the same `exp_train_input_partial_map_occ_and_sem.yaml` configuration file.      
Key options include:
* Adjust `BATCH_SIZE`, `EPOCHS`, and `NUM_WORKERS` based on your computer hardware and GPU memory.

