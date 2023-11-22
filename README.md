# Bellman-Exploration
Detect frontiers between explored and unknown areas as subgoals.  
Select subgoals with maximum values computed from a Bellman Equation designed for exploration.  
<img src='Figs/example_traj.jpg'>
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
Install the following dependencies before you run the code:  
```
pip install -r requirements.txt
```
You can install Habitat-Lab and Habitat-Sim following guidance from [here](https://github.com/facebookresearch/habitat-lab "here").  
We recommend to install Habitat-Lab and Habitat-Sim from the source code.  
We use `habitat==0.2.1` and `habitat_sim==0.2.1`.  
Use the following commands to set it up:  
```
# install habitat-lab
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
git checkout tags/stable
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
git checkout tags/stable
python setup.py install --with-cuda
```
If you don't want to install all the requirements, the necessary dependencies are:  
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
Download Habitat MP3D scene data from [here](https://github.com/facebookresearch/habitat-lab "here").    
Or you can download it from [google drive](https://drive.google.com/drive/folders/180gcW5xq6ZWM4f7yHK_kPc-iAVpGGNfl?usp=sharing "google drive").  
Upzip the scene data and put it under `habitat-lab/data/scene_datasets/mp3d`.  
You also need to download self-generated task episode data from [here](https://drive.google.com/drive/folders/1raUypuI9Zgig3dfFgWINv40bnKfvUadW?usp=sharing "here")  
Unzip the episode data and put it under `habitat-lab/data/datasets/pointnav/mp3d`.  
Create softlinks to the data.  
```
cd  bellman-exploration
ln -s habitat-lab/data data
```
The code requires the datasets in data folder in the following format:
```
habitat-lab/data
                /datasets/pointnav/mp3d
                                        /temp_train
                                        /temp_val
                                        /temp_test
                scene_datasets/mp3d
                                    /1LXtFkjw3qL
                                    /1pXnuDYAj8r
                                    /....
```

### How to Run?
The code can do  
(a) exploring scenes  
(b) building top-down view semantic maps of MP3D scenes   
(c) building occupancy maps of MP3D scenes.   
All the parameters are controlled by the configuration file `core/config.py`.   
Please create a new configuration file when you initialize a new task and saved in folder `configs`.
##### Exploring the environment
To run the large-scale evaluation, you need to download pre-generated 'scene maps' and 'scene floor heights' from [here](https://drive.google.com/drive/folders/10ApKQzaIPDvEAvbcVXQkaGBjxnvUIpND?usp=sharing "here").  
Download it and put it under ` bellman-exploration/output`.  
Then you can start the evaluation.  
For example, if you want to evaluate the baseline Greedy approach, use the following command.  
```
python main_eval.py --config='exp_360degree_Greedy_GT_Potential_1STEP_500STEPS.yaml'
```
If you want to evaluate the proposed Bellman Equation approach, use this command.
```
python main_eval.py --config='exp_360degree_DP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
```
##### Build top-down view semantic maps
```
python -m scripts.build_semantic_BEV_map.py
```
##### Build top-down view occupancy maps
```
python -m scripts.build_occupancy_map_from_continuous_habitat.py
```
