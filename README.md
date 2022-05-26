# Bellman-Equation-2022
Detect frontiers between explored and unknown areas as subgoals.
Select subgoals with maximum values computed from a Bellman Equation designed for exploration.

### Installation
```
git clone --branch yimeng https://github.com/RAIL-group/bellman-exploration-2022.git
cd  bellman-exploration-2022
mkdir output
```

### Dependencies
Install the following dependencies before you run the code:
```
pip install -r requirements.txt
```
Install habitat-lab and habitat-sim following guidance from [here](https://github.com/facebookresearch/habitat-lab "here").

### Dataset Setup
Download Habitat MP3D scene data from [here](https://github.com/facebookresearch/habitat-lab "here").
Or you can download it from [google drive](https://drive.google.com/drive/folders/180gcW5xq6ZWM4f7yHK_kPc-iAVpGGNfl?usp=sharing "google drive").
Upzip the scene data and put it under `habitat-lab/data/scene_datasets/mp3d`.
You also need to download self-generated task episode data from [here](https://drive.google.com/drive/folders/1raUypuI9Zgig3dfFgWINv40bnKfvUadW?usp=sharing "here")
Unzip the episode data and put it under `habitat-lab/data/datasets/pointnav/mp3d`.
Create softlinks to the data.
```
cd  bellman-exploration-2022
mkdir data
ln -s data/habitat_data habitat-lab/data
```
Then you need to organize the habitat data in the following way.
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
The code can do (a) exploring scenes (b) building top-down view semantic maps of MP3D scenes (c) building occupancy maps of MP3D scenes.
All the parameters are controlled by the configuration file `core/config.py`.
##### Exploring the environment
To run the large-scale evaluation with Bellman Equation, you need to download pre-generated 'scene maps' and 'scene floor heights files' from [here](https://drive.google.com/drive/folders/10ApKQzaIPDvEAvbcVXQkaGBjxnvUIpND?usp=sharing "here").
Download it and put it under ` bellman-exploration-2022/output`.
Then you can start the evaluation of the exploration model.
```
python main_eval.py
```
##### Build top-down view semantic maps
```
python -m scripts.build_semantic_BEV_map.py
```
##### Build top-down view occupancy maps
```
python -m scripts.build_occupancy_map_from_continuous_habitat.py
```
