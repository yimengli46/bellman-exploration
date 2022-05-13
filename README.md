#Bellman-Equation-2022
Detect frontiers between explored and unknown areas as subgoals.
Select subgoals with maximum values computed from a Bellman Equation designed for exploration.

### Dependencies
Install the following dependencies before you run the code:
```
pip install -r requirements.txt
```

### Dataset Setup
Download Habitat MP3D data from [here](http://https://github.com/facebookresearch/habitat-lab "here").
Create softlinks to the data.
```
mkdir data
ln -s data/habitat_data directory_to_MP3D_data
```

### How to Run?
The code can do (a) exploring scenes (b) building top-down view semantic maps of MP3D scenes (c) building occupancy maps of MP3D scenes.
All the parameters are controlled by the configuration file `core/config.py`.
##### Exploring the environment
```
python main_eval.py
```
##### Build top-down view semantic maps
```
python scripts/build_semantic_BEV_map.py
```
##### Build top-down view occupancy maps
```
python scripts/build_occupancy_map_from_continuous_habitat.py
```
