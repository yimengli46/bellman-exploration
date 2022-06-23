from core import cfg
from temp_config_2 import print_cfg

print(cfg)
print('------------------------------------------')
cfg.merge_from_file('configs/test_config.yaml')
cfg.freeze()
#print(cfg)

print_cfg()