import os
import numpy as np
from modeling.utils.ResNet import ResNet
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
from sseg_utils.metrics import Evaluator
import matplotlib.pyplot as plt
from dataloader_input_view import get_all_view_dataset
import torch.utils.data as data
import torch
import torch.nn as nn
from core import cfg
from collections import OrderedDict

#======================================================================================
#cfg.merge_from_file('configs/exp_train_input_view_depth_and_sem.yaml')
cfg.merge_from_file('configs/exp_train_input_view_depth_only.yaml')
cfg.freeze()

device = torch.device('cuda')

criterion = nn.L1Loss()

#=========================================================== Define Dataloader ==================================================
data_folder = cfg.PRED.VIEW.GEN_SAMPLES_SAVED_FOLDER
dataset_val = get_all_view_dataset('val', cfg.MAIN.VAL_SCENE_LIST, data_folder)
dataloader_val = data.DataLoader(dataset_val, 
	batch_size=cfg.PRED.VIEW.BATCH_SIZE, 
	num_workers=cfg.PRED.VIEW.NUM_WORKERS,
	shuffle=False,
	)

#================================================================================================================================
# Define network
model = ResNet(n_channel_in=cfg.PRED.VIEW.INPUT_CHANNEL, n_class_out=cfg.PRED.VIEW.OUTPUT_CHANNEL)
model = model.cuda()

checkpoint = torch.load(f'{cfg.PRED.VIEW.SAVED_FOLDER}/{cfg.PRED.VIEW.INPUT}/experiment_1/checkpoint.pth.tar', map_location=device)
new_state_dict = OrderedDict()
for k, v in checkpoint['state_dict'].items():
	name = k[7:] #remove 'module'
	new_state_dict[name] = v
model.load_state_dict(new_state_dict)
#assert 1==2

#======================================================== evaluation stage =====================================================
model.eval()
test_loss = 0.0
iter_num = 0

for batch in dataloader_val:
	print(f'iter_num = {iter_num}')
	images, targets = batch['input'], batch['output']
	#print('images = {}'.format(images))
	#print('targets = {}'.format(targets))
	images, targets = images.cuda(), targets.cuda()

	#========================== compute loss =====================
	with torch.no_grad():
		output = model(images)
	
	loss = criterion(output, targets)

	test_loss += loss.item()
	print(f'loss = {loss.item()}')

	iter_num += 1








