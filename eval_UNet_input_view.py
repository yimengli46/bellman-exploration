import os
import numpy as np
from modeling.utils.UNet import UNet
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
from sseg_utils.metrics import Evaluator
import matplotlib.pyplot as plt
from dataloader_input_partial_map import get_all_scene_dataset, my_collate
import torch.utils.data as data
import torch
import torch.nn as nn
from core import cfg
from collections import OrderedDict

#======================================================================================
#cfg.merge_from_file('configs/exp_train_input_partial_map_occ_and_sem.yaml')
cfg.merge_from_file('configs/exp_train_input_partial_map_occ_only.yaml')
cfg.freeze()

device = torch.device('cuda')

def L1Loss(logit, target):
	mask_zero = (target > 0)
	logit = logit * mask_zero
	num_nonzero = torch.sum(mask_zero) + 1.
	#print(f'num_nonzero = {num_nonzero}')

	#result = loss(logit, target)
	result = (torch.abs(logit - target)).sum() / num_nonzero

	return result

criterion = L1Loss

#=========================================================== Define Dataloader ==================================================
data_folder = cfg.PRED.PARTIAL_MAP.GEN_SAMPLES_SAVED_FOLDER
dataset_val = get_all_scene_dataset('val', cfg.MAIN.VAL_SCENE_LIST, data_folder)
dataloader_val = data.DataLoader(dataset_val, 
	batch_size=cfg.PRED.PARTIAL_MAP.BATCH_SIZE, 
	num_workers=cfg.PRED.PARTIAL_MAP.NUM_WORKERS,
	shuffle=False,
	collate_fn=my_collate,
	)

#================================================================================================================================
# Define network
model = UNet(n_channel_in=cfg.PRED.PARTIAL_MAP.INPUT_CHANNEL, n_class_out=cfg.PRED.PARTIAL_MAP.OUTPUT_CHANNEL)
model = model.cuda()

#checkpoint = torch.load(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/{cfg.PRED.PARTIAL_MAP.INPUT}/experiment_29/best_checkpoint.pth.tar', map_location=device)
checkpoint = torch.load(f'{cfg.PRED.PARTIAL_MAP.SAVED_FOLDER}/{cfg.PRED.PARTIAL_MAP.INPUT}/experiment_4/best_checkpoint.pth.tar', map_location=device)
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
	images, targets, frontiers = batch['input'], batch['output'], batch['frontiers']
	#print('images = {}'.format(images))
	#print('targets = {}'.format(targets))
	images, targets = images.cuda(), targets.cuda()

	#========================== compute loss =====================
	with torch.no_grad():
		output = model(images)

	#assert 1==2
	
	loss = criterion(output, targets)

	test_loss += loss.item()
	print(f'loss = {loss.item()}')

	iter_num += 1








