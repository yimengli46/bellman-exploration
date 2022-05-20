import os
import numpy as np
from UNet import UNet
from sseg_utils.loss import SegmentationLosses
from sseg_utils.saver import Saver
from sseg_utils.summaries import TensorboardSummary
from sseg_utils.metrics import Evaluator
from sseg_utils.lr_scheduler import PolyLR
import matplotlib.pyplot as plt
from dataloader_MP3D import MP3DDataset
import torch.utils.data as data
import torch
from core import cfg
from itertools import islice

loss = torch.nn.MSELoss()

def MSELoss(logit, target):
    mask_zero = (target > 0)
    logit = logit * mask_zero
    num_nonzero = torch.sum(mask_zero)
    #print(f'num_nonzero = {num_nonzero}')

    #result = loss(logit, target)
    result = ((logit - target)**2).sum() / num_nonzero

    return result

#=========================================================== Define Saver =======================================================
saver = Saver()
# Define Tensorboard Summary
summary = TensorboardSummary(saver.experiment_dir)
writer = summary.create_summary()

#=========================================================== Define Dataloader ==================================================
#scene_name = '17DRP5sb8fy_0'
dataset_train = MP3DDataset(scene_names=cfg.MAIN.TRAIN_SCENE_LIST, worker_size=cfg.PRED.NUM_WORKERS, seed=cfg.GENERAL.RANDOM_SEED)
dataloader_train = data.DataLoader(dataset_train, batch_size=cfg.PRED.BATCH_SIZE, num_workers=cfg.PRED.NUM_WORKERS)

dataset_val = MP3DDataset(scene_names=cfg.MAIN.TRAIN_SCENE_LIST, worker_size=0, seed=cfg.GENERAL.RANDOM_SEED, num_elems=10000)
dataloader_val = data.DataLoader(dataset_train, batch_size=cfg.PRED.BATCH_SIZE, num_workers=0)

#================================================================================================================================
# Define network
model = UNet(n_channel_in=2, n_class_out=1).cuda()

#=========================================================== Define Optimizer ================================================
import torch.optim as optim
train_params = [{'params': model.parameters(), 'lr': cfg.PRED.LR}]
optimizer = optim.SGD(train_params, lr=cfg.PRED.LR, momentum=0.9, weight_decay=1e-4)
scheduler = PolyLR(optimizer, 10000, power=0.9)

# Define Criterion
# whether to use class balanced weights
weight = None
criterion = MSELoss

# Define Evaluator
#evaluator = Evaluator(num_classes)

#===================================================== Resuming checkpoint ====================================================
best_pred = 0.0
if cfg.PRED.RESUME != '':
    if not os.path.isfile(cfg.PRED.RESUME):
        raise RuntimeError("=> no checkpoint found at '{}'" .format(cfg.PRED.RESUME))
    checkpoint = torch.load(cfg.PRED.RESUME)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_pred = checkpoint['best_pred']
    print("=> loaded checkpoint '{}' (epoch {})".format(cfg.PRED.RESUME, checkpoint['epoch']))

#=================================================================trainin
for epoch in range(cfg.PRED.EPOCHS):
    train_loss = 0.0
    model.train()
    num_img_tr = cfg.PRED.NUM_ITER_TRAIN
    iter_num = 0
    
    for batch in islice(dataloader_train, num_img_tr):
        print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
        images, targets = batch[0], batch[1]
        #print('images = {}'.format(images.shape))
        #print('targets = {}'.format(targets.shape))
        #assert 1==2
        images, targets = images.cuda(), targets.cuda()
        
        #================================================ compute loss =============================================
        output = model(images) # batchsize x 1 x H x W
        #print(f'output.shape = {output.shape}')
        loss = criterion(output, targets)

        #================================================= compute gradient =================================================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        print('Train loss: %.3f' % (train_loss / (iter_num + 1)))
        writer.add_scalar('train/total_loss_iter', loss.item(), iter_num + num_img_tr * epoch)

        iter_num += 1

    writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
    print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * cfg.PRED.BATCH_SIZE + images.data.shape[0]))
    print('Loss: %.3f' % train_loss)

#======================================================== evaluation stage =====================================================

    if epoch % cfg.PRED.EVAL_INTERVAL == 0:
        model.eval()
        #evaluator.reset()
        test_loss = 0.0
        iter_num = 0
        num_img_ts = cfg.PRED.NUM_ITER_EVAL

        for batch in islice(dataloader_val, num_img_ts):
            print('epoch = {}, iter_num = {}'.format(epoch, iter_num))
            images, targets = batch[0], batch[1]
            #print('images = {}'.format(images))
            #print('targets = {}'.format(targets))
            images, targets = images.cuda(), targets.cuda()

            #========================== compute loss =====================
            with torch.no_grad():
                output = model(images)
            loss = criterion(output, targets)

            test_loss += loss.item()
            print('Test loss: %.3f' % (test_loss / (iter_num + 1)))
            pred = output.data.cpu().numpy()
            targets = targets.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            #evaluator.add_batch(targets, pred)

            iter_num += 1

        # Fast test during the training
        #Acc = evaluator.Pixel_Accuracy()
        #Acc_class = evaluator.Pixel_Accuracy_Class()
        #mIoU = evaluator.Mean_Intersection_over_Union()
        #FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        #writer.add_scalar('val/mIoU', mIoU, epoch)
        #writer.add_scalar('val/Acc', Acc, epoch)
        #writer.add_scalar('val/Acc_class', Acc_class, epoch)
        #writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter_num * cfg.PRED.BATCH_SIZE + images.data.shape[0]))
        #print("Acc:{:.5}, Acc_class:{:.5}, mIoU:{:.5}, fwIoU: {:.5}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)

        #new_pred = mIoU
        if True: #new_pred > best_pred:
            is_best = True
            #best_pred = new_pred
            saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': test_loss,
            }, is_best)

trainer.writer.close()







