import os
import shutil
import torch
from collections import OrderedDict
import glob
from core import cfg

class Saver(object):

    def __init__(self, output_folder):
        if cfg.PRED.TYPE == 'MAP':
            self.directory = os.path.join(output_folder, cfg.PRED.PARTIAL_MAP.INPUT)
        elif cfg.PRED.TYPE == 'VIEW':
            self.directory = os.path.join(output_folder, cfg.PRED.VIEW.INPUT)
        self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
        run_id = len(self.runs) if self.runs else 0

        self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))


