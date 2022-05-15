""" Prediction models
"""

from models.networks import get_network_from_options
from .map_predictor_model import OccupancyPredictor

def get_predictor_from_options(cfg):
    return OccupancyPredictor(segmentation_model=get_network_from_options(cfg),
                            map_loss_scale=cfg.MAP.MAP_LOSS_SCALE)