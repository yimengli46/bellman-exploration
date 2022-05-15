

from .resnetUnet import ResNetUNet

'''
Model ResNetUnet taken from:
https://github.com/usuyama/pytorch-unet
'''

def get_network_from_options(cfg):
    """ Gets the network given the options
    """
    return ResNetUNet(n_channel_in=cfg.MAP.N_SPATIAL_CLASSES, n_class_out=cfg.MAP.N_SPATIAL_CLASSES)