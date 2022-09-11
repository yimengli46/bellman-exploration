import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import numpy as np
from .distributions_utils import Categorical, DiagGaussian
from .model_utils import get_grid, ChannelPool, Flatten, NNBase
from core import cfg
from modeling.utils.baseline_utils import crop_map
from skimage.draw import line
import scipy.ndimage
import matplotlib.pyplot as plt

# Global Policy model code
class Global_Policy(NNBase):

    def __init__(self, input_shape, recurrent=False, hidden_size=512,
                 downscaling=1):
        super(Global_Policy, self).__init__(recurrent, hidden_size,
                                            hidden_size)

        out_size = int(input_shape[1] / 16. * input_shape[2] / 16.)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(8, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten()
        )

        self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)
        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        orientation_emb = self.orientation_emb(extras).squeeze(1)
        x = torch.cat((x, orientation_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs

# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):

    def __init__(self, obs_shape, device, model_type=0,
                 base_kwargs=None):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 0:
            self.network = Global_Policy(obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        
        num_outputs = 2
        self.dist = DiagGaussian(self.network.output_size, num_outputs, device)

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


def compute_long_term_goal(occupancy_map, agent_map_coords, theta, traverse_coords_list, g_policy, device):
    H, W = occupancy_map.shape
    M_p = np.zeros((1, 4, H, W), dtype='float32')
    M_p[0, 0, :, :] = (occupancy_map == cfg.FE.COLLISION_VAL) # first channel obstacle
    M_p[0, 1, :, :] = (occupancy_map != cfg.FE.UNOBSERVED_VAL) # second channel explored
    M_p[0, 2, agent_map_coords[1]-1:agent_map_coords[1]+2, agent_map_coords[0]-1:agent_map_coords[0]+2] = 1 # third channel current location
    for coords in traverse_coords_list:
        M_p[0, 3, coords[1]-1:coords[1]+2, coords[0]-1: coords[0]+2] = 1 # fourth channel visited places
    tensor_M_p = torch.tensor(M_p).float()
    #print(f'tensor_M_p.shape = {tensor_M_p.shape}')

    '''
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
    ax[0][0].imshow(M_p[0, 0, :, :], cmap='gray')
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].get_yaxis().set_visible(False)
    ax[0][0].set_title('input: occupancy_map_Mp')
    
    ax[0][1].imshow(M_p[0, 1, :, :])
    ax[0][1].get_xaxis().set_visible(False)
    ax[0][1].get_yaxis().set_visible(False)
    ax[0][1].set_title('input: semantic_map_Mp')
    ax[1][0].imshow(M_p[0, 2, :, :])
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title('output: U_a')
    ax[1][1].imshow(M_p[0, 3, :, :])
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_title('output: U_dall')
    fig.tight_layout()
    plt.show()
    '''

    #================== crop out the map centered at the agent ==========================
    _, _, H, W = M_p.shape
    Wby2, Hby2 = W // 2, H // 2
    tform_trans = torch.Tensor([[agent_map_coords[0] - Wby2, agent_map_coords[1] - Hby2, 0]])
    crop_center = torch.Tensor([[W / 2.0, H / 2.0]]) + tform_trans[:, :2]
    # Crop out the appropriate size of the map
    local_map_size = int(240)
    tensor_local_M_p = crop_map(tensor_M_p, crop_center, local_map_size, 'nearest')
    global_map_size = int(480)
    tensor_global_M_p = crop_map(tensor_M_p, crop_center, global_map_size, 'nearest')

    local_w = 240
    local_h = 240
    num_scenes = 1
    global_downscaling = 2

    global_input = torch.zeros(num_scenes, 8, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()
    global_orientation[0] = int(90 / 5.)

    global_input[:, 0:4, :, :] = tensor_local_M_p
    global_input[:, 4:, :, :] = nn.MaxPool2d(global_downscaling)(tensor_global_M_p)

    global_input = global_input.to(device)
    global_orientation = global_orientation.to(device)

    with torch.no_grad():
        # Run Global Policy (global_goals = Long-Term Goal)
        _, g_action, g_action_log_prob, _ = \
            g_policy.act(
                global_input,
                None,
                None,
                extras=global_orientation,
                deterministic=False
            )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
    global_goals_coords = (agent_map_coords[0]+global_goals[0][0], agent_map_coords[1]+global_goals[0][1]) 

    '''
    # find reachable global_goals
    rr_line, cc_line = line(agent_map_coords[1], agent_map_coords[0], global_goals_coords[1], global_goals_coords[0])
    # make sure the points are inside the map
    mask_line = np.logical_and(np.logical_and(rr_line >= 0, rr_line < H), 
        np.logical_and(cc_line >= 0, cc_line < W))
    rr_line = rr_line[mask_line]
    cc_line = cc_line[mask_line]
    binary_occupancy_map = occupancy_map.copy()
    binary_occupancy_map[binary_occupancy_map == cfg.FE.UNOBSERVED_VAL] = cfg.FE.COLLISION_VAL
    binary_occupancy_map[binary_occupancy_map == cfg.FE.COLLISION_VAL] = 0
    labels, nb = scipy.ndimage.label(binary_occupancy_map, structure=np.ones((3,3)))
    agent_label = labels[agent_map_coords[1], agent_map_coords[0]]
    #print(f'len(rr_line) = {len(rr_line)}')
    for idx in list(reversed(range(len(rr_line)))):
        if labels[rr_line[idx], cc_line[idx]] == agent_label:
            reachable_global_goals_coords = (cc_line[idx], rr_line[idx])
            break
    '''

    return global_goals_coords