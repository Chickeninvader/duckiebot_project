import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

import urllib.request

ckpt_path = "/tmp/tmp_trained_models/test/20250216151821/models/model_epoch_2.pth"
device = TorchUtils.get_torch_device(try_to_use_cuda=True)

# restore policy
import urllib.request
policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)
# obs_dict = {"obs": {"observation": torch.tensor(np.random.random((1, 3, 480, 640)), dtype=torch.float32)},
#             "goal": {"observation": torch.tensor(np.random.random((1, 3, 480, 640)), dtype=torch.float32)},}
obs_dict = {"observation": torch.tensor(np.random.random((3, 480, 640)), dtype=torch.float32)}
policy.start_episode()
policy(obs_dict)
print('hi')