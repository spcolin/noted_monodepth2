# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import torch
import random
import numpy as np

options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    np.random.seed(20)
    random.seed(20)
    torch.backends.cudnn.deterministic = True
    trainer = Trainer(opts)
    trainer.train()
