import easydict
import numpy as np
import os
import random
import torch
import yaml

def seed_everything(seed = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        config = easydict.EasyDict(config)
    return config