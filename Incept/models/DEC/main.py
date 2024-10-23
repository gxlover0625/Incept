import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.cluster import KMeans
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Optional, Tuple

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from data_transform import CachedMNIST
from model import DenoisingAutoencoder, StackedDenoisingAutoEncoder

class DECTrainer:
    def __init__(self, config):
        self.config = config
    
    def predict(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

import sys
sys.path.append("/data2/liangguanbao/opendeepclustering/Incept")
from Incept.utils import load_config, seed_everything

seed_everything(42)

config = load_config("/data2/liangguanbao/opendeepclustering/Incept/Incept/configs/DEC/DEC_Mnist.yaml")

# 加载autoencoder
autoencoder = StackedDenoisingAutoEncoder(
    [28 * 28, 500, 500, 2000, 10], final_activation=None
)
autoencoder.load_state_dict(torch.load(os.path.join(config.output_dir, "autoencoder.pth")))




