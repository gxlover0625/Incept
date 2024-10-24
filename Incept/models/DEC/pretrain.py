import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Optional, Tuple

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)
from model import DenoisingAutoencoder, StackedDenoisingAutoEncoder
from dec_utils import img_transform, target_transform, train_autoencoder, eval_autoencoder

class DECPretrainer:
    def __init__(self, config):
        self.config = config
        self.autoencoder = StackedDenoisingAutoEncoder(
            config.dims, final_activation=None
        ).to(config.device)

    def pretrain_autoencoder(
        self,
        train_dataset,
        val_dataset = None,
        silent = False,
        eval_callback = None,
    ):
        corruption = self.config.dropout_rate
        current_dataset = train_dataset
        current_validation = val_dataset
        number_of_subautoencoders = len(self.autoencoder.dimensions) - 1
        for index in range(number_of_subautoencoders):
            encoder, decoder = self.autoencoder.get_stack(index)
            embedding_dimension = self.autoencoder.dimensions[index]
            hidden_dimension = self.autoencoder.dimensions[index + 1]
            # manual override to prevent corruption for the last subautoencoder
            if index == (number_of_subautoencoders - 1):
                corruption = None
            # initialise the subautoencoder
            sub_autoencoder = DenoisingAutoencoder(
                embedding_dimension=embedding_dimension,
                hidden_dimension=hidden_dimension,
                activation=torch.nn.ReLU()
                if index != (number_of_subautoencoders - 1)
                else None,
                corruption=nn.Dropout(corruption) if corruption is not None else None,
            )
            sub_autoencoder.to(self.config.device)
            ae_optimizer = SGD(params=sub_autoencoder.parameters(), lr=self.config.lr, momentum=self.config.momentum)
            ae_scheduler = StepLR(ae_optimizer, self.config.scheduler_step, gamma=self.config.scheduler_gamma)

            train_autoencoder(
                current_dataset,
                sub_autoencoder,
                self.config.pretrain_epochs,
                self.config.batch_size,
                ae_optimizer,
                scheduler=ae_scheduler,
                val_dataset=current_validation,
                corruption=None,  # already have dropout in the DAE
                silent=silent,
                eval_epochs=self.config.eval_epochs,
                eval_callback=eval_callback,
                num_workers=self.config.num_workers,
                device=self.config.device,
            )

            sub_autoencoder.copy_weights(encoder, decoder)
            if index != (number_of_subautoencoders - 1):
                current_dataset = TensorDataset(
                    eval_autoencoder(
                        current_dataset,
                        sub_autoencoder,
                        self.config.batch_size,
                        encode=True,
                        silent=silent,
                        device=self.config.device,
                        num_workers=self.config.num_workers
                    )
                )

                if current_validation is not None:
                    current_validation = TensorDataset(
                        eval_autoencoder(
                            current_validation,
                            sub_autoencoder,
                            self.config.batch_size,
                            encode=True,
                            silent=silent,
                            device=self.config.device,
                            num_workers=self.config.num_workers
                        )
                    )
            else:
                current_dataset = None
                current_validation = None
        
    def finetuning_autoencoder(
        self
    ):
        pass

        
import sys
sys.path.append("/data2/liangguanbao/opendeepclustering/Incept")
from Incept.utils import load_config, seed_everything
from Incept.utils.data import CommonDataset

seed_everything(42)
config = load_config("/data2/liangguanbao/opendeepclustering/Incept/Incept/configs/DEC/DEC_Mnist.yaml")

ds_train = CommonDataset(
    config.dataset_name,
    config.data_dir,
    True,
    img_transform,
    target_transform,
)

ds_val = CommonDataset(
    config.dataset_name,
    config.data_dir,
    False,
    img_transform,
    target_transform,
)

trainer = DECPretrainer(config)
trainer.pretrain_autoencoder(
    ds_train,
    val_dataset=ds_val,
)
ae_optimizer = SGD(params=trainer.autoencoder.parameters(), lr=0.1, momentum=0.9)
train_autoencoder(
    ds_train,
    trainer.autoencoder,
    val_dataset=ds_val,
    epochs=config.finetune_epochs,
    batch_size=config.batch_size,
    optimizer=ae_optimizer,
    scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
    corruption=0.2,
    eval_callback=None,
)

# 保存autoencoder
if not os.path.exists(config.output_dir):
    os.mkdir(config.output_dir)
torch.save(trainer.autoencoder.state_dict(), os.path.join(config.output_dir, "autoencoder.pth"))

# 加载autoencoder
# autoencoder = StackedDenoisingAutoEncoder(
#     [28 * 28, 500, 500, 2000, 10], final_activation=None
# )
# autoencoder.load_state_dict(torch.load(os.path.join(config.output_dir, "autoencoder.pth")))
# print(autoencoder)