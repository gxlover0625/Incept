import os
import sys
import torch
import torch.nn as nn

from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter

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
        
    def finetune_autoencoder(
        self,
        train_dataset,
        val_dataset = None,
        silent = False,
        eval_callback = None,
    ):
        ae_optimizer = SGD(params=self.autoencoder.parameters(), lr=self.config.lr, momentum=self.config.momentum)
        ae_scheduler = StepLR(ae_optimizer, self.config.scheduler_step, gamma=self.config.scheduler_gamma)
        train_autoencoder(
            train_dataset,
            self.autoencoder,
            epochs=self.config.finetune_epochs,
            batch_size=self.config.batch_size,
            optimizer=ae_optimizer,
            scheduler=ae_scheduler,
            val_dataset=val_dataset,
            corruption=self.config.dropout_rate,
            silent=silent,
            eval_epochs=self.config.eval_epochs,
            eval_callback=eval_callback,
            num_workers=self.config.num_workers,
            device=self.config.device
        )
    
    def train(
        self,
        train_dataset,
        val_dataset = None,
        eval_callback = None
    ):
        print("Pretraining stage.")
        self.pretrain_autoencoder(
            train_dataset,
            val_dataset,
            eval_callback=None
        )
        print("Training stage.")
        self.finetune_autoencoder(
            train_dataset,
            val_dataset,
            eval_callback=eval_callback
        )
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir, exist_ok=True)
        torch.save(self.autoencoder.state_dict(), os.path.join(self.config.output_dir, "autoencoder.pth"))

        
import sys
sys.path.append("/data2/liangguanbao/opendeepclustering/Incept")
from Incept.utils import load_config, seed_everything
from Incept.utils.data import CommonDataset

seed_everything(42)
config = load_config("/data2/liangguanbao/opendeepclustering/Incept/Incept/configs/DEC/DEC_Mnist.yaml")

writer = SummaryWriter(log_dir="/data2/liangguanbao/opendeepclustering/saves/DEC/mnist")
def training_callback(epoch, lr, loss, validation_loss):
    writer.add_scalars(
        "data/autoencoder",
        {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
        epoch,
    )

ds_train = CommonDataset(
    config.dataset_name, config.data_dir, True,
    img_transform, target_transform,
)

ds_val = CommonDataset(
    config.dataset_name, config.data_dir, False,
    img_transform, target_transform,
)

trainer = DECPretrainer(config)
trainer.train(ds_train, ds_val, training_callback)