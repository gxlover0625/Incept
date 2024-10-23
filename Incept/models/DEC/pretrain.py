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
from data_transform import CachedMNIST
from model import DenoisingAutoencoder, StackedDenoisingAutoEncoder

def img_transform(img):
    np_array = np.array(img, dtype = np.uint8)
    tensor = torch.from_numpy(np_array).reshape(-1)
    tensor = tensor.float() * 0.02
    return tensor

def target_transform(target):
    return torch.tensor(target, dtype=torch.long)

class DECPretrainer:
    def __init__(self, config):
        self.config = config
        self.autoencoder = StackedDenoisingAutoEncoder(
            config.dims, final_activation=None
        ).to(config.device)

    def pretrain(
        self,
        dataset,
        # autoencoder: StackedDenoisingAutoEncoder,
        epochs: int,
        batch_size: int,
        optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer],
        scheduler: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
        validation: Optional[torch.utils.data.Dataset] = None,
        corruption: Optional[float] = None,
        cuda: bool = True,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        silent: bool = False,
        update_freq: Optional[int] = 1,
        update_callback: Optional[Callable[[float, float], None]] = None,
        num_workers: Optional[int] = None,
        epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    ):
        current_dataset = dataset
        current_validation = validation
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
            if cuda:
                sub_autoencoder = sub_autoencoder.cuda()
            ae_optimizer = optimizer(sub_autoencoder)
            ae_scheduler = scheduler(ae_optimizer) if scheduler is not None else scheduler

            self.train(
                current_dataset,
                sub_autoencoder,
                epochs,
                batch_size,
                ae_optimizer,
                validation=current_validation,
                corruption=None,  # already have dropout in the DAE
                scheduler=ae_scheduler,
                cuda=cuda,
                sampler=sampler,
                silent=silent,
                update_freq=update_freq,
                update_callback=update_callback,
                num_workers=num_workers,
                epoch_callback=epoch_callback,
            )

            sub_autoencoder.copy_weights(encoder, decoder)
            if index != (number_of_subautoencoders - 1):
                current_dataset = TensorDataset(
                    self.predict(
                        current_dataset,
                        sub_autoencoder,
                        batch_size,
                        cuda=cuda,
                        silent=silent,
                    )
                )

                if current_validation is not None:
                    current_validation = TensorDataset(
                        self.predict(
                            current_validation,
                            sub_autoencoder,
                            batch_size,
                            cuda=cuda,
                            silent=silent,
                        )
                    )
            else:
                current_dataset = None  # minor optimisation on the last subautoencoder
                current_validation = None

    def train(
        self,
        dataset: torch.utils.data.Dataset,
        autoencoder: torch.nn.Module,
        epochs: int,
        batch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any = None,
        validation: Optional[torch.utils.data.Dataset] = None,
        corruption: Optional[float] = None,
        cuda: bool = True,
        sampler: Optional[torch.utils.data.sampler.Sampler] = None,
        silent: bool = False,
        update_freq: Optional[int] = 1,
        update_callback: Optional[Callable[[float, float], None]] = None,
        num_workers: Optional[int] = None,
        epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
    ):
        # self.pretrain()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=False,
            sampler=sampler,
            shuffle=True if sampler is None else False,
            num_workers=num_workers if num_workers is not None else 0,
        )
        if validation is not None:
            validation_loader = DataLoader(
                validation,
                batch_size=batch_size,
                pin_memory=False,
                sampler=None,
                shuffle=False,
            )
        else:
            validation_loader = None

        loss_function = nn.MSELoss()
        autoencoder.train()
        validation_loss_value = -1
        loss_value = 0

        for epoch in range(epochs):
            if scheduler is not None:
                scheduler.step()

            data_iterator = tqdm(
                dataloader,
                leave=True,
                unit="batch",
                postfix={"epo": epoch, "lss": "%.6f" % 0.0, "vls": "%.6f" % -1,},
                disable=silent,
            )

            for index, batch in enumerate(data_iterator):
                if (
                    isinstance(batch, tuple)
                    or isinstance(batch, list)
                    and len(batch) in [1, 2]
                ):
                    batch = batch[0]
                    if cuda:
                        batch = batch.cuda(non_blocking=True)
                    
                    if corruption is not None:
                        output = autoencoder(F.dropout(batch, corruption))
                    else:
                        output = autoencoder(batch)
                                        
                    loss = loss_function(output, batch)
                    loss_value = float(loss.item())
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step(closure=None)
                    data_iterator.set_postfix(
                        epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % validation_loss_value,
                    )
                
                if update_freq is not None and epoch % update_freq == 0:
                    if validation_loader is not None:
                        validation_output = self.predict(
                            validation,
                            autoencoder,
                            batch_size,
                            cuda=cuda,
                            silent=True,
                            encode=False,
                        )

                        validation_inputs = []
                        for val_batch in validation_loader:
                            if (
                                isinstance(val_batch, tuple) or isinstance(val_batch, list)
                            ) and len(val_batch) in [1, 2]:
                                validation_inputs.append(val_batch[0])
                            else:
                                validation_inputs.append(val_batch)
                        validation_actual = torch.cat(validation_inputs)
                        if cuda:
                            validation_actual = validation_actual.cuda(non_blocking=True)
                            validation_output = validation_output.cuda(non_blocking=True)
                        validation_loss = loss_function(validation_output, validation_actual)
                        # validation_accuracy = pretrain_accuracy(validation_output, validation_actual)
                        validation_loss_value = float(validation_loss.item())
                        data_iterator.set_postfix(
                            epo=epoch,
                            lss="%.6f" % loss_value,
                            vls="%.6f" % validation_loss_value,
                        )
                        autoencoder.train()
                    else:
                        validation_loss_value = -1
                        # validation_accuracy = -1
                        data_iterator.set_postfix(
                            epo=epoch, lss="%.6f" % loss_value, vls="%.6f" % -1,
                        )
                    
                    if update_callback is not None:
                        update_callback(
                            epoch,
                            optimizer.param_groups[0]["lr"],
                            loss_value,
                            validation_loss_value,
                        )
            if epoch_callback is not None:
                autoencoder.eval()
                epoch_callback(epoch, autoencoder)
                autoencoder.train()
        
    def predict(
        self,
        dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        batch_size: int,
        cuda: bool = True,
        silent: bool = False,
        encode: bool = True,
    ):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=False, shuffle=False
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent,)
        features = []
        if isinstance(model, torch.nn.Module):
            model.eval()
        for batch in data_iterator:
            if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
                batch = batch[0]
            if cuda:
                batch = batch.cuda(non_blocking=True)
            batch = batch.squeeze(1).view(batch.size(0), -1)
            if encode:
                output = model.encode(batch)
            else:
                output = model(batch)
            features.append(
                output.detach().cpu()
            )  # move to the CPU to prevent out of memory on the GPU
        return torch.cat(features)

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
    transforms.Lambda(img_transform),
    transforms.Lambda(target_transform),
    config.device
)

ds_val = CommonDataset(
    config.dataset_name,
    config.data_dir,
    False,
    transforms.Lambda(img_transform),
    transforms.Lambda(target_transform),
    config.device
)

trainer = DECPretrainer(config)
# autoencoder = StackedDenoisingAutoEncoder(
#     [28 * 28, 500, 500, 2000, 10], final_activation=None
# ).to(config.device)
trainer.pretrain(
    ds_train,
    # autoencoder,
    cuda=config.device,
    validation=ds_val,
    epochs=config.pretrain_epochs,
    batch_size=config.batch_size,
    optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
    scheduler=lambda x: StepLR(x, 100, gamma=0.1),
    corruption=0.2,
)
ae_optimizer = SGD(params=trainer.autoencoder.parameters(), lr=0.1, momentum=0.9)
trainer.train(
    ds_train,
    trainer.autoencoder,
    cuda=config.device,
    validation=ds_val,
    epochs=config.finetune_epochs,
    batch_size=config.batch_size,
    optimizer=ae_optimizer,
    scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
    corruption=0.2,
    update_callback=None,
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