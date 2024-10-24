import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

def img_transform_func(img):
    np_array = np.array(img, dtype = np.uint8)
    tensor = torch.from_numpy(np_array).reshape(-1)
    tensor = tensor.float() * 0.02
    return tensor

def target_transform_func(target):
    return torch.tensor(target, dtype=torch.long)

img_transform = transforms.Lambda(img_transform_func)
target_transform = transforms.Lambda(target_transform_func)

def train_autoencoder(
    train_dataset,
    autoencoder,
    epochs,
    batch_size,
    optimizer,
    scheduler = None,
    val_dataset = None,
    corruption = None,
    silent = False,
    eval_epochs = 1,
    eval_callback = None,
    num_workers = 0,
    device = "cuda",
):
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers
    )
    if val_dataset is not None:
        validation_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers
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
            postfix={"epo": epoch, "loss": "%.6f" % 0.0, "vloss": "%.6f" % -1,},
            disable=silent,
        )

        for index, batch in enumerate(data_iterator):
            if (
                isinstance(batch, tuple)
                or isinstance(batch, list)
                and len(batch) in [1, 2]
            ):
                batch = batch[0]
                batch = batch.to(device, non_blocking=True)
                
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
                epo=epoch, loss="%.6f" % loss_value, vloss="%.6f" % validation_loss_value,
            )
            
        if eval_epochs is not None and epoch % eval_epochs == 0:
            if validation_loader is not None:
                validation_output = eval_autoencoder(
                    val_dataset,
                    autoencoder,
                    batch_size,
                    encode=False,
                    silent=True,
                    device=device,
                    num_workers=num_workers
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
                validation_actual = validation_actual.to(device, non_blocking=True)
                validation_output = validation_output.to(device, non_blocking=True)
                validation_loss = loss_function(validation_output, validation_actual)
                validation_loss_value = float(validation_loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    loss="%.6f" % loss_value,
                    vloss="%.6f" % validation_loss_value,
                )
                autoencoder.train()
            else:
                validation_loss_value = -1
                data_iterator.set_postfix(
                    epo=epoch, loss="%.6f" % loss_value, vloss="%.6f" % -1,
                )
            
            if eval_callback is not None:
                eval_callback(
                    epoch,
                    optimizer.param_groups[0]["lr"],
                    loss_value,
                    validation_loss_value,
                )

def eval_autoencoder(
        dataset,
        autoencoder,
        batch_size,
        encode = True,
        silent = False,
        device = "cuda",
        num_workers = 0
    ):
        dataloader = DataLoader(
            dataset, batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers
        )
        data_iterator = tqdm(dataloader, leave=False, unit="batch", disable=silent)
        features = []
        autoencoder.eval()
        for batch in data_iterator:
            if isinstance(batch, tuple) or isinstance(batch, list) and len(batch) in [1, 2]:
                batch = batch[0]
                batch = batch.to(device, non_blocking=True)
            batch = batch.squeeze(1).view(batch.size(0), -1)
            if encode:
                output = autoencoder.encode(batch)
            else:
                output = autoencoder(batch)
            features.append(
                output.detach().cpu()
            )
        return torch.cat(features)