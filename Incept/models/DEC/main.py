import numpy as np
import os
import sys
import torch
import torch.nn as nn

from sklearn.cluster import KMeans
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# current_path = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(current_path)
from .dec_utils import img_transform, target_transform
from .model import DEC, StackedDenoisingAutoEncoder, target_distribution
from Incept.evaluation import acc

class DECTrainer:
    def __init__(self, config, pretrain = True):
        self.config = config
        self.img_transform = img_transform
        self.target_transform = target_transform

        autoencoder = StackedDenoisingAutoEncoder(
            self.config.dims, final_activation=None
        )
        if pretrain:
            model_dir = self.config.output_dir
            autoencoder.load_state_dict(torch.load(os.path.join(model_dir, "autoencoder.pth")))

        self.model = DEC(
            cluster_number=self.config.cluster_num,
            hidden_dimension=self.config.hidden_dim,
            encoder=autoencoder.encoder,
            alpha=1
        ).to(self.config.device)

        self.optimizer = SGD(self.model.parameters(), lr=self.config.train_lr, momentum=self.config.train_momentum)
    
    def train(
        self,
        dataset,
        stopping_delta = 1e-6,
        silent = False,
        update_freq = 10,
    ):
        static_dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.config.num_workers,
        )
        data_iterator = tqdm(
            static_dataloader,
            leave=True,
            unit="batch",
            postfix={
                "epo": -1,
                "acc": "%.4f" % 0.0,
                "lss": "%.8f" % 0.0,
                "dlb": "%.4f" % -1,
            },
            disable=silent,
        )
        kmeans = KMeans(n_clusters=self.model.cluster_number, n_init=20)
        self.model.train()
        features = []
        actual = []
        # form initial cluster centres
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch  # if we have a prediction label, separate it to actual
                actual.append(value)

            batch = batch.to(self.config.device, non_blocking=True)
            features.append(self.model.encoder(batch).detach().cpu())
        actual = torch.cat(actual).long()
        predicted = kmeans.fit_predict(torch.cat(features).numpy())
        predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
        # _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
        accuracy = acc(actual.cpu().numpy(), predicted)
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
        )

        cluster_centers = cluster_centers.to(self.config.device, non_blocking=True)
        with torch.no_grad():
            # initialise the cluster centers
            self.model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
        loss_function = nn.KLDivLoss(size_average=False)
        delta_label = None
        for epoch in range(self.config.train_epochs):
            features = []
            data_iterator = tqdm(
                train_dataloader,
                leave=True,
                unit="batch",
                postfix={
                    "epo": epoch,
                    "acc": "%.4f" % (accuracy or 0.0),
                    "lss": "%.8f" % 0.0,
                    "dlb": "%.4f" % (delta_label or 0.0),
                },
                disable=silent,
            )
            self.model.train()
            for index, batch in enumerate(data_iterator):
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(
                    batch
                ) == 2:
                    batch, _ = batch  # if we have a prediction label, strip it away

                batch = batch.to(self.config.device, non_blocking=True)
                output = self.model(batch)
                target = target_distribution(output).detach()
                loss = loss_function(output.log(), target) / output.shape[0]
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % float(loss.item()),
                    dlb="%.4f" % (delta_label or 0.0),
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(closure=None)
                features.append(self.model.encoder(batch).detach().cpu())
                if update_freq is not None and index % update_freq == 0:
                    loss_value = float(loss.item())
                    data_iterator.set_postfix(
                        epo=epoch,
                        acc="%.4f" % (accuracy or 0.0),
                        lss="%.8f" % loss_value,
                        dlb="%.4f" % (delta_label or 0.0),
                    )
            predicted, actual = self.predict(
                dataset,
                silent=True,
                return_actual=True,
            )
            delta_label = (
                float((predicted != predicted_previous).float().sum().item())
                / predicted_previous.shape[0]
            )
            if stopping_delta is not None and delta_label < stopping_delta:
                print(
                    'Early stopping as label delta "%1.5f" less than "%1.5f".'
                    % (delta_label, stopping_delta)
                )
                break
            predicted_previous = predicted
            accuracy = acc(actual.cpu().numpy(), predicted.cpu().numpy())
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % 0.0,
                dlb="%.4f" % (delta_label or 0.0),
            )

    def predict(
        self,
        dataset,
        silent = False,
        return_actual = False,
    ):
        dataloader = DataLoader(
            dataset, batch_size=self.config.eval_batch_size, shuffle=False,
            pin_memory=True, num_workers=self.config.num_workers,
        )
        data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent)
        features = []
        actual = []
        self.model.eval()
        for batch in data_iterator:
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, value = batch
                if return_actual:
                    actual.append(value)
            elif return_actual:
                raise ValueError(
                    "Dataset has no actual value to unpack, but return_actual is set."
                )
            
            batch = batch.to(self.config.device, non_blocking=True)
            features.append(
                self.model(batch).detach().cpu()
            )
        if return_actual:
            return torch.cat(features).max(1)[1], torch.cat(actual).long()
        else:
            return torch.cat(features).max(1)[1]

        
# import sys
# sys.path.append("/data2/liangguanbao/opendeepclustering/Incept")
# from Incept.evaluation import acc
# from Incept.utils import load_config, seed_everything
# from Incept.utils.data import CommonDataset

# seed_everything(42)
# config = load_config("/data2/liangguanbao/opendeepclustering/Incept/Incept/configs/DEC/DEC_Mnist.yaml")

# ds_train = CommonDataset(
#     config.dataset_name, config.data_dir, True,
#     img_transform, target_transform,
# )

# trainer = DECTrainer(config)
# # dec_optimizer = SGD(trainer.model.parameters(), lr=0.01, momentum=0.9)
# trainer.train(
#     dataset=ds_train,
#     stopping_delta=0.000001,
# )

# predicted, actual = trainer.predict(
#     ds_train, model, 1024, silent=True, return_actual=True, cuda=config.device
# )
# actual = actual.cpu().numpy()
# predicted = predicted.cpu().numpy()
# reassignment, accuracy = cluster_accuracy(actual, predicted)
# print("Final DEC accuracy: %s" % accuracy)