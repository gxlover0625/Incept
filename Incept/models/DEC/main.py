import numpy as np
import os
import sys
import torch
import torch.nn as nn

from sklearn.cluster import KMeans
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dec_utils import img_transform, target_transform
from .model import DEC, StackedDenoisingAutoEncoder, target_distribution
from Incept.evaluation import Evaluator
from Incept.utils.early_stopping import EarlyStopping
from Incept.utils.logger import BasicLogger

class DECTrainer:
    def __init__(self, config, pretrained = True, strategy = "earlystop", logger="basic"):
        self.config = config
        self.pretrained = pretrained

        # dataset processing
        self.img_transform = img_transform
        self.target_transform = target_transform

        # evaluation component
        self.evaluator = Evaluator(metrics=["acc", "nmi", "ari"])

        # stop strategy
        if strategy == "earlystop":
            self.early_stopper = EarlyStopping()
        else:
            self.early_stopper = None
        
        # log component
        if logger == "basic":
            self.logger = BasicLogger(backends=["json", "tensorboard"], log_dir=self.config.output_dir)
        else:
            self.logger = None
    
    def setup(self):
        # model
        autoencoder = StackedDenoisingAutoEncoder(
            self.config.dims, final_activation=None
        )
        if self.pretrained:
            model_dir = self.config.output_dir
            autoencoder.load_state_dict(torch.load(os.path.join(model_dir, "autoencoder.pth")))

        self.model = DEC(
            cluster_number=self.config.cluster_num,
            hidden_dimension=self.config.hidden_dim,
            encoder=autoencoder.encoder,
            alpha=1
        ).to(self.config.device)

        # optimizer
        self.optimizer = SGD(self.model.parameters(), lr=self.config.train_lr, momentum=self.config.train_momentum)
        
    def train(
        self,
        dataset,
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
            desc="Initialize the cluster centers",
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
        results = self.evaluator.eval(actual.cpu().numpy(), predicted)
        acc, nmi, ari = results["acc"], results["nmi"], results["ari"]
        cluster_centers = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
        )

        cluster_centers = cluster_centers.to(self.config.device, non_blocking=True)
        with torch.no_grad():
            # initialise the cluster centers
            self.model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)

        loss_function = nn.KLDivLoss(size_average=False)
        delta_label = None
        # training
        for epoch in range(self.config.train_epochs):
            features = []
            data_iterator = tqdm(
                train_dataloader,
                leave=True,
                unit="batch",
                postfix={
                    "epoch": epoch,
                    "loss": "%.8f" % 0.0,
                    "delta": "%.4f" % (delta_label or 0.0),
                    "acc": "%.4f" % (acc or 0.0),
                    "nmi": "%.4f" % (nmi or 0.0),
                    "ari": "%.4f" % (ari or 0.0),
                },
                disable=False,
            )
            self.model.train()
            # batch training
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
                    epoch=epoch,
                    loss="%.8f" % float(loss.item()),
                    delta="%.4f" % (delta_label or 0.0),
                    acc="%.4f" % (acc or 0.0),
                    nmi="%.4f" % (nmi or 0.0),
                    ari="%.4f" % (ari or 0.0),
                )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(closure=None)
                features.append(self.model.encoder(batch).detach().cpu())
            
            # validation for each epoch
            predicted, actual = self.predict(
                dataset,
                return_actual=True,
            )
            results = self.evaluator.eval(actual.cpu().numpy(), predicted.cpu().numpy())
            acc, nmi, ari = results["acc"], results["nmi"], results["ari"]
            
            # set the early stop
            if self.early_stopper is not None:
                self.early_stopper(predicted, predicted_previous)
                delta_label = self.early_stopper.delta_label

                if self.early_stopper.is_early_stop:
                    print("Early stopping triggered.")
                    if self.logger is not None:
                        self.logger.log(
                            epoch=epoch,
                            loss=float(loss.item()),
                            delta_label=delta_label,
                            acc=acc,
                            nmi=nmi,
                            ari=ari,
                        )
                        self.logger.update(
                            epoch=epoch,
                            acc=acc,
                            nmi=nmi,
                            ari=ari,
                            model=self.model,
                        )
                    break

            predicted_previous = predicted
            data_iterator.set_postfix(
                epoch=epoch,
                loss="%.8f" % float(loss.item()),
                delta="%.4f" % (delta_label or 0.0),
                acc="%.4f" % (acc or 0.0),
                nmi="%.4f" % (nmi or 0.0),
                ari="%.4f" % (ari or 0.0),
            )

            if self.logger is not None:
                self.logger.log(
                    epoch=epoch,
                    loss=float(loss.item()),
                    delta_label=delta_label,
                    acc=acc,
                    nmi=nmi,
                    ari=ari,
                )
                self.logger.update(
                    epoch=epoch,
                    acc=acc,
                    nmi=nmi,
                    ari=ari,
                    model=self.model,
                )
        
        if self.logger:
            self.logger.summary()

    def predict(
        self,
        dataset,
        return_actual = False,
    ):
        dataloader = DataLoader(
            dataset, batch_size=self.config.eval_batch_size, shuffle=False,
            pin_memory=True, num_workers=self.config.num_workers,
        )
        data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=True)
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