import numpy as np
import torch

from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from cc_utils import Transforms
from model import get_resnet, CC, InstanceLoss, ClusterLoss

from Incept.models import Trainer
from Incept.utils import get_pretrained_resnet

class CCTrainer(Trainer):
    def __init__(self, config, strategy="earlystop", logger_backends=["json", "tensorboard"]):
        super().__init__(config, strategy, logger_backends)
        self.config = config
        # dataset processing
        self.img_transform = Transforms(size=config.image_size, s=0.5)

    def setup(self):
        config = self.config
        if config.use_original_resnet:
            resnet = get_resnet(config.resnet)
        else:
            resnet = get_pretrained_resnet(config.resnet, pretrained=True)
            resnet.rep_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            
        self.model = CC(resnet, config.feature_dim, config.cluster_num)
        self.model.to(config.device)
        self.optimizer = Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        self.criterion_instance = InstanceLoss(config.batch_size, config.instance_temperature)
        self.criterion_cluster = ClusterLoss(config.cluster_num, config.cluster_temperature)
    
    def compute_loss(self, x_i, x_j):
        config = self.config
        x_i = x_i.to(config.device)
        x_j = x_j.to(config.device)
        z_i, z_j, c_i, c_j = self.model(x_i, x_j)
        loss_instance = self.criterion_instance(z_i, z_j)
        loss_cluster = self.criterion_cluster(c_i, c_j)
        loss = loss_instance + loss_cluster
        return loss, loss_instance, loss_cluster

    def train(self, dataset, eval_dataset=None):
        config = self.config
        train_dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=config.num_workers,
        )

        delta_label = -1
        acc, nmi, ari = -1, -1, -1
        predicted_previous = None
        ### Epoch Training
        for epoch in range(self.config.epochs):
            # step1, prepare model
            self.model.train()
            # step2, prepare data
            data_iterator = tqdm(train_dataloader, unit="batch")

            # step3, Batch Training
            ## step3.1, transport data
            for step, ((x_i, x_j), _) in enumerate(data_iterator):
                ## step3.2, calculate loss
                loss, loss_instance, loss_cluster = self.compute_loss(x_i, x_j)
                ## step3.3, update model
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                ## step3.4, update progress
                self.update_postfix(
                    data_iterator, epoch, config.epochs-1, step, len(data_iterator)-1,
                    loss=loss.item(), delta_label=delta_label, acc=acc, nmi=nmi, ari=ari
                )
            
            # step4, evaluation
            if epoch % config.eval_epochs == 0:
                predicted, actual = self.predict(eval_dataset)
                results = self.evaluator.eval(actual, predicted)
                acc, nmi, ari = results["acc"], results["nmi"], results["ari"]

                # step5, early stop
                if predicted_previous is not None and self.early_stopper:
                    self.early_stopper(predicted, predicted_previous)
                    delta_label = self.early_stopper.delta_label
                    if self.early_stopper.is_early_stop:
                        print("Early stopping triggered.")
                        self.log_update(
                            epoch=epoch, loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"],
                            acc=acc, nmi=nmi, ari=ari,
                        )
                        break

                predicted_previous = predicted
                # step6, logging and saving
                self.log_update(
                    epoch=epoch, loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"],
                    acc=acc, nmi=nmi, ari=ari,
                )
        self.summary()

    def predict(self, dataset):
        config = self.config
        dataloader = DataLoader(
            dataset,
            batch_size=config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=config.num_workers,
        )
        data_iterator = tqdm(dataloader, unit="batch")

        features = []
        actual = []
        
        with torch.no_grad():
            self.model.eval()
            for step, (x, y) in enumerate(data_iterator):
                x = x.to(config.device)
                c = self.model.forward_cluster(x)
                features.extend(c.cpu().numpy())
                actual.extend(y.numpy())
        features = np.array(features)
        actual = np.array(actual)
        return features, actual

from Incept.configs import load_exp_config
from Incept.utils import seed_everything
import torchvision
seed_everything(42)
config = load_exp_config("CC", "CIFAR10")
trainer = CCTrainer(config)
train_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir,
    download=True,
    train=True,
    transform=trainer.img_transform,
)
test_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir,
    download=True,
    train=False,
    transform=trainer.img_transform,
)
dataset = ConcatDataset([train_dataset, test_dataset])

eval_train_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir,
    download=True,
    train=True,
    transform=trainer.img_transform.test_transform,
)
eval_test_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir,
    download=True,
    train=False,
    transform=trainer.img_transform.test_transform,
)
eval_dataset = ConcatDataset([eval_train_dataset, eval_test_dataset])
trainer.setup()
trainer.train(dataset, eval_dataset)