import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn import functional as F
from torch.optim import SGD
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from idfd_utils import transform
from model import ResNet18, Normalize, NonParametricClassifier

from Incept.models import Trainer

class IDFDTrainer(Trainer):
    def __init__(self, config, strategy="earlystop", logger_backends=["json", "tensorboard"]):
        super().__init__(config, strategy, logger_backends)
        self.config = config
        self.img_transform = transform
    
    def setup(self):
        config = self.config
        resnet = ResNet18(low_dim=config.low_dim)
        self.model = resnet
        self.model.to(config.device)
        # self.model = DataParallel(self.model)
        self.norm = Normalize(2)
        self.optimizer = SGD(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.sgd_momentum,
            weight_decay=config.weight_decay,
        )
        self.scheduler = MultiStepLR(self.optimizer, milestones=config.milestones, gamma=config.gamma)
    
    def compute_loss(self, x, ff, y):
        L_id = F.cross_entropy(x, y)
        norm_ff = ff / (ff**2).sum(0, keepdim=True).sqrt()
        coef_mat = torch.mm(norm_ff.t(), norm_ff)
        coef_mat.div_(self.config.tau2)
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)
        L_fd = F.cross_entropy(coef_mat, a)
        return L_id, L_fd
    
    def train(self, dataset):
        config = self.config
        train_loader = DataLoader(dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=config.num_workers
        )
        self.npc = NonParametricClassifier(
            input_dim=config.low_dim,
            output_dim=len(dataset),
            tau=config.tau,
            momentum=config.momentum
        )
        self.npc.to(config.device)

        delta_label = -1
        acc, nmi, ari = -1, -1, -1
        predicted_previous = None
        ### Epoch Training
        for epoch in range(config.epochs):
            # step1, prepare model
            self.model.train()
            # step2, prepare data
            data_iterator = tqdm(train_loader, unit="batch")

            # step3, Batch Training
            ## step3.1, transport data
            for step, (inputs, _, indexes) in enumerate(data_iterator):
                self.optimizer.zero_grad()
                inputs = inputs.to(config.device, dtype=torch.float32, non_blocking=True)
                indexes = indexes.to(config.device, non_blocking=True)
                ## step3.2, calculate loss
                features = self.norm(self.model(inputs))
                outputs = self.npc(features, indexes)
                loss_id, loss_fd = self.compute_loss(outputs, features, indexes)
                loss = loss_id + loss_fd
                ## step3.3, update model
                loss.backward()
                self.optimizer.step()
                ## step3.4, update progress
                self.update_postfix(
                    data_iterator, epoch, config.epochs-1, step, len(data_iterator)-1,
                    loss=loss.item(), delta_label=delta_label, acc=acc, nmi=nmi, ari=ari
                )
                # exit(0)
            
            self.scheduler.step()
            # step4, evaluation
            if epoch % config.eval_epochs == 0:
                predicted, actual = self.predict(train_loader.dataset)
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
    
    def predict(self, dataset):
        config = self.config
        trainFeatures = self.npc.memory
        z = trainFeatures.cpu().numpy()
        kmeans = KMeans(n_clusters=config.cluster_num, n_init=20)
        features = kmeans.fit_predict(z)
        actual = np.array(dataset.targets)
        return features, actual
        # return super().train()

from Incept.utils import seed_everything
from Incept.configs import load_exp_config
seed_everything(42)
config = load_exp_config("IDFD", "CIFAR10")
trainer = IDFDTrainer(config)

import torchvision
class CIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index
    
train_dataset = CIFAR10(
    root=config.data_dir,
    download=True,
    train=True,
    transform=trainer.img_transform,
)

trainer.setup()
trainer.train(train_dataset)