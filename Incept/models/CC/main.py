import torchvision
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.adam import Adam
import os
import json
from cc_utils import Transforms
from model import get_resnet, Network, InstanceLoss, ClusterLoss
from Incept.evaluation import Evaluator
from Incept.utils import EarlyStopping, BasicLogger
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict

from Incept.configs import load_exp_config
class CCTrainer:
    def __init__(self, config, strategy="earlystop", logger_backends=["json", "tensorboard"]):
        self.config = config

        # dataset processing
        self.img_transform = Transforms(size=config.image_size, s=0.5)

        # evaluation component
        self.evaluator = Evaluator(metrics=["acc", "nmi", "ari"])
        self.best_acc, self.best_nmi, self.best_ari = -1, -1, -1
        self.best_epoch = None

        # stop strategy
        if strategy == "earlystop":
            self.early_stopper = EarlyStopping()
        else:
            self.early_stopper = None
        
        # log component
        self.json_data = defaultdict(dict) if "json" in logger_backends else None
        self.writer = SummaryWriter(log_dir=config.output_dir) if "tensorboard" in logger_backends else None
        
    def setup(self):
        config = self.config
        resnet = get_resnet(config.resnet)
        self.model = Network(resnet, config.feature_dim, config.cluster_num)
        self.model.cuda()
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

    def update_postfix(self, iterator, epoch, total_epochs, step, total_steps, *args, **kwargs):
        update_info = {
            "epoch": f"{epoch}/{total_epochs}",
            "step": f"{step}/{total_steps}",
        }
        for key, value in kwargs.items():
            if not value:
                update_info[key] = f"{value:.4f}"
            update_info[key] = value
        
        iterator.set_postfix(update_info)
    
    def log_update(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        for key, value in kwargs.items():
            if key != "epoch":
                if self.json_data is not None:
                    self.json_data[epoch][key] = value
                
                if self.writer is not None:
                    self.writer.add_scalar(key, value, epoch)
        
        acc, nmi, ari = kwargs["acc"], kwargs["nmi"], kwargs["ari"]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_nmi = nmi
            self.best_ari = ari
            self.best_epoch = epoch
            if not os.path.exists(config.output_dir):
                os.makedirs(config.output_dir, exist_ok=True)
            # 保存模型
            torch.save(self.model.state_dict(), os.path.join(config.output_dir, "best_model.pth"))
    
    def summary(self):
        summary_data = {
            "best_epoch": self.best_epoch,
            "best_acc": self.best_acc,
            "best_nmi": self.best_nmi,
            "best_ari": self.best_ari
        }
        save_path = os.path.join(config.output_dir, "summary.json")
        with open(save_path, "w") as f:
            json.dump(summary_data, f, indent=4, ensure_ascii=False)
        
        save_path = os.path.join(config.output_dir, "log.json")
        if self.json_data is not None:
            with open(save_path, "w") as f:
                json.dump(self.json_data, f, indent=4, ensure_ascii=False)
            
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

                if step % 50 == 0:
                    print(
                        f"Step [{step}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
            
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


from Incept.utils import seed_everything
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

# from torch.utils.data import Subset
# dataset = Subset(dataset, indices=np.random.choice(len(dataset), 512))
# eval_dataset = Subset(eval_dataset, indices=np.random.choice(len(eval_dataset), 512))
trainer.train(dataset, eval_dataset)
# trainer.predict(eval_dataset)
# data_loader = DataLoader(
#     dataset,
#     batch_size=config.batch_size,
#     shuffle=True,
#     drop_last=True,
#     num_workers=config.num_workers,
# )
# res = get_resnet(config.resnet)
# model = Network(res, config.feature_dim, config.cluster_num)
# model.cuda()
# optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
# criterion_instance = InstanceLoss(config.batch_size, config.instance_temperature, config.device).to(config.device)
# print(criterion_instance)
# criterion_cluster = ClusterLoss(config.cluster_num, config.cluster_temperature, config.device).to(config.device)

# def train():
#     loss_epoch = 0
#     for step, ((x_i, x_j), _) in enumerate(tqdm(data_loader)):
#         optimizer.zero_grad()
#         x_i = x_i.to('cuda')
#         x_j = x_j.to('cuda')
#         z_i, z_j, c_i, c_j = model(x_i, x_j)
#         loss_instance = criterion_instance(z_i, z_j)
#         loss_cluster = criterion_cluster(c_i, c_j)
#         loss = loss_instance + loss_cluster
#         loss.backward()
#         optimizer.step()
#         if step % 50 == 0:
#             print(
#                 f"Step [{step}/{len(data_loader)}]\t loss_instance: {loss_instance.item()}\t loss_cluster: {loss_cluster.item()}")
#         loss_epoch += loss.item()
#     return loss_epoch


# for epoch in range(config.start_epoch, config.train_epochs):
#     lr = optimizer.param_groups[0]["lr"]
#     # break
#     loss_epoch = train()
#     break
#     # if epoch % 10 == 0:
#     #     save_model(args, model, optimizer, epoch)
#     # print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")