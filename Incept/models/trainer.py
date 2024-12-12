import json
import os
import torch

from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from Incept.evaluation import Evaluator
from Incept.utils import EarlyStopping

class Trainer:
    def __init__(self, config, strategy="earlystop", logger_backends=["json", "tensorboard"]):
        self.config = config

        # evaluation component
        self.evaluator = Evaluator(metrics=["acc", "nmi", "ari"])
        self.best_acc, self.best_nmi, self.best_ari = -1, -1, -1
        self.best_epoch = None

        # stop component
        if strategy == "earlystop":
            self.early_stopper = EarlyStopping()
        else:
            self.early_stopper = None

        # log component
        self.json_data = defaultdict(dict) if "json" in logger_backends else None
        self.writer = SummaryWriter(log_dir=config.output_dir) if "tensorboard" in logger_backends else None
    
    def setup(self):
        pass
    
    def compute_loss(self):
        pass

    def update_postfix(self, iterator, epoch, total_epochs, step, total_steps, *args, **kwargs):
        update_info = {
            "epoch": f"{epoch}/{total_epochs}",
            "step": f"{step}/{total_steps}",
        }
        for key, value in kwargs.items():
            if isinstance(value, float):
                value = f"{value:.8f}"
            update_info[key] = value
        iterator.set_postfix(update_info)
    
    def log_update(self, *args, **kwargs):
        config = self.config
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
        config = self.config
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
    
    def train(self):
        pass

    def predict(self):
        pass