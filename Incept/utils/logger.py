import json
import os
import torch

from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

class BasicLogger:
    def __init__(self, backends = ["json", "tensorboard"], log_dir = None):
        self.log_dir = log_dir

        # initialize logger
        if "json" in backends:
            self.json_data = defaultdict(dict)
        else:
            self.json_data = None

        if "tensorboard" in backends:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
          
        self.best_acc, self.best_nmi, self.best_ari = -1, -1, -1
        self.best_epoch = -1
        self.best_model = None

    def log(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        for key, value in kwargs.items():
            if key != "epoch":
                if self.json_data is not None:
                    self.json_data[epoch][key] = value
                
                if self.writer is not None:
                    self.writer.add_scalar(key, value, epoch)
    
    def update(self, *args, **kwargs):
        acc, nmi, ari = kwargs["acc"], kwargs["nmi"], kwargs["ari"]
        epoch = kwargs["epoch"]
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_nmi = nmi
            self.best_ari = ari
            self.best_epoch = epoch
            self.best_model = kwargs["model"]
    def summary(self):
        summary_data = {
            "best_epoch": self.best_epoch,
            "best_acc": self.best_acc,
            "best_nmi": self.best_nmi,
            "best_ari": self.best_ari
        }
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        with open(os.path.join(self.log_dir, "summary.json"), "w") as f:
            json.dump(summary_data, f, indent=4)

        if self.json_data is not None:
            with open(os.path.join(self.log_dir, "log.json"), "w") as f:
                json.dump(self.json_data, f, indent=4)
        
        torch.save(self.best_model.state_dict(), os.path.join(self.log_dir, "best_model.pth"))