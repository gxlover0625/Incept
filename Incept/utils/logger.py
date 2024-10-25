from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

class BasicLogger:
    def __init__(self, backends = ["json", "tensorboard"], log_dir = None):
        if "json" in backends:
            self.json_data = defaultdict(dict)
        else:
            self.json_data = None
        if "tensorboard" in backends:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
    
    def log(self, *args, **kwargs):
        epoch = kwargs["epoch"]
        for key, value in kwargs.items():
            if key != "epoch":
                if self.json_data is not None:
                    self.json_data[epoch][key] = value
                
                if self.writer is not None:
                    self.writer.add_scalar(key, value, epoch)
    
    def store_best(self):
        pass