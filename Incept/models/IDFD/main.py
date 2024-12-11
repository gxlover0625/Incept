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
        self.norm = Normalize(2)
    
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
        print(self.npc)
        # return super().train()
    
    def predict(self):
        return super().predict()

from Incept.utils import seed_everything
from Incept.configs import load_exp_config
seed_everything(42)
config = load_exp_config("IDFD", "CIFAR10")
trainer = IDFDTrainer(config)

import torchvision
train_dataset = torchvision.datasets.CIFAR10(
    root=config.data_dir,
    download=True,
    train=True,
    transform=trainer.img_transform,
)

trainer.setup()
trainer.train(train_dataset)