from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10

class CommonDataset(Dataset):
    def __init__(self, dataset_name, data_dir, split, img_transform = None, target_transform = None):
        if dataset_name == "MNIST":
            assert split in [True, False]
            self.ds = MNIST(data_dir, download=True, train=split, transform=img_transform, target_transform=target_transform)
        elif dataset_name == "CIFAR10":
            assert split in [True, False]
            self.ds = CIFAR10(data_dir, download=True, train=split, transform=img_transform, target_transform=target_transform)
    def __getitem__(self, index):
        return self.ds[index]

    def __len__(self):
        return len(self.ds)