from torch.utils.data import Dataset
from torchvision.datasets import MNIST

class CommonDataset(Dataset):
    def __init__(self, dataset_name, data_dir, split, img_transform = None, target_transform = None, device = "cuda"):
        if dataset_name == "mnist":
            assert split in [True, False]
            self.ds = MNIST(data_dir, download=True, train=split, transform=img_transform, target_transform=target_transform)
        self.device = device
        self._cache = dict()
       
    def __getitem__(self, index):
        if index not in self._cache:
            img, target = self.ds[index]
            img = img.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self._cache[index] = (img, target)
        return self._cache[index]

    def __len__(self):
        return len(self.ds)