import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

class CachedMNIST(Dataset):
    def __init__(self, train, cuda, testing_mode=False):
        img_transform = transforms.Compose([transforms.Lambda(self._transformation)])
        self.ds = MNIST("/data2/liangguanbao/opendeepclustering/datasets", download=True, train=train, transform=img_transform)
        self.cuda = cuda
        self.testing_mode = testing_mode
        self._cache = dict()

    @staticmethod
    def _transformation(img):
        np_array = np.array(img, dtype = np.uint8)
        tensor = torch.from_numpy(np_array).reshape(-1)
        tensor = tensor.float() * 0.02
        return tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        if index not in self._cache:
            self._cache[index] = list(self.ds[index])
            if self.cuda:
                self._cache[index][0] = self._cache[index][0].cuda(non_blocking=True)
                self._cache[index][1] = torch.tensor(
                    self._cache[index][1], dtype=torch.long
                ).cuda(non_blocking=True)
        return self._cache[index]

    def __len__(self) -> int:
        return len(self.ds)

