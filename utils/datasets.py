import PIL
import torch
import argparse
import numpy as np
import os
import copy
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder, DatasetFolder
import torch.utils.data
import re
import warnings
from PIL import Image
from PIL import ImageFile
import random
import torch.nn.functional as F
from torch.utils.data import Dataset
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


class LoaderGenerator():
    def __init__(self, root, val_batch_size=1, num_workers=0, kwargs={}):
        self.root = root
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self._train_set = None
        self._val_set = None
        self._calib_set = None
        self.train_transform = None
        self.val_transform = None
        self.train_loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': True,
            'drop_last': False,
        }
        self.val_loader_kwargs = {
            'num_workers': self.num_workers,
            'pin_memory': False,
            'drop_last': False,
        }
        self.load()
    
    @property
    def train_set(self):
        pass

    @property
    def val_set(self):
        pass
    
    def load(self):
        pass
    
    def val_loader(self):
        assert self.val_set is not None
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.val_batch_size, shuffle=False, **self.val_loader_kwargs)

    def calib_loader(self, num=1024, batch_size=32, seed=3, in_memory=True):
        np.random.seed(seed)
        inds = np.random.permutation(len(self.train_set))[:num]
        if in_memory:
            preloaded_data = [self.train_set[i] for i in inds]
            self._calib_set = CacheDataset(preloaded_data)
        else:
            self._calib_set = torch.utils.data.Subset(copy.deepcopy(self.train_set),inds)
            self._calib_set.dataset.transform=self.val_transform
        return torch.utils.data.DataLoader(self._calib_set, batch_size=batch_size, shuffle=False, **self.train_loader_kwargs)


class ImageNetLoaderGenerator(LoaderGenerator):
    def load(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    @property
    def train_set(self):
        if self._train_set is None:
            self._train_set = ImageFolder(os.path.join(self.root,'train'), self.train_transform)
        return self._train_set

    @property
    def val_set(self):
        if self._val_set is None:
            self._val_set = ImageFolder(os.path.join(self.root,'val'), self.val_transform)
        return self._val_set

class CacheDataset(Dataset):
    def __init__(self, datas) -> None:
        super().__init__()
        self.datas = datas
        
    def __getitem__(self,idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


class ViTImageNetLoaderGenerator(ImageNetLoaderGenerator):
    """
    DataLoader for Vision Transformer. 
    To comply with timm's framework, we use the model's corresponding transform.
    """
    def __init__(self, root, val_batch_size, num_workers, kwargs={}):
        super().__init__(root, val_batch_size=val_batch_size, num_workers=num_workers, kwargs=kwargs)

    def load(self):
        model = self.kwargs.get("model", None)
        assert model != None, f"No model in ViTImageNetLoaderGenerator!"
        config = resolve_data_config(model.default_cfg, model=model)
        self.train_transform = create_transform(**config, is_training=True)
        self.val_transform = create_transform(**config)


class CifarLoaderGenerator(LoaderGenerator):
    """
    DataLoader for CIFAR-10/100 datasets.
    Compatible with FIMA-Q quantization pipeline.
    """
    def __init__(self, root, which='cifar100', val_batch_size=64, num_workers=2, kwargs={}, train_augment=True):
        which = which.lower()
        if which == 'cifar100':
            from torchvision.datasets import CIFAR100
            self.ds_cls, self.num_classes = CIFAR100, 100
        elif which == 'cifar10':
            from torchvision.datasets import CIFAR10
            self.ds_cls, self.num_classes = CIFAR10, 10
        else:
            raise ValueError(f'which must be cifar10|cifar100, got {which!r}')
        
        self.which = which
        self.train_augment = train_augment
        super().__init__(root, val_batch_size=val_batch_size, num_workers=num_workers, kwargs=kwargs)
    
    def load(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        bicubic = transforms.InterpolationMode.BICUBIC
        
        self.val_transform = transforms.Compose([
            transforms.Resize(224, interpolation=bicubic),
            transforms.ToTensor(),
            normalize,
        ])
        
        if self.train_augment:
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(224, interpolation=bicubic),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            self.train_transform = self.val_transform
        
        print(f'Loading {self.which} into {self.root} ...')
        self._train_set = self.ds_cls(root=self.root, train=True, download=True,
                                       transform=self.train_transform)
        self._val_set = self.ds_cls(root=self.root, train=False, download=True,
                                     transform=self.val_transform)
        print(f'{self.which}: train={len(self._train_set)}, val={len(self._val_set)}, '
              f'classes={self.num_classes}')
    
    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set
