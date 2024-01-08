from typing import Optional, Dict, Set, Tuple, List

import numpy as np
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset, DataLoader
from utils import GaussianBlur, get_multiclient_trainloader_list, Subset
from PIL import Image

from typing import Optional, Dict, Set, Tuple, List
import os
from torch.utils.data import DataLoader, Dataset

import os
import requests
import zipfile

STL10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
STL10_TRAIN_STD = (0.2471, 0.2435, 0.2616)
CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
TINYIMAGENET_TRAIN_MEAN = (0.5141, 0.5775, 0.3985)
TINYIMAGENET_TRAIN_STD = (0.2927, 0.2570, 0.1434)
SVHN_TRAIN_MEAN = (0.3522, 0.4004, 0.4463)
SVHN_TRAIN_STD = (0.1189, 0.1377, 0.1784)
IMAGENET_TRAIN_MEAN = (0.485, 0.456, 0.406)
IMAGENET_TRAIN_STD = (0.229, 0.224, 0.225)
DOMAINNET_TRAIN_MEAN = (0.485, 0.456, 0.406)
DOMAINNET_TRAIN_STD = (0.229, 0.224, 0.225)

def denormalize(x, dataset): # normalize a zero mean, std = 1 to range [0, 1]
    
    if dataset == "cifar10":
        std = [0.2023, 0.1994, 0.2010]
        mean = [0.4914, 0.4822, 0.4465]
    elif dataset == "cifar100":
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    elif dataset == "imagenet":
        std = [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
    elif dataset == "tinyimagenet":
        std = (0.2927, 0.2570, 0.1434)
        mean = (0.5141, 0.5775, 0.3985)   
    elif dataset == "svhn":
        std = (0.1189, 0.1377, 0.1784)
        mean = (0.3522, 0.4004, 0.4463)
    elif dataset == "stl10":
        std = (0.2471, 0.2435, 0.2616)
        mean = (0.4914, 0.4822, 0.4465)
    # 3, H, W, B
    tensor = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(range(tensor.size(0)), mean, std):
        tensor[t] = tensor[t].mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(tensor, 0, 1).permute(3, 0, 1, 2)


def get_cifar10(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        if data_proportion > 0.0:
            train_loader = get_cifar10_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        else:
            train_loader = None
        mem_loader = get_cifar10_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_cifar10_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        if data_proportion > 0.0:
            train_loader = get_cifar10_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        else:
            train_loader = None
        test_loader = get_cifar10_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_cifar100(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        per_client_train_loaders, client_to_labels = get_cifar100_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader, _ = get_cifar100_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader, per_client_test_loaders = get_cifar100_testloaders(128, num_workers, False, path_to_data, client_to_labels=client_to_labels)




        return per_client_train_loaders, mem_loader, test_loader, per_client_test_loaders, client_to_labels
    else:
        per_client_train_loaders, client_to_labels = get_cifar100_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader, per_client_test_loaders = get_cifar100_testloaders(128, num_workers, False, path_to_data, client_to_labels=client_to_labels)

        return per_client_train_loaders, test_loader, per_client_test_loaders, client_to_labels

def get_tinyimagenet(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./tiny-imagenet-200"):
    if pairloader_option != "None":
        train_loader = get_tinyimagenet_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_tinyimagenet_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_tinyimagenet_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_tinyimagenet_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_tinyimagenet_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_imagenet12(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    if pairloader_option != "None":
        train_loader = get_imagenet12_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_imagenet12_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_imagenet12_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_imagenet12_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_imagenet12_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_stl10(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        train_loader = get_stl10_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_stl10_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_stl10_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_stl10_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_stl10_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

def get_svhn(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    if pairloader_option != "None":
        train_loader = get_svhn_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_SVHN_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_SVHN_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_SVHN_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_SVHN_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader


def get_flowers102(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    if pairloader_option != "None":
        train_loader = get_flowers102_pairloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, pairloader_option, hetero, hetero_string, path_to_data)
        mem_loader = get_flowers102_trainloader(128, num_workers, False, path_to_data = path_to_data)
        test_loader = get_flowers102_testloader(128, num_workers, False, path_to_data)
        return train_loader, mem_loader, test_loader
    else:
        train_loader = get_flowers102_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string, path_to_data)
        test_loader = get_flowers102_testloader(128, num_workers, False, path_to_data)
        return train_loader, test_loader

# def get_svhn(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
#     train_loader = get_SVHN_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string)
#     test_loader = get_SVHN_testloader(128, num_workers, False)
#     return train_loader, test_loader

# def get_imagenet(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_proportion = 1.0, noniid_ratio =1.0, augmentation_option = False, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2"):
#     train_loader = get_imagenet_trainloader(batch_size, num_workers, shuffle, num_client, data_proportion, noniid_ratio, augmentation_option, hetero, hetero_string)
#     test_loader = get_imagenet_testloader(128, num_workers, False)
#     return train_loader, test_loader

def get_tinyimagenet_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./tiny-imagenet-200"):
    class tinyimagenetPair(torchvision.datasets.ImageFolder):
        """tinyimagenet Dataset.
        """
        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, _ = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                im_1 = self.transform(sample)
                im_2 = self.transform(sample)
            
            return im_1, im_2

    # tinyimagenet_training = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
    # tinyimagenet_testing = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)])
    # data prepare
    train_data = tinyimagenetPair(f'{path_to_data}/train', transform=train_transform)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    cifar100_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 200, hetero, hetero_string)
    
    return cifar100_training_loader

def get_cifar100_pairloader(
        batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"
) -> Tuple[List[torch.utils.data.DataLoader], Dict[int, Set[int]]]:
    class CIFAR100Pair(torchvision.datasets.CIFAR100):
        """CIFAR100 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)

            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            label = self.targets[index]
            return (im_1, im_2), label

    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)])
    # data prepare
    train_data = CIFAR100Pair(root=path_to_data, train=True, transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)

    return get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 100, hetero, hetero_string)
    

def get_cifar10_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class CIFAR10Pair(torchvision.datasets.CIFAR10):
        """CIFAR10 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    # data prepare
    
    train_data = CIFAR10Pair(root=path_to_data, train=True, transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    cifar10_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)
    
    return cifar10_training_loader

def get_tinyimagenet_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./tiny-imagenet-200"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
        ])

    if not os.path.isdir(f"{path_to_data}/train"):
        import subprocess
        subprocess.call("python prepare_tinyimagenet.py", shell=True)
    tinyimagenet_training = datasets.ImageFolder(f'{path_to_data}/train', transform=transform_train)
    

    indices = torch.randperm(len(tinyimagenet_training))[:int(len(tinyimagenet_training)* data_portion)]

    tinyimagenet_training = torch.utils.data.Subset(tinyimagenet_training, indices)

    tinyimagenet_training_loader = get_multiclient_trainloader_list(tinyimagenet_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 200, hetero, hetero_string)

    

    return tinyimagenet_training_loader

def get_tinyimagenet_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./tiny-imagenet-200"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_TRAIN_MEAN, TINYIMAGENET_TRAIN_STD)
    ])
    tinyimagenet_testing = datasets.ImageFolder(f'{path_to_data}/val', transform=transform_test)
    tinyimagenet_testing_loader = torch.utils.data.DataLoader(tinyimagenet_testing,  batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers)
    return tinyimagenet_testing_loader



def get_imagenet_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "../../imagenet"):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    train_dir = os.path.join(path_to_data, 'train')
    imagenet_training = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)

    indices = torch.randperm(len(imagenet_training))[:int(len(imagenet_training)* data_portion)]

    imagenet_training = torch.utils.data.Subset(imagenet_training, indices)

    imagenet_training_loader = get_multiclient_trainloader_list(imagenet_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 1000, hetero, hetero_string)

    return imagenet_training_loader


def get_imagenet_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "../../imagenet"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
    ])
    train_dir = os.path.join(path_to_data, 'val')
    imagenet_test = torchvision.datasets.ImageFolder(train_dir, transform=transform_test)
    imagenet_test_loader = DataLoader(imagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return imagenet_test_loader



def get_flowers102_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    class Flowers102Pair(torchvision.datasets.Flowers102):
        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            im_1, _  = super().__getitem__(index)
            im_2, _ = super().__getitem__(index)
            
            return im_1, im_2

    # tinyimagenet_training = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
    # tinyimagenet_testing = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    # data prepare
    train_data = Flowers102Pair(path_to_data, split="train", transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    imagenet_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 12, hetero, hetero_string)
    
    return imagenet_training_loader

def get_flowers102_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data/imagnet-12"):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    # train_dir = os.path.join(path_to_data, 'train')
    # imagenet_training = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    flowers_training = torchvision.datasets.Flowers102(path_to_data, split="train", transform=transform_train)

    n_data = len(flowers_training)

    indices = torch.randperm(n_data)[:int(n_data* data_portion)]

    imagenet_training = torch.utils.data.Subset(flowers_training, indices)

    loader = get_multiclient_trainloader_list(imagenet_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 12, hetero, hetero_string)

    return loader


def get_flowers102_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data/imagnet-12"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
    ])
    flowers_test = torchvision.datasets.Flowers102(path_to_data, split="test", transform=transform_test)

    imagenet_test_loader = DataLoader(flowers_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    
    return imagenet_test_loader


def get_imagenet12_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client=1, data_portion=1.0,
                              noniid_ratio=1.0, pairloader_option="None", hetero=False,
                              hetero_string="0.2_0.8|16|0.8_0.2", path_to_data="./data/imagnet-12"):
    class imagenet12Pair(torchvision.datasets.ImageFolder):
        """tinyimagenet Dataset.
        """

        def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            path, _ = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                im_1 = self.transform(sample)
                im_2 = self.transform(sample)

            return im_1, im_2

    # tinyimagenet_training = datasets.ImageFolder('tiny-imagenet-200/train', transform=transform_train)
    # tinyimagenet_testing = datasets.ImageFolder('tiny-imagenet-200/val', transform=transform_test)
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)])
    # data prepare
    train_data = imagenet12Pair(f'{path_to_data}/train', transform=train_transform)

    indices = torch.randperm(len(train_data))[:int(len(train_data) * data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)

    imagenet_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers,
                                                                batch_size, noniid_ratio, 12, hetero, hetero_string)

    return imagenet_training_loader


def get_imagenet12_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client=1, data_portion=1.0,
                               noniid_ratio=1.0, augmentation_option=False, hetero=False,
                               hetero_string="0.2_0.8|16|0.8_0.2", path_to_data="./data/imagnet-12"):
    """ return training dataloader
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
        ])
    train_dir = os.path.join(path_to_data, 'train')
    imagenet_training = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)

    indices = torch.randperm(len(imagenet_training))[:int(len(imagenet_training) * data_portion)]

    imagenet_training = torch.utils.data.Subset(imagenet_training, indices)

    imagenet_training_loader = get_multiclient_trainloader_list(imagenet_training, num_client, shuffle, num_workers,
                                                                batch_size, noniid_ratio, 12, hetero, hetero_string)

    return imagenet_training_loader


def get_imagenet12_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data="./data/imagnet-12"):
    """ return training dataloader
    Returns: imagenet_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_TRAIN_MEAN, IMAGENET_TRAIN_STD)
    ])
    train_dir = os.path.join(path_to_data, 'val')
    imagenet_test = torchvision.datasets.ImageFolder(train_dir, transform=transform_test)
    imagenet_test_loader = DataLoader(imagenet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return imagenet_test_loader

def get_stl10_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class STL10Pair(torchvision.datasets.STL10):
        """CIFAR10 Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(img)
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)])
    # data prepare
    train_data = STL10Pair(root=path_to_data, split = 'train+unlabeled', transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    stl10_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)
    
    return stl10_training_loader

def get_stl10_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(STL10_TRAIN_MEAN, STL10_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(STL10_TRAIN_MEAN, STL10_TRAIN_STD)
        ])
    #cifar00_training = CIFAR10Train(path, transform=transform_train)
    stl10_training = torchvision.datasets.STL10(root=path_to_data, split='train', download=True, transform=transform_train)

    indices = torch.randperm(len(stl10_training))[:int(len(stl10_training)* data_portion)]

    stl10_training = torch.utils.data.Subset(stl10_training, indices)

    stl10_training_loader = get_multiclient_trainloader_list(stl10_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)

    return stl10_training_loader

def get_stl10_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of stl10 test dataset
        std: std of stl10 test dataset
        path: path to stl10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: stl10_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(STL10_TRAIN_MEAN, STL10_TRAIN_STD)
    ])

    stl10_test = torchvision.datasets.STL10(root=path_to_data, split='test', download=True, transform=transform_test)
    stl10_test_loader = DataLoader(
        stl10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return stl10_test_loader

def get_cifar10_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 training dataset
        std: std of cifar10 training dataset
        path: path to cifar10 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
        ])
    #cifar00_training = CIFAR10Train(path, transform=transform_train)
    cifar10_training = torchvision.datasets.CIFAR10(root=path_to_data, train=True, download=True, transform=transform_train)

    indices = torch.randperm(len(cifar10_training))[:int(len(cifar10_training)* data_portion)]

    cifar10_training = torch.utils.data.Subset(cifar10_training, indices)

    cifar10_training_loader = get_multiclient_trainloader_list(cifar10_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)

    return cifar10_training_loader

def get_cifar10_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar10 test dataset
        std: std of cifar10 test dataset
        path: path to cifar10 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar10_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD)
    ])

    cifar10_test = torchvision.datasets.CIFAR10(root=path_to_data, train=False, download=True, transform=transform_test)
    cifar10_test_loader = DataLoader(
        cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar10_test_loader

def get_cifar100_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
        ])
    # print("num_client is", num_client)
    cifar100_training = torchvision.datasets.CIFAR100(root=path_to_data, train=True, download=True, transform=transform_train)

    indices = torch.randperm(len(cifar100_training))[:int(len(cifar100_training)* data_portion)]

    cifar100_training = torch.utils.data.Subset(cifar100_training, indices)

    cifar100_training_loader, client_to_labels = get_multiclient_trainloader_list(cifar100_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 100, hetero, hetero_string)

    return cifar100_training_loader, client_to_labels

def get_cifar100_testloaders(
        batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data", client_to_labels: Optional[Dict[int, Set[int]]] = None
):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root=path_to_data, train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)



    targets = np.array(cifar100_test.targets)

    per_client_test_loaders = {
        c_id:  DataLoader(
            Subset(cifar100_test, [i for (i,t) in enumerate(targets) if t in c_labels]),
            shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=False
        )
        for (c_id, c_labels)
        in client_to_labels.items()
    } if client_to_labels is not None else None


    return cifar100_test_loader, per_client_test_loaders

def get_SVHN_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, augmentation_option = False, hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of SVHN training dataset
        std: std of SVHN training dataset
        path: path to SVHN training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
        ])
    #cifar00_training = SVHNTrain(path, transform=transform_train)
    SVHN_training = torchvision.datasets.SVHN(root=path_to_data, split='train', download=True, transform=transform_train)

    indices = torch.randperm(len(SVHN_training))[:int(len(SVHN_training)* data_portion)]

    SVHN_training = torch.utils.data.Subset(SVHN_training, indices)

    SVHN_training_loader = get_multiclient_trainloader_list(SVHN_training, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)

    return SVHN_training_loader

def get_SVHN_testloader(batch_size=16, num_workers=2, shuffle=False, path_to_data = "./data"):
    """ return training dataloader
    Args:
        mean: mean of SVHN test dataset
        std: std of SVHN test dataset
        path: path to SVHN test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: SVHN_test_loader:torch dataloader object
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)
    ])
    SVHN_test = torchvision.datasets.SVHN(root=path_to_data, split='test', download=True, transform=transform_test)
    SVHN_test_loader = DataLoader(
        SVHN_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return SVHN_test_loader



def get_svhn_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client = 1, data_portion = 1.0, noniid_ratio = 1.0, pairloader_option = "None", hetero = False, hetero_string = "0.2_0.8|16|0.8_0.2", path_to_data = "./data"):
    class SVHNPair(torchvision.datasets.SVHN):
        """SVHN Dataset.
        """
        def __getitem__(self, index):
            img = self.data[index]
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                im_1 = self.transform(img)
                im_2 = self.transform(img)

            return im_1, im_2
    if pairloader_option == "mocov1":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)])
    elif pairloader_option == "mocov2":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(SVHN_TRAIN_MEAN, SVHN_TRAIN_STD)])
    # data prepare
    
    train_data = SVHNPair(root=path_to_data, split='train', transform=train_transform, download=True)
    
    indices = torch.randperm(len(train_data))[:int(len(train_data)* data_portion)]

    train_data = torch.utils.data.Subset(train_data, indices)
    
    svhn_training_loader = get_multiclient_trainloader_list(train_data, num_client, shuffle, num_workers, batch_size, noniid_ratio, 10, hetero, hetero_string)
    
    return svhn_training_loader


def get_multi_client_trainloader_list(
        multi_domain_train_data: List[Dataset], num_clients: int, shuffle: bool, num_workers: int, batch_size: int, client_labels: Set) -> Tuple[List[DataLoader], Dict[int, Set[int]]]:
    """
    Create a list of DataLoaders for multi-client training.

    :param multi_domain_train_data: List of datasets for each domain.
    :param num_clients: Number of clients.
    :param shuffle: Whether to shuffle the data.
    :param num_workers: Number of workers for loading data.
    :param batch_size: Batch size for each DataLoader.
    :param non_iid_ratio: Ratio for non-IID data split.
    :param client_labels: Set of choosen class labels
    :param heterogeneity: Whether to have heterogeneous data distribution.
    :param hetero_config: Configuration string for heterogeneity.

    :return: Tuple of DataLoader list and dictionary mapping clients to their labels.
    """

    if num_clients not in [1, len(multi_domain_train_data)]:
        raise ValueError("Number of clients must be either 1 or equal to the number of domains.")

    train_loader_list = []
    client_to_labels = {}

    if num_clients == 1:
        train_loader = DataLoader(multi_domain_train_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, persistent_workers=(num_workers > 0))
        train_loader_list.append(train_loader)
        client_to_labels[0] = client_labels
    else:
        for i, domain_data in enumerate(multi_domain_train_data):
            client_to_labels[i] = client_labels
            subset_train_loader = DataLoader(domain_data, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, persistent_workers=(num_workers > 0))
            train_loader_list.append(subset_train_loader)

    return train_loader_list, client_to_labels


def read_all_domainnet_data(dataset_path: str, split: str = "train") -> Dict[
    str, Tuple[List[str], List[int]]]:
    """
    Load data from all domains in the DomainNet dataset, filtering for specific classes based on file paths.

    :param dataset_path: Path to the DomainNet data folder.
    :param split: Data split type (default 'train').
    :return: Dictionary with domain names as keys and (data_paths, data_labels) as values.
    """

    label_names = ['bird', 'feather', 'headphones', 'ice_cream', 'teapot', 'tiger', 'whale', 'windmill', 'wine_glass',
                   'zebra']

    all_data = {}
    class_counts = {}
    domains = sorted(["clipart", "infograph", "painting", "quickdraw", "real", "sketch"])

    for domain in domains:
        data_paths, data_labels = [], []
        split_file = os.path.join(dataset_path, "splits", f"{domain}_{split}.txt")

        with open(split_file, "r") as file:
            for line in file:
                data_path, label = line.strip().split(' ')
                # Extracting class name from the data path
                class_name = data_path.split('/')[1]

                if label_names is None or class_name in label_names:
                    data_paths.append(os.path.join(dataset_path, data_path))
                    data_labels.append(int(label))

        all_data[domain] = (data_paths, data_labels)

        # Count labels in the domain
        for label in data_labels:
            if domain not in class_counts:
                class_counts[domain] = {}
            if label not in class_counts[domain]:
                class_counts[domain][label] = 0
            class_counts[domain][label] += 1

    # Display count of instances for each class in each domain
    for domain, counts in class_counts.items():
        print(f"Domain: {domain}")
        for label, count in counts.items():
            print(f"Class {label}: {count} instances")

    return all_data


# DomainNet Dataset Class
class DomainNet(Dataset):
    """
    Dataset class for DomainNet data.

    Attributes:
        data_paths (List[str]): Paths to the image files.
        data_labels (List[int]): Labels corresponding to the images.
        transforms (transforms.Compose): Transformations to be applied to the images.
    """
    def __init__(self, data_paths: List[str], data_labels: List[int], transforms: transforms.Compose):
        super().__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms

    def __getitem__(self, index: int):
        img = Image.open(self.data_paths[index]).convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.data_paths)

def get_domainnet_pairloader(batch_size=16, num_workers=2, shuffle=True, num_client=1, data_portion=1.0,
                            path_to_data="./data"):

    all_domain_data = read_all_domainnet_data(path_to_data, split="train")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DOMAINNET_TRAIN_MEAN, DOMAINNET_TRAIN_STD)
    ])

    class DomainNetPair(DomainNet):
        def __init__(self, data_paths, data_labels, transform=None):
            self.data_paths = data_paths
            self.data_labels = data_labels
            self.transform = transform

        def __getitem__(self, index):
            img_path = self.data_paths[index]
            label = self.data_labels[index]
            img = Image.open(img_path).convert("RGB")
            img_pair1 = self.transform(img)
            img_pair2 = self.transform(img)
            return (img_pair1, img_pair2), label

        def __len__(self):
            return len(self.data_paths)

    multi_domain_train_data = []
    client_labels = set()
    for i, domain in enumerate(all_domain_data):
        print(i, domain)
        data_paths, data_labels = all_domain_data[domain]
        client_labels.update(set(data_labels))
        domain_train_data = DomainNetPair(data_paths, data_labels, train_transform)
        indices = torch.randperm(len(domain_train_data))[:int(len(domain_train_data) * data_portion)]
        subset_domain_train_data = torch.utils.data.Subset(domain_train_data, indices)
        multi_domain_train_data.append(subset_domain_train_data)

    return get_multi_client_trainloader_list(multi_domain_train_data, num_client, shuffle, num_workers, batch_size, client_labels)


def get_domainnet_trainloader(batch_size=16, num_workers=2, shuffle=True, num_client=1, data_portion=1.0, augmentation_option=False, path_to_data="./data", split="train"):
    """

    Parameters:
        batch_size: Batch size for the DataLoader.
        num_workers: Number of worker threads for loading data.
        shuffle: Whether to shuffle the data.
        num_client: Number of clients (domains) in federated learning.
        data_portion: What portion of data to be used.
        augmentation_option: Whether to apply data augmentation.
        path_to_data: Path to the DomainNet data.

    Returns:
        A list of DataLoaders for each client and a dictionary assigning clients to classes.
    """

    if augmentation_option:
        transform_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(DOMAINNET_TRAIN_MEAN, DOMAINNET_TRAIN_STD)
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(DOMAINNET_TRAIN_MEAN, DOMAINNET_TRAIN_STD)
        ])

    all_domain_data = read_all_domainnet_data(path_to_data, split=split)

    if num_client == 1:
        client_labels = set()
        all_data_paths = []
        all_data_labels = []
        for i, domain in enumerate(all_domain_data):
            data_paths, data_labels = all_domain_data[domain]
            all_data_paths.extend(data_paths)
            all_data_labels.extend(data_labels)
            client_labels.update(set(data_labels))

        all_domain_train_data = DomainNet(all_data_paths, all_data_labels, transforms=transform_train)
        indices = torch.randperm(len(all_domain_train_data))[:int(len(all_domain_train_data) * data_portion)]
        multi_domain_train_data = torch.utils.data.Subset(all_domain_train_data, indices)

    else:
        client_labels = set()
        multi_domain_train_data = []
        for domain in all_domain_data:
            data_paths, data_labels = all_domain_data[domain]
            client_labels.update(set(data_labels))
            domain_dataset = DomainNet(data_paths, data_labels, transforms=transform_train)
            indices = torch.randperm(len(domain_dataset))[:int(len(domain_dataset) * data_portion)]
            subset_domain_data = torch.utils.data.Subset(domain_dataset, indices)
            multi_domain_train_data.append(subset_domain_data)


    domainnet_training_loader, client_to_labels = get_multi_client_trainloader_list(multi_domain_train_data, num_client, shuffle, num_workers, batch_size, client_labels)
    return domainnet_training_loader, client_to_labels


def generate_domain_net_data(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    root = dir_path + "rawdata"
    os.makedirs(root, exist_ok=True)

    domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
    urls = [
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
        'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
    ]
    http_head = 'http://csr.bu.edu/ftp/visda/2019/multi-source/'
    # Get DomainNet data

    for d, u in zip(domains, urls):
        zip_path = os.path.join(root, f"{d}.zip")
        extract_path = os.path.join(root, d)

        if not os.path.exists(extract_path):
            if not os.path.exists(zip_path):
                r = requests.get(u, stream=True)
                with open(zip_path, 'wb') as f:
                    f.write(r.content)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(root)
                print(f"File {d}.zip unziped succesfully.")
            except Exception as e:
                print(f"Error when unziping {d}.zip: {e}")

        for suffix in ['train', 'test']:
            txt_url = f"{http_head}domainnet/txt/{d}_{suffix}.txt"
            txt_path = os.path.join(root, 'splits', f"{d}_{suffix}.txt")
            if not os.path.exists(txt_path):
                r = requests.get(txt_url, stream=True)
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                with open(txt_path, 'wb') as f:
                    f.write(r.content)


def get_domainnet(batch_size=16, num_workers=2, shuffle=True, num_client=1, data_proportion=1.0, noniid_ratio=1.0, augmentation_option=False, pairloader_option="None", hetero=False, hetero_string="0.2_0.8|16|0.8_0.2", path_to_data="./data/DomainNet/rawdata"):
    generate_domain_net_data("data/DomainNet/")

    if pairloader_option != "None":
        per_client_train_loaders, client_to_labels = get_domainnet_pairloader(batch_size, num_workers,
                                                                              shuffle, num_client,
                                                                              data_proportion,
                                                                              path_to_data)
        # todo change batch size here
        mem_loader, _ = get_domainnet_trainloader(64, num_workers, False, 1,
                                                  data_proportion, augmentation_option, path_to_data)

        # todo change batch size here
        test_loader, per_client_test_loaders = get_domainnet_testloader(64, num_workers, False,
                                                                        path_to_data,
                                                                        client_to_labels)

        return per_client_train_loaders, mem_loader, test_loader, per_client_test_loaders, client_to_labels
    else:

        per_client_train_loaders, client_to_labels = get_domainnet_trainloader(batch_size, num_workers, shuffle, num_client,
                                                  data_proportion, augmentation_option, path_to_data)

        # todo change batch size here (if needed)
        test_loader, per_client_test_loaders = get_domainnet_testloader(64, num_workers, False,
                                                                        path_to_data,
                                                                        client_to_labels)

        return per_client_train_loaders, test_loader, per_client_test_loaders, client_to_labels

def get_domainnet_testloader(
        batch_size=16, num_workers=2, shuffle=False, path_to_data="./data", client_to_labels: Optional[Dict[int, Set[int]]] = None, split="test"
):
    """
    Returns a DataLoader for the DomainNet test data set.

    Parameters:
        batch_size: Batch size for the DataLoader.
        num_workers: Number of threads used for loading data.
        shuffle: Whether to shuffle the data.
        path_to_data: Path to DomainNet data.
        client_to_labels: Dictionary assigning clients to classes.
        split: Type of data split (default 'test').

    Returns:
        domainnet_test_loader: The main test DataLoader.
        per_client_test_loaders: Dictionary of DataLoaders for each client.
    """
    transform_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(DOMAINNET_TRAIN_MEAN, DOMAINNET_TRAIN_STD)
    ])

    all_domain_data = read_all_domainnet_data(path_to_data, split=split)

    assert len(all_domain_data) == len(client_to_labels), (len(all_domain_data), len(client_to_labels))
    domain_keys = sorted(all_domain_data.keys())

    all_test_data_paths, all_test_data_labels = [], []

    per_client_test_loaders = dict()
    for c_id, domain in enumerate(domain_keys):
        paths, labels = all_domain_data[domain]
        assert all([domain in p for p in paths])  # path should contain the name of the current domain
        all_test_data_paths.extend(paths)
        all_test_data_labels.extend(labels)

        indices = [i for i, label in enumerate(labels) if label in client_to_labels[c_id]]
        domain_test_set =  DomainNet(paths, labels, transforms=transform_test)
        subset = Subset(domain_test_set, indices)
        per_client_test_loaders[c_id] = DataLoader(
            subset,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            drop_last=False
        )

    domainnet_all_test = DomainNet(all_test_data_paths, all_test_data_labels, transforms=transform_test)
    all_clients_labels = set().union(*client_to_labels.values())
    all_indices = [i for i, label in enumerate(all_test_data_labels) if label in all_clients_labels]
    all_subset = Subset(domainnet_all_test, all_indices)

    domainnet_test_loader = DataLoader(all_subset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return domainnet_test_loader, per_client_test_loaders

