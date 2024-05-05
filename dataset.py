import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Normalize, Compose
from manipulate import addConceptDrift




def getCifar10(data_path: str = './data'):

    transformer = Compose([ToTensor(),Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10(data_path, train=True, download=True, transform=transformer)
    testset = CIFAR10(data_path, train=False, download=True, transform=transformer)

    # implementation of function that introduces iid and concept drift
    # which will basically take trainset , testset and add concpet drift and IID

    return trainset, testset


def dataset_preprocessing(partition_size: int, batch_size: int, val_ratio: float = 0.1):


    trainset, testset = getCifar10()

    partition_len = [len(trainset)// partition_size] * partition_size

    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(32))


    trainloaders =[]
    valloaders = []

    for item_ in trainsets:
        trainsets_item_len = len(item_)
        val_len = int(val_ratio*trainsets_item_len)
        actual_trainsets_item_len = trainsets_item_len - val_len

        for_train, for_val = random_split(item_, [actual_trainsets_item_len,val_len], 
                                          torch.Generator().manual_seed(32))

        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, 
                                       num_workers=0))

        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, 
                                     num_workers=0))

    testloaders = DataLoader(testset, batch_size=128) # need to change later

    
    return trainloaders, valloaders, testloaders