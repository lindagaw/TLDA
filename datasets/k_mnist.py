"""Dataset setting and data loader for MNIST."""


import torch
from torchvision import datasets, transforms

import params


def get_kmnist(train):
    """Get MNIST dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=params.dataset_mean,
                                          std=params.dataset_std)])
    pre_process =  transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # dataset and data loader
    k_mnist_dataset = datasets.KMNIST(root=params.data_root,
                                   train=train,
                                   transform=pre_process,
                                   download=True)

    k_mnist_data_loader = torch.utils.data.DataLoader(
        dataset=k_mnist_dataset,
        batch_size=params.batch_size,
        shuffle=True)

    return k_mnist_data_loader
