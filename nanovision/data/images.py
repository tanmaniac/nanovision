"""Thin torchvision wrappers for MNIST / CIFAR-10. Provided boilerplate."""

from torchvision import datasets, transforms


def mnist(root: str = "./data", train: bool = True, download: bool = True):
    tf = transforms.Compose([transforms.ToTensor()])
    return datasets.MNIST(root=root, train=train, download=download, transform=tf)


def cifar10(root: str = "./data", train: bool = True, download: bool = True):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    return datasets.CIFAR10(root=root, train=train, download=download, transform=tf)
