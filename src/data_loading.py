import os
from torchvision import datasets, transforms


def load_dataset(path="../data"):
    """
    This function loads the Fashion MNIST dataset, checks if it is already downloaded in the specified path,
    and if not, downloads it. The dataset is normalized and stored in the given path.

    Parameters:
    - batch_size (int): The number of samples per batch to load. Default is 64.
    - path (str): The directory where the dataset should be stored. Default is "../data".

    """
    # Check if the dataset is already downloaded
    if any(os.scandir(path)):
        return

    # Normalization of dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download the dataset and store it in the path
    datasets.FashionMNIST(root=path, train=True, download=True, transform=transform)
    datasets.FashionMNIST(root=path, train=False, download=True, transform=transform)

    return
