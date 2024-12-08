import torch
import torch.nn as nn
import numpy as np


def create_layer(size_in, size_out, dropout=False, normalize=True):
    """
       Initializes a list of layers for a neural network.

       Parameters:
       - size_in (int): Input features.
       - size_out (int): Output features.
       - dropout (bool, optional): Add a dropout layer.
       - normalize (bool, optional): Add a batch normalization layer.

       Returns:
       - list: A list of layers including a linear layer, optional dropout and batch normalization layers, and a LeakyReLU activation function.
    """
    layers = [nn.Linear(size_in, size_out)]
    if dropout:
        layers.append(nn.Dropout(0.4))
    if normalize:
        layers.append(nn.BatchNorm1d(size_out))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

class Generator(nn.Module):
    """
       A neural network generator class for generating images.

       Parameters:
       - classes (int): The number of classes for the label embedding.
       - img_size (int): The size of the generated images.
       - latent_dim (int): The dimensionality of the latent space.

    """
    def __init__(self, latent_dim, classes=10, img_size=28,):
        super(Generator, self).__init__()
        self.img_shape = (1, img_size, img_size)
        self.label_embedding = nn.Embedding(classes, classes)

        self.model = nn.Sequential(
            *create_layer(latent_dim + classes, 128, False),
            *create_layer(128, 256),
            *create_layer(256, 512),
            *create_layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh())

    def forward(self, noise, labels):
        z = torch.cat((self.label_embedding(labels), noise), -1)
        x = self.model(z)
        x = x.view(x.size(0), *self.img_shape)
        return x

class Discriminator(nn.Module):
    """
       A neural network discriminator class for distinguishing between real and generated images.

       Parameters:
       - classes (int): The number of classes for the label embedding.
       - img_size (int): The size of the generated images.
       - latent_dim (int): The dimensionality of the latent space.

    """
    def __init__(self, classes=10, img_size=28):
        super(Discriminator, self).__init__()
        self.img_shape = (1, img_size, img_size)
        self.label_embedding = nn.Embedding(classes, classes)

        self.model = nn.Sequential(
            *create_layer(classes + int(np.prod(self.img_shape)), 1024, False, True),
            *create_layer(1024, 512, True, True),
            *create_layer(512, 256, True, True),
            *create_layer(256, 128, False, False),
            *create_layer(128, 1, False, False),
            nn.Sigmoid())

    def forward(self, image, labels):
        x = torch.cat((image.view(image.size(0), -1), self.label_embedding(labels)), -1)
        return self.model(x)

    @staticmethod
    def loss(output, label):
        criterion = torch.nn.BCELoss()
        return criterion(output, label)
