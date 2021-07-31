'''
importance weighted autoencoder
'''

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

torch.manual_seed(0)


class autoencoder(nn.Module):

    def __init__(self, channel, width, height, hyperparameters, mode):
        super().__init__()

        '''initialize parameters of the model
        k is the num of sampling per image'''
        self.reduced_dim = 8 * (width - 4) * (height - 4)
        try:
            self.k = hyperparameters['k_' + mode]
            self.latent_dim = hyperparameters['latent_dim_' + mode]
        except NameError:
            print('wrong mode')

        # CNN to encode the images
        self.encConv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.encConv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)

        # to produce mean and variance of each image
        self.encMean = nn.Linear(in_features=self.reduced_dim, out_features=self.latent_dim)
        self.enclogVar = nn.Linear(in_features=self.reduced_dim, out_features=self.latent_dim)

        # pack decoder as a sequential model to facilitate storage and reuse
        self.decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.reduced_dim),
            nn.ReLU(),
            Reshape(width=width, height=height),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3),
            nn.Sigmoid()
        )

    def encoder(self, tensor):
        # return mean and logVar vector from encoding process
        tensor = F.relu(self.encConv1(tensor))
        tensor = F.relu(self.encConv2(tensor))
        tensor = torch.flatten(tensor, start_dim=1)

        mean = self.encMean(tensor)
        logVar = self.enclogVar(tensor)

        return mean, logVar

    def forward(self, tensor):
        """

        :return: random vector list (from k different times of sampling)
                 output images list from the decoder (from k different times of sampling)
                 mean vector derived from the encoder
                 logVar vector derived from the encoder
        """
        h_list = []
        out_list = []
        mean, logVar = self.encoder(tensor)

        # for each image in the batch, sampling from a same distribution k times
        for i in range(self.k):
            h = self.random_sample(mean, logVar)
            h_list.append(h)
            out = self.decoder(h)
            out_list.append(out)

        return h_list, out_list, mean, logVar

    def logp_x_h(self, x, x_hat):
        # logp_x_h stands for logp(x|h)
        logp_x_h_value = F.binary_cross_entropy(input=x_hat, target=x, reduction='none')
        logp_x_h_value = -torch.sum(logp_x_h_value, dim=[2, 3])

        return logp_x_h_value

    def logp_h(self, h):
        # logp_h stands for logp(h)
        logp_h_value = - 0.5 * torch.sum(h.pow(2) + np.log(2 * np.pi), dim=1)

        return logp_h_value

    def logq_h_x(self, mean, logVar, h):
        # logq_h_x stands for logq(h|x)
        logq_h_x_value = - 0.5 * torch.sum(((h - mean).pow(2) / torch.exp(logVar)) + logVar + np.log(2 * np.pi), dim=1)

        return logq_h_x_value

    def random_sample(self, mean, logVar):
        # random sampling from multivariate normal distribution
        std = torch.exp_(0.5 * logVar)
        epsilon = torch.randn_like(std)

        return mean + std * epsilon

    def loss_function(self, mean, logVar, img, out_list, h_list):
        # stack img, mean and logVar together in a new dimension to produce a tensor, same in each layer
        img = torch.where(img > 0.5, 1.0, 0.0)
        img_stack = torch.stack([img for i in range(self.k)], dim=4)
        mean_stack = torch.stack([mean for i in range(self.k)], dim=2)
        logVar_stack = torch.stack([logVar for i in range(self.k)], dim=2)

        # stack output images together in a new dimension to produce a tensor, different in each layer
        out_stack = torch.stack(out_list, dim=4)
        h_stack = torch.stack(h_list, dim=2)

        # retrieve the log probabilities
        logq_h_x_value = self.logq_h_x(mean=mean_stack, logVar=logVar_stack, h=h_stack)
        logp_h_value = self.logp_h(h=h_stack)
        logp_x_h_value = self.logp_x_h(x=img_stack, x_hat=out_stack).view(-1, self.k)
        log_w = logp_x_h_value + logp_h_value - logq_h_x_value

        # to produce the loss of the two models
        loss = - (torch.logsumexp(log_w, dim=1) - np.log(self.k))

        # take the mean of loss over every image in the batch
        loss = torch.sum(loss, dim=0)

        return loss

    def decoder_model(self):
        # return the decoder model of this network for sampling
        return self.decoder


class Reshape(nn.Module):
    def __init__(self, width, height, channel=8):
        super().__init__()
        self.width = width
        self.height = height
        self.channel = channel

    def forward(self, tensor):
        return tensor.view(-1, self.channel, self.width - 4, self.height - 4)
