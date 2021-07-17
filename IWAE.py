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


class IWAE(nn.Module):

    def __init__(self, batch, width, height, latent_dim, k):
        super().__init__()

        '''initialize parameters of the model
        k is the num of sampling per image'''
        self.batch = batch
        self.k = k
        self.dim = width * height
        self.reduced_dim = 8 * (width - 4) * (height - 4)
        self.latent_dim = latent_dim

        # CNN to encode the images
        self.encConv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)
        self.encConv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)

        # to produce mean and variance of each image
        self.encMean = nn.Linear(in_features=self.reduced_dim, out_features=self.latent_dim)
        self.enclogVar = nn.Linear(in_features=self.reduced_dim, out_features=self.latent_dim)

        # decode the sampled tensors and reconstruct images
        self.decFC = nn.Linear(in_features=self.latent_dim, out_features=self.reduced_dim)
        self.decConv1 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3)
        self.decConv2 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3)

    def encoder(self, tensor):
        # return mean and logVar vector from encoding process
        tensor = F.relu(self.encConv1(tensor))
        tensor = F.relu(self.encConv2(tensor))
        self.reduced_dim_tuple = tuple(tensor.size())
        tensor = torch.flatten(tensor, start_dim=1)

        mean = self.encMean(tensor)
        logVar = self.enclogVar(tensor)

        return mean, logVar

    # this might be included in a nn.Sequential model, but it is not compatible with reshape function
    def decoder(self, tensor):
        tensor = F.relu(self.decFC(tensor))
        tensor = torch.reshape(tensor, shape=self.reduced_dim_tuple)
        tensor = F.relu(self.decConv1(tensor))
        # using sigmoid activation function at the end to produce images
        tensor = torch.sigmoid(self.decConv2(tensor))

        return tensor

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

    def log_q(self, mean, logVar, h):
        """
        log_q stands for logp(h) - logq(h|x)
        p(h) is a identity multivariate normal distribution,
        q(h|x) is a multivariate Gaussian with mean vector: mean, and covariance matrix: Var * I

        log_q_value is divided be self.dim (num of pixels in every image)
        to keep the scale between log_q and log_p
        """
        log_q_value = 0.5 * torch.sum((-h.pow(2) + ((h - mean).pow(2) / torch.exp(logVar)) + logVar), dim=1)
        log_q_value = log_q_value / self.dim

        return log_q_value

    def log_p(self, x, x_hat):
        """
        log_p stands for logp(x|h)

        According to the article,
        here the value of logp(x|h) should be the sum of BCE loss over each pixels,
        but rather it is taken as mean.
        This implementation together with clipping help to avoid gradient vanishing/explosion,
        since sum will give log_p_value to somewhere around -400,
        making it unreasonable to put in exp().

        note F.binary_cross_entropy has an in-built negative sign -
        """
        log_p_value = F.binary_cross_entropy(input=x_hat, target=x, reduction='none')
        log_p_value = torch.mean(log_p_value, dim=[2, 3])
        log_p_value = torch.clip_(log_p_value, min=-30, max=30)

        return log_p_value

    def w(self, x, x_hat, mean, logVar, h):
        """

        :param x: input image with size (batch, channel, width, height, k)
        :param x_hat: output image from the network with size (batch, channel, width, height, k)
        :param mean: mean vector calculated from the encoder with size (batch, latent_dim, k)
        :param logVar: log variance calculated from the encoder with size (batch, latent_dim, k)
        :param h: random vector sampled from random_sample with size (batch, latent_dim, k)
        :return: w = p(x,h)/q(h|x) with size (batch, k)
        """
        log_q_value = self.log_q(mean=mean, logVar=logVar, h=h)
        log_p_value = torch.squeeze(self.log_p(x=x, x_hat=x_hat))
        log_w = log_q_value - log_p_value
        w = torch.exp(log_w)

        return w

    def random_sample(self, mean, logVar):
        # random sampling from multivariate normal distribution
        std = torch.exp_(0.5 * logVar)
        epsilon = torch.randn_like(std)

        return mean + std * epsilon

    def loss_function(self, mean, logVar, img, out_list, h_list, k):
        # stack img, mean and logVar together in a new dimension to produce a tensor, same in each layer
        img = torch.where(img > 0.5, 1.0, 0.0)
        img_stack = torch.stack([img for i in range(k)], dim=4)
        mean_stack = torch.stack([mean for i in range(k)], dim=2)
        logVar_stack = torch.stack([logVar for i in range(k)], dim=2)

        # stack output images together in a new dimension to produce a tensor, different in each layer
        out_stack = torch.stack(out_list, dim=4)
        h_stack = torch.stack(h_list, dim=2)

        # retrieve w from each image which is sampled k times and sum the k results together
        sum_w = torch.sum(self.w(x=img_stack, x_hat=out_stack, mean=mean_stack, logVar=logVar_stack, h=h_stack), dim=1)

        # average over the k results and take log to produce the loss function we want to minimize
        loss = -torch.log(sum_w / k)
        # take the mean of loss over every image in the batch
        loss = torch.sum(loss, dim=0)

        return loss

    def IWAE_parameters(self):
        IWAE_parameters_dict = {
            'decFC': self.decFC,
            'decConv1': self.decConv1,
            'decConv2': self.decConv2,
            'reduced_dim_tuple': self.reduced_dim_tuple
        }

        return IWAE_parameters_dict
