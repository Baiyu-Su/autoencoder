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

        self.batch = batch
        self.k = k
        self.dim = width * height
        self.reduced_dim = 8 * (width - 6) * (height - 6)
        self.reduced_dim_tuple = (batch, 8, width - 6, height - 6)
        self.latent_dim = latent_dim

        self.encConv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5)
        self.encConv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3)

        self.encMean = nn.Linear(in_features=self.reduced_dim, out_features=self.latent_dim)
        self.enclogVar = nn.Linear(in_features=self.reduced_dim, out_features=self.latent_dim)

        self.decFC = nn.Linear(in_features=self.latent_dim, out_features=self.reduced_dim)
        self.decConv1 = nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3)
        self.decConv2 = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=5)

    def encoder(self, tensor):
        tensor = F.relu(self.encConv1(tensor))
        tensor = F.relu(self.encConv2(tensor))
        tensor = torch.flatten(tensor, start_dim=1)

        mean = self.encMean(tensor)
        logVar = self.enclogVar(tensor)

        return mean, logVar

    def decoder(self, tensor):
        tensor = F.relu(self.decFC(tensor))
        tensor = torch.reshape(tensor, shape=self.reduced_dim_tuple)
        tensor = F.relu(self.decConv1(tensor))
        tensor = torch.sigmoid(self.decConv2(tensor))

        return tensor

    def w(self, x, x_hat, mean, logVar):
        log_q = self.log_q(mean, logVar)
        log_p = torch.squeeze(self.log_p(x=x, x_hat=x_hat))

        log_w = log_p - log_q
        #print(log_p)
        #print(log_q)
        w = torch.exp(log_w)

        return w

    def forward(self, tensor):
        out_list = []
        mean, logVar = self.encoder(tensor)

        for i in range(self.k):
            z = self.random_sample(mean, logVar)
            out = self.decoder(z)
            out_list.append(out)

        return out_list, mean, logVar

    def loss_function(self, mean, logVar, img, out_list):
        img_list = [img for i in range(self.k)]
        img_stack = torch.stack(img_list, dim=4)
        out_stack = torch.stack(out_list, dim=4)
        mean_list = [mean for i in range(self.k)]
        mean_stack = torch.stack(mean_list, dim=2)
        logVar_list = [logVar for i in range(self.k)]
        logVar_stack = torch.stack(logVar_list, dim=2)
        '''for out in out_list:
            print(out.size())
            w = self.w(x=img, x_hat=out, mean=mean, logVar=logVar)
            sum += w'''
        w = self.w(x=img_stack, x_hat=out_stack, mean=mean_stack, logVar=logVar_stack)
        sum = torch.sum(w, dim=1)

        loss = torch.log(sum/self.k)
        loss = torch.sum(loss, dim=0)

        return loss

    def log_q(self, mean, logVar):
        log_q = 0.5 * torch.sum((logVar - mean.pow(2) - torch.exp_(logVar) + 1), dim=1)

        return log_q

    def log_p(self, x, x_hat):
        log_p = -F.binary_cross_entropy(input=x_hat, target=x, reduction='none')
        log_p = torch.sum(log_p, dim=[2, 3])
        log_p = torch.clip_(log_p, min=-30, max=30)

        return log_p

    def random_sample(self, mean, logVar):
        std = torch.exp_(0.5 * logVar)
        epsilon = torch.randn_like(std)

        return mean + std * epsilon
