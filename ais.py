import torch
import numpy as np
import tqdm
from autoencoder import logp_h
from autoencoder import logp_x_h
from autoencoder import random_sample


def ais(decoder, img, K, T, batch, sampleTimes, latent_dim=8):
    '''
    This function performs the forward step of annealed importance sampling and returns the upper bound
    of the negative log likelihood (the lower bound of marginal likelihood).
    :param decoder: the decoder module from network VAE or IWAE
    :param img: original images inputted to network
    :param K: times of annealed importance sampling for each item in the batch
    :param T: total steps of one anneal importance sampling
    :param batch: batch size
    :param sampleTimes: total times a network sampled from the normal distribution
    :param latent_dim: dimension of latent layer
    :return: negative log lower bound of marginal likelihood
    '''

    # return a list of beta to be the exponent of likelihood function
    beta = torch.linspace(start=0.0, end=1.0, steps=T)

    # prepare a list of w to store the partition function calculated each time
    logw = [0] * K

    # define f_t to be the intermediate distributions and return the log value
    def logf_t(t, z):

        return logp_h(z) + torch.squeeze(logp_x_h(x=img, x_hat=decoder(z))) * beta[t]

    # return the acceptance rate of M-H sampling algorithm
    def accept(t, z_list):
        log_a = logp_h(z_list[t - 1]) + logf_t(t, z_list[t]) - logp_h(z_list[t]) - logf_t(t, z_list[t - 1])
        log_a = log_a.double()
        a = torch.exp_(torch.where(log_a > 0.0, 0.0, log_a))
        return a

    # sample K independent times in total
    for k in range(K):

        z_list = [0] * T
        z_list[1] = torch.randn(batch, latent_dim)
        f_values = [0] * T

        for t in range(2, T - 1):

            # record the partition function each time
            logw[k] = logw[k] + logf_t(t, z_list[t - 1]) - logf_t(t - 1, z_list[t - 1])

            # z is initially sampled from normal distribution
            z_list[t] = torch.randn(batch, latent_dim)

            # use t and z to define acceptance rate
            a = accept(t, z_list)

            # u is sampled from a uniform distribution and z will be updated according to the comparison of u and a
            u = torch.rand_like(a)
            for i in range(batch):
                if u[i] > a[i]: z_list[t][i, :] = z_list[t - 1][i, :]

    # average over the partition function from the K samples
    logZ = torch.logsumexp(torch.stack(logw, dim=1), dim=1) - torch.log(torch.tensor(K))
    logZ = logZ.clone().detach()
    logZ = logZ.float()

    return - torch.mean(logZ, dim=0)
