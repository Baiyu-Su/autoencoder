import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IWAE import IWAE
import time
import torchvision
import cv2


# initialize the training
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set hyperparameters
batch_size = 20
epoch = 3
learning_rate = 1e-3

# Datasets
train_dataset = datasets.MNIST('~/PycharmProjects/IWAE',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

test_dataset = datasets.MNIST('~/PycharmProjects/IWAE',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
)

test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1
)

# set the neural network and Adam optimizer
net = IWAE(batch=batch_size, width=28, height=28, latent_dim=8, k=5).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# initialize the list to store loss vs epoch
num_of_epoch = np.arange(epoch) + 1
list_of_training_loss = np.zeros_like(num_of_epoch, dtype='float32')

# initialize the list to store original/reconstructed images
original_images_list = []
out_activated_images_list = []


def train():

    for i in range(epoch):
        training_loss = 0

        for data in train_loader:

            # load data
            img, _ = data
            img = img.to(device)
            img = torch.where(img > 0.5, 1.0, 0.0)

            # retrieve the outputs from the neural network (out is the direct output without final activation)
            out_list, mean, logVar = net(img)

            # calculate loss function and the average loss for each image over the whole dataset
            loss = net.loss_function(mean=mean, logVar=logVar, out_list=out_list, img=img)
            optimizer.zero_grad()
            training_loss += loss.item()/len(train_loader.sampler)

            # back prop
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

        # note the performance of each epoch
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch: {}, Loss: {}, Time: {}'.format(i, training_loss, duration))
        list_of_training_loss[i] = training_loss

    # plot loss vs num of epoch
    plt.plot(num_of_epoch, list_of_training_loss, '-r', label='avg train loss')
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('loss')
    plt.savefig('~/PycharmProjects/IWAE/images/loss_vs_epoch.png')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    train()
