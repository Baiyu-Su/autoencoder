import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image
from autoencoder import autoencoder
from utils import grid_generation
from utils import sample
from utils import image_generation
import time
import torchvision
import cv2


# initialize the training
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set hyperparameters
batch_size = 20
epoch = 10
learning_rate = 1e-3

parameters_dict = {
    'latent_dim_VAE': 6,
    'latent_dim_IWAE': 16,
    'k_VAE': 1,
    'k_IWAE': 5
}

# Datasets
train_dataset = datasets.MNIST('/Users/byronsu/PycharmProjects/autoencoder',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

test_dataset = datasets.MNIST('~/PycharmProjects/autoencoder',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)


# set the neural network and Adam optimizer
net = autoencoder(channel=1,
                  width=28,
                  height=28,
                  hyperparameters=parameters_dict,
                  mode='IWAE').to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# initialize the list to store loss vs epoch
num_of_epoch = np.arange(epoch) + 1
list_of_training_loss = np.zeros_like(num_of_epoch, dtype='float32')
list_of_validation_loss = np.zeros_like(num_of_epoch, dtype='float32')

# initialize the list to store original/reconstructed images
original_images_list = []


def train(mode='IWAE'):
    net.train()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size
    )

    for i in range(epoch):
        training_loss = 0.0
        validation_loss = 0.0

        for data in train_loader:
            img, _ = data
            # load data
            img = img.to(device)
            optimizer.zero_grad()

            # retrieve the outputs from the neural network (out is the direct output without final activation)
            h_list, out_list, mean, logVar = net(img)

            # calculate loss function and the average loss for each image over the whole dataset
            loss = net.loss_function(mean=mean, logVar=logVar, h_list=h_list, out_list=out_list, img=img)
            training_loss += loss.item()/len(train_loader.sampler)

            # back prop, turn on autograd to detect gradient vanishing/explosion
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

        for data in test_loader:
            img, _ = data
            img = img.to(device)

            h_list, out_list, mean, logVar = net(img)

            loss = net.loss_function(mean=mean, logVar=logVar, h_list=h_list, out_list=out_list, img=img)
            validation_loss += loss.item()/len(test_loader.sampler)

        # note the performance of each epoch
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch: {}, Loss: {}, Time: {}'.format(i, training_loss, duration))
        list_of_training_loss[i] = training_loss
        list_of_validation_loss[i] = validation_loss

    torch.save(net.state_dict(), '//nn_parameters/'+mode+'_model.pt')

    # plot loss vs num of epoch
    plt.plot(num_of_epoch, list_of_training_loss, '-r', label='avg train loss')
    plt.plot(num_of_epoch, list_of_validation_loss, '-b', label='avg valid loss')
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('loss')
    plt.savefig('//PycharmProjects/autoencoder/images/loss_vs_epoch.png')
    plt.show()


# evaluate the performance of VAE on the train dataset after training
def evaluate(mode='IWAE'):
    net.eval()

    # reconfigure the train loader to save running time
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=100,
    )

    # retrieve the first batch of images from the dataset
    data = next(iter(train_loader))

    # load data
    img, _ = data
    img = img.to(device)
    original_images_list.append(img)

    # pass the data trough network to find the reconstructed images
    _, out_list, mean, logVar = net(img)

    # retrieve parameters for later use in CNN decoder
    decoder = net.decoder_model()

    # show the original images for training
    grid_generation(img_list=original_images_list,
                    save_path='//images/original_images_'+mode+'.png')

    # show the reconstructed images through VAE
    grid_generation(img_list=out_list,
                    save_path='//images/reconstruction_'+mode+'.png')

    # show the sampled images
    sampled_images_list = [sample(decoder, parameters_dict['latent_dim_'+mode], 100).cpu().detach()]
    grid_generation(img_list=sampled_images_list,
                    save_path='//images/sampled_images.png')


if __name__ == '__main__':
    path1 = '//PycharmProjects/autoencoder/images/original_images.png'
    path2 = '//PycharmProjects/autoencoder/images/reconstruction.png'
    save_path = '//images/compare.png'
    start_time = time.time()
    train()
    evaluate()
    image_generation(path1=path1, path2=path2, save_path=save_path)
