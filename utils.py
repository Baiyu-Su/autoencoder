import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import numpy as np


# make a grid plot with 100 images
def grid_generation(img_list, save_path):
    images_tensor = torch.cat(img_list, dim=0)
    examples = images_tensor.clone().detach()
    # select first 100 numbers to generate a grid
    output_examples = examples[0:100, :, :, :]
    grid = torchvision.utils.make_grid(output_examples, nrow=10, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(save_path)
    plt.show()


# sample images from autoencoder
def sample(decoder, latent_dim, batch_size):
    h = torch.randn(batch_size, latent_dim)
    tensor = decoder(h)

    return tensor


# concatenate two images together to compare
def image_generation(path1, path2, save_path):
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    comb_img = cv2.hconcat([img1, img2])
    cv2.imwrite(save_path, comb_img)
