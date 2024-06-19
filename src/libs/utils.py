import os
from datetime import datetime

import configargparse
import cv2
import imutils
from torchvision import transforms
import random
import copy
import numpy as np


import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.util import random_noise


# =====> Constants
POSSILE_MODELS = ["NGConvNet"]
CLASSES = ["empty", "right", "wrong"]
DIMS = (45, 45)


# =====> Help functions
def set_config():
    """Function that passes the paramenters to the main."""
    parser = configargparse.get_argument_parser()
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="path to the dataset",
    )
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epoch for the training")
    parser.add_argument("-p", "--plot", action="store_true", default=False, help="plot losses")
    parser.add_argument("-b", "--batches", type=int, default=64, help="number of batches")
    parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-m", "--model", type=_possible_models, default="NGConvNet", help="model to use")
    parser.add_argument("-g", "--in-gray", action="store_true", default=False)
    parser.add_argument("-s", "--save", choices=["pth", "onnx", "not"], default="onnx")
    #    parser.add_argument("-s","--save", action="store_true", default=False)
    config = parser.parse_args()

    return config


def _possible_models(model):
    """Return a valid model argument that can be used to create a model argument."""
    if model not in POSSILE_MODELS:
        raise configargparse.ArgumentTypeError(f"{model} is not a valid model")
    return model


def save_model(model, device, epoch_num, extension: str):
    """save a model weights to file

    Args:
        model ([type]): [description]
        epoch_num ([type]): [description]
        extension (str): [description]
    """
    print(f"saving model weights in {extension}...")
    now = datetime.now()
    save_filename = f"{now.strftime('%d%m%Y%H%M%S')}_{epoch_num}"
    save_path = os.path.join(os.getcwd(), "data/models", save_filename + "." + extension)
    if extension == "pth":
        import torch

        torch.save(model.cpu().state_dict(), save_path)
    elif extension == "onnx":
        import torch.onnx

        img_tensor = torch.randn(1, 3, 45, 45)
        torch.onnx.export(
            model.cpu(),
            img_tensor,
            save_path,
        )
        #   export_params=True,
        #   opset_version=10,
        #   verbose=True,              # Print verbose output
        #   input_names=['input'],     # Names for input tensor
        #   output_names=['output'])
    print("... done!")


# def get_stat_dataset(root_dir):
#     """Get the stat dataset in a directory of the cv2 library.
#     """
#     imgs_list = []
#     classes_path = glob.glob(root_dir + "*")
#     for class_path in classes_path:
#         img_paths = glob.glob(class_path + "/*")
#         for img_path in img_paths:
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (45, 45))
#             pil_img = Image.fromarray(img)
#             pil_img = pil_img.convert("L")
#             tensor_img = transforms.ToTensor()(pil_img)
#             imgs_list.append(tensor_img)
#             img = imgs_list.append(tensor_img)
#             break
#         break


#     # imgs = torch.stack(imgs_list, dim=1)
#     # norms = imgs.view(1, -1).mean(dim=1)
#     # stds = imgs.view(1, -1).std(dim=1)
#     return img
def add_salt_pepper(image, s_vs_p=0, amount=0.004):
    """
    Replaces random pixels with 0 or 1
    """
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 144

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1.0 - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out


def sp_noise2(image, b_prob=0, w_prob=0):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < b_prob:
                output[i][j] = 255
            elif rdn < b_prob + w_prob:
                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output


def sp_noise(image, n_pixel=2, color=255):
    n_b = random.randint(0, n_pixel)
    for _ in range(n_b):
        x = random.randint(0, image.shape[0] - 1)
        y = random.randint(0, image.shape[1] - 1)
        image[x][y] = color
    return image


def sp_noise_tensor(image, n_pixel=2, color=0):
    n_b = random.randint(0, n_pixel)
    for _ in range(n_b):
        x = random.randint(0, image.shape[1] - 1)
        y = random.randint(0, image.shape[2] - 1)
        image[0][x][y] = color
        image[1][x][y] = color
        image[2][x][y] = color
    return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {"image": torch.from_numpy(image), "landmarks": torch.from_numpy(landmarks)}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {"image": img, "landmarks": landmarks}


class Salt(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        salt_img = torch.tensor(random_noise(image, mode="salt", amount=0.5))

        return {"image": salt_img, "landmarks": landmarks}


def addnoise(img, noise_factor=0.8):
    inputs = transforms.ToTensor()(img)
    noise = sp_noise_tensor(inputs, n_pixel=20, color=0)
    noise = sp_noise_tensor(noise, n_pixel=20, color=144)
    # noise = torch.clip (noise,0,1.)
    output_image = transforms.ToPILImage()
    image = output_image(noise)
    image = np.array(image)
    return image


def debug_sp(img_path):
    img = cv2.imread(img_path)
    img2 = copy.copy(img)
    img2 = addnoise(img)

    img = cv2.resize(img, (150, 150))
    img2 = cv2.resize(img2, (150, 150))

    all = np.concatenate(
        (img, img2),
        axis=1,
    )
    cv2.imshow("test", all)
    cv2.waitKey(1000)
