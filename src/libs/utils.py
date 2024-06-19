import glob
import os
import random
import cv2
from PIL import Image
import torchvision.transforms as transforms
import configargparse
import imutils
from datetime import datetime
import numpy as np

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
    parser.add_argument("-b", "--batches", type=int, default=64, help="number of batches")
    parser.add_argument("-r", "--rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("-m", "--model", type=_possible_models, default="NGConvNet", help="model to use")
    parser.add_argument("-g", "--in-gray", action="store_true", default=False)
    parser.add_argument("-s", "--save", choices=["pth", "onnx", "not"], default="onnx")
    #    parser.add_argument("-s","--save", action="store_true", default=False)
    config = parser.parse_args()

    return config


def _possible_models(model):
    """"""
    if model not in POSSILE_MODELS:
        raise configargparse.ArgumentTypeError(f"{model} is not a valid model")
    return model


def get_stat_dataset(root_dir):
    imgs_list = []
    classes_path = glob.glob(root_dir + "*")
    for class_path in classes_path:
        img_paths = glob.glob(class_path + "/*")
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (45, 45))
            pil_img = Image.fromarray(img)
            pil_img = pil_img.convert("L")
            tensor_img = transforms.ToTensor()(pil_img)
            imgs_list.append(tensor_img)
            img = imgs_list.append(tensor_img)
            break
        break

    # imgs = torch.stack(imgs_list, dim=1)
    # norms = imgs.view(1, -1).mean(dim=1)
    # stds = imgs.view(1, -1).std(dim=1)
    return img


def save_model(model, epoch_num, format: str):
    print(f"saving model weights in {format}...")
    now = datetime.now()
    save_filename = f"{now.strftime('%d%m%Y%H%M%S')}_{epoch_num}"
    save_path = os.path.join(os.getcwd(), "models", save_filename + "." + format)
    if format == "pth":
        import torch

        torch.save(model.cpu().state_dict(), save_path)
    elif format == "onnx":
        import torch.onnx

        img_tensor = torch.randn(1, 3, 45, 45)
        torch.onnx.export(
            model,
            img_tensor,
            save_path,
        )
        #   export_params=True,
        #   opset_version=10,
        #   verbose=True,              # Print verbose output
        #   input_names=['input'],     # Names for input tensor
        #   output_names=['output'])
    print("... done!")


def rotate(path, rotations=[0, 90, 180, 270]):
    img = cv2.imread(path)
    path, filename = os.path.split(path)
    to_delete = False

    # rotate the image by r degree clockwise
    for angle in rotations:
        img_rotated = imutils.rotate(img, angle=angle)
        if len(filename.split(".")) == 3:
            name, _, ext = filename.split(".")
        elif len(filename.split(".")) == 2:
            name, ext = filename.split(".")

        # check to avoid saving the same rotation multiple times
        if int(name.split("_")[-1]) not in rotations:
            print("saving")
            cv2.imwrite(os.path.join(path, name + "_" + str(angle) + "." + ext), img_rotated)
            to_delete = True

    if to_delete:
        file = os.path.join(path, filename)
        os.remove(file)


def clean(path):
    img = cv2.imread(path)
    path, filename = os.path.split(path)

    if len(filename.split(".")) == 3:
        name, _, ext = filename.split(".")
    elif len(filename.split(".")) == 2:
        name, ext = filename.split(".")

    new_name = os.path.join(path, name + "." + ext)
    cv2.imwrite(new_name, img)

    file = os.path.join(path, filename)
    os.remove(file)


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


def sp_noise(image, b_prob=0, w_prob=0):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < b_prob:
                output[i][j] = (0, 0, 255)
            elif rdn < b_prob + w_prob:
                output[i][j] = (255, 255, 0)
            else:
                output[i][j] = image[i][j]
    return output


def debug_sp(img_path):
    img = cv2.imread(img_path)
    # img2 = add_salt_pepper(img, s_vs_p=0.5, amount=0.00004)

    img2 = sp_noise(img, b_prob=0.01, w_prob=0.02)

    img = cv2.resize(img, (150, 150))
    img2 = cv2.resize(img2, (150, 150))

    all = np.concatenate(
        (img, img2),
        axis=1,
    )
    cv2.imshow("test", all)
    cv2.waitKey(1000)
