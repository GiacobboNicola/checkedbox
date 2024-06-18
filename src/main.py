"""This script trains an neural network model for image classification using a custom dataset.
It performs training and validation loops, calculating
accuracy on both sets. The script can also save the trained model.

**Configuration:**

* `dataset`: Path to the dataset directory.
* `in_gray`: Boolean flag indicating if images should be loaded in grayscale.
* `epochs`: Number of training epochs.
* `batches`: Batch size for training and validation.
* `save`: Value to specify the format on which save the trained model (or "not" to disable saving).
"""

import os
import torch
from torch import nn

# custom modules
from libs.utils import save_model, set_config
from libs.dataset import BoxesDataset
from libs.models import NGConvNet
from libs.mlops import train, validation, accuracy


def run():
    """Runs the training and validation steps and compute their accuracy."""
    config = set_config()
    # TODO: write docstring
    train_dir = os.path.join(config.dataset, "train/")
    val_dir = os.path.join(config.dataset, "validation/")

    # upload the dataset for both training and validation
    training_set = BoxesDataset(train_dir, config.in_gray, format=".jpg")
    validation_set = BoxesDataset(val_dir, config.in_gray, format=".jpg")

    # instantiating model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: switch between models according to config.model
    model = NGConvNet(3, in_gray=config.in_gray).to(device, non_blocking=True)
    # defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    loss_history = []

    # TODO: log config
    for epoch in range(config.epochs):
        # training/optimizing parameters
        # TODO: print log
        print(f"Starting epoch {epoch}")

        # training_losses
        loss_train = train(
            model,
            training_set,
            optimizer,
            nn.CrossEntropyLoss(),
            device=device,
            batch_size=config.batches,
        )

        # validation_losses
        loss_validation = validation(
            model,
            validation_set,
            nn.CrossEntropyLoss(),
            batch_size=config.batches,
            device=device,
        )

        loss_history.append([loss_train, loss_validation])

        training_accuracy = accuracy(model, training_set, device=device, batch_size=config.batches)
        print(f"\ttraining accuracy: {training_accuracy}")
        validation_accuracy = accuracy(model, validation_set, device=device, batch_size=config.batches)
        print(f"\tvalidation accuracy: {validation_accuracy}")

    if config.save != "not":
        save_model(model, device, config.epochs, config.save)

    if config.plot:
        import matplotlib.pyplot as plt

        for epoch, losses in enumerate(loss_history):
            print(len(losses))
            plt.plot(losses[0], label=f"epoch {epoch}")

        plt.legend()

        if config.save:
            # Saving the figure.
            plt.savefig(os.path.join(os.getcwd(), "data/plots/loss_plot.jpg"))
        else:
            plt.show()


def prepare():
    pass


if __name__ == "__main__":
    run()
