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
import numpy as np
from datetime import datetime
from torch import nn

# custom modules
from libs.utils import save_model, set_config
from libs.dataset import BoxesDataset
from libs.models import NGConvNet
from libs.mlops import train, validation, accuracy, train_transformers
from tqdm import tqdm


def run():
    """Runs the training and validation steps and compute their accuracy."""
    config = set_config()
    # TODO: write docstring
    train_dir = os.path.join(config.dataset, "train/")
    val_dir = os.path.join(config.dataset, "validation/")

    # upload the dataset for both training and validation
    training_set = BoxesDataset(train_dir, config.in_gray, transform=train_transformers, format=".jpg")
    validation_set = BoxesDataset(val_dir, config.in_gray, format=".jpg")

    # instantiating model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: switch between models according to config.model
    model = NGConvNet(3, in_gray=config.in_gray).to(device, non_blocking=True)
    # defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    train_history = []
    validation_history = []
    loss_history_batches = []
    accuracy_history = []
    loss_train = 0.0
    train_accuracy = 0.0

    # TODO: log config
    for epoch in (pbar := tqdm(range(config.epochs))):
        pbar.set_description(f"Epoch {epoch+1}-> loss {round(loss_train,2)} accuracy {round(train_accuracy,2)}")

        # training/optimizing parameters
        # TODO: save log

        # training_losses
        loss_train_batches, loss_train = train(
            model,
            training_set,
            optimizer,
            nn.CrossEntropyLoss(),
            device=device,
            batch_size=config.batches,
        )

        # validation_losses
        loss_validation_batches, loss_validation = validation(
            model,
            validation_set,
            nn.CrossEntropyLoss(),
            batch_size=config.batches,
            device=device,
        )

        train_history.append([loss_train])
        validation_history.append([loss_validation])
        loss_history_batches.append([loss_train_batches, loss_validation_batches])

        train_accuracy = accuracy(model, training_set, device=device, batch_size=config.batches)
        #        print(f"\ttraining accuracy: {training_accuracy}")
        validation_accuracy = accuracy(model, validation_set, device=device, batch_size=config.batches)
        #        print(f"\tvalidation accuracy: {validation_accuracy}")

        accuracy_history.append([train_accuracy, validation_accuracy])

    if config.save != "not":
        save_model(model, device, config.epochs, config.save)

    if config.plot:
        #! move to utils
        import matplotlib.pyplot as plt
        from matplotlib.legend_handler import HandlerTuple

        colors = plt.cm.viridis(np.linspace(0, 1, config.epochs))

        # Placing the plots in the plane
        plot1 = plt.subplot2grid((2, 4), (0, 0), colspan=2)
        plot2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
        plot3 = plt.subplot2grid((2, 4), (0, 2), rowspan=2, colspan=2)

        train_loss = [losses[0] for losses in loss_history_batches]
        lines = []
        for epoch, losses in enumerate(train_loss):
            (line,) = plot1.plot(losses, color=colors[epoch])
            lines.append(line)
        plot1.legend(
            handles=[tuple(lines)],
            labels=["Epoch"],
            handlelength=3,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        )
        plot1.set_ylabel("loss")

        validation_loss = [losses[1] for losses in loss_history_batches]
        lines = []
        for epoch, losses in enumerate(validation_loss):
            (line,) = plot2.plot(losses, color=colors[epoch])
            lines.append(line)
        plot2.legend(
            handles=[tuple(lines)],
            labels=["Epoch"],
            handlelength=3,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},
        )
        plot2.set_xlabel("batch")
        plot2.set_ylabel("loss")

        plot3.plot(train_history, color="black", ls="-", label="train")
        plot3.plot(validation_history, color="red", ls="--", label="Validation")

        plot3.legend()
        plot3.set_xlabel("epoch")
        plot3.set_ylabel("loss")

        plt.tight_layout()

        if config.save:
            # Saving the figure.
            plt.savefig(
                os.path.join(
                    os.getcwd(), f"data/plots/loss_plot_{datetime.now().strftime('%d%m%Y%_H%M%S')}_{config.epochs}.jpg"
                )
            )
        else:
            plt.show()


def prepare():
    pass


if __name__ == "__main__":
    run()
