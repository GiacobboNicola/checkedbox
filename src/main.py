import os
import torch
import torch.nn as nn

# custom modules
from libs.utils import save_model, set_config
from libs.dataset import BoxesDataset
from libs.models import NGConvNet
from libs.mlops import train, validation, accuracy


def run():
    config = set_config()
    # TODO: write docstring
    train_dir = os.path.join(config.dataset, "train/")
    val_dir = os.path.join(config.dataset, "validation/")

    # upload the dataset for both training and validation
    training_set = BoxesDataset(train_dir, config.in_gray, format=".jpg")
    validation_set = BoxesDataset(val_dir, config.in_gray, format=".jpg")

    # instantiating model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: switch between models
    model = NGConvNet(3, in_gray=config.in_gray).to(device, non_blocking=True)
    # defining optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # TODO: log config
    for epoch in range(config.epochs):
        # training/optimizing parameters
        # TODO: print log
        print(f"Starting epoch {epoch}")

        # training_losses
        _ = train(
            model,
            training_set,
            optimizer,
            nn.CrossEntropyLoss(),
            device=device,
            batch_size=config.batches,
        )
        # validation_losses
        _ = validation(
            model,
            validation_set,
            nn.CrossEntropyLoss(),
            batch_size=config.batches,
            device=device,
        )
        training_accuracy = accuracy(model, training_set, device=device)
        print(f"\ttraining accuracy: {training_accuracy}")
        validation_accuracy = accuracy(model, validation_set, device=device)
        print(f"\tvalidation accuracy: {validation_accuracy}")

    if config.save != "not":
        save_model(model, config.epochs, config.save)


if __name__ == "__main__":
    run()
