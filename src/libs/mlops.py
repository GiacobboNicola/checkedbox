import torch
from torch.utils.data import DataLoader


def train(dnn, training_set, optimizer, loss_fuction, device="cpu", batch_size=64):
    """Training function that optimizes the network weights."""
    # creating list to hold loss per batch
    loss_per_batch = []

    # defining dataloader
    train_loader = DataLoader(training_set, batch_size, shuffle=True)

    # iterating through batches
    print("\ttraining ...")
    for images, labels in train_loader:
        # send images to device
        images, labels = images.to(device), labels.to(device)
        # zeroing optimizer gradients
        optimizer.zero_grad()
        # classifying images
        classifications = dnn(images)
        # get the loss
        loss = loss_fuction(classifications, labels)
        loss_per_batch.append(loss.item())
        # compute the gradients
        loss.backward()
        # optimizing the weights
        optimizer.step()
    print("\t... complited!")

    return loss_per_batch


def validation(dnn, validation_set, loss_function, device="cpu", batch_size=64):
    """Validation function that validates the network parameter optimizations."""
    # creating list to hold loss per batch
    loss_per_batch = []
    # defining model state
    dnn.eval()
    # defining dataloader
    val_loader = DataLoader(validation_set, batch_size, shuffle=True)

    print("\tvalidating ...")
    with torch.no_grad():
        # iterating through batches
        for images, labels in val_loader:
            # send images to device
            images, labels = images.to(device), labels.to(device)
            # classifying images
            classifications = dnn(images)
            # computing the loss
            loss = loss_function(classifications, labels)
            loss_per_batch.append(loss.item())
    print("\t... complited!")

    return loss_per_batch


def accuracy(dnn, dataset, device="cpu", batch_size=64):
    """This function computes accuracy."""
    # setting model state
    dnn.eval()

    # instantiating counters
    total_correct = 0
    total_instances = 0
    # creating dataloader
    dataloader = DataLoader(dataset, batch_size)

    # iterating through batches
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            # making classifications and deriving indices of maximum value via argmax
            classifications = torch.argmax(dnn(images), dim=1)
            # comparing indicies of maximum values and labels
            correct_predictions = sum(classifications == labels).item()
            # incrementing counters
            total_correct += correct_predictions
            total_instances += len(images)

    return round(total_correct / total_instances, 3)
