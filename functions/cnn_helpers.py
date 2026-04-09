### Functions to assist dataset preprocessing and CNN training

# Import packages
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset
from PIL import Image

import shutil
import os
import random
import torch


# Define classes
class trace(Dataset):
    """
    Dataset class to load .tsv files into tensors for pytorch.
    Given paths should already define cross-validation (0-4) fold and set (train/validation/test)

    """

    def __init__(self, fold_set_path, transform=None):
        self.samples = []
        self.transform = transform

        # Get path and label of all images
        classes = sorted(
            [
                class_i
                for class_i in os.listdir(fold_set_path)
                if os.path.isdir(os.path.join(fold_set_path, class_i))
            ],
            reverse=True,
        )
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for label in os.listdir(fold_set_path):
            class_dir = os.path.join(fold_set_path, label)
            if not os.path.isdir(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.endswith(".tsv"):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, int(class_to_idx[label])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        # Reshape label
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)

        # Load .tsv file
        trace_arr = np.array(pd.read_csv(path, sep="\t", header=None))

        # Convert to tensor
        trace_tensor = torch.tensor(trace_arr, dtype=torch.float32)

        # Add channel dimension for CNNs
        trace_tensor = trace_tensor.unsqueeze(0)  # (1, H, W)

        if self.transform:
            trace_tensor = self.transform(trace_tensor)

        return trace_tensor, label


class CNN_1(nn.Module):
    """
    Module class to define CNN based on Rajpurkar et al 2021.

    """

    # Contructor
    def __init__(self, out_1, kernel_cnn_1, out_2, kernel_cnn_2, h, w):
        super(CNN_1, self).__init__()

        # First Conv2D
        self.cnn1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_1,
            kernel_size=kernel_cnn_1,
            stride=1,
            padding="same",
        )
        # First batch normalization
        self.conv1_bn = nn.BatchNorm2d(out_1)
        # First pooling
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Second Conv2D
        self.cnn2 = nn.Conv2d(
            in_channels=out_1,
            out_channels=out_2,
            kernel_size=kernel_cnn_2,
            stride=1,
            padding="valid",
        )
        # Second batch normalization
        self.conv2_bn = nn.BatchNorm2d(out_2)
        # Second pooling
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Calculate fully connected layer dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 1, h, w)
            x = self.cnn1(dummy)
            x = self.conv1_bn(x)
            x = torch.relu(x)
            x = self.maxpool1(x)

            x = self.cnn2(x)
            x = self.conv2_bn(x)
            x = torch.relu(x)
            x = self.maxpool2(x)

            flattened_dim = x.view(1, -1).size(1)

        # Fully connected layer - FIX dimensions
        self.fc1 = nn.Linear(flattened_dim, 1)

    # Prediction
    def forward(self, x):
        # First convolutional layer
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        # Second convolutional layer
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)

        # Fully connected layer with sigmoid activation function
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.sigmoid(x)

        return x


# Define functions
def kfold_splits(
    src_folder, dst_folder, test_size=0.2, val_size=0.25, n_splits=5, seed=42
):
    """
    Creates n_splits different splits for k-fold cross-validation.
    Saves raw and normalized (z-score) images as .PNG in dst_folder}/fold_i/{train,validation,test}/transcription_state.

    Parameters
    ----------
    src_folder: str
        Path to folder containing entire dataset.
    dst_folder: str
        Path to folder where splits will be saved.
    test_size: float
        Fraction of dataset included in test set.
    val_size: float
        Fraction of dataset extracted from training dataset to create validation set.
    n_splits: int
        Number of splits in k-fold cross-valudation

    Returns
    -------

    """

    random.seed(seed)
    np.random.seed(seed)
    shutil.rmtree(dst_folder, ignore_errors=True)

    # Get labels and assign indices
    classes = [
        d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))
    ]
    classes = sorted(classes, reverse=True)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Get paths of all files and corresponding labels
    filepaths = []
    labels = []
    for cls in classes:
        cls_path = os.path.join(src_folder, cls)
        for fname in os.listdir(cls_path):
            if fname.lower().endswith(".tsv"):
                filepaths.append(os.path.join(cls_path, fname))
                labels.append(class_to_idx[cls])

    filepaths = np.array(filepaths)
    labels = np.array(labels)

    # Generate splits
    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=seed
    )

    return_folder = []

    # Get test set and an intermediate train one first (contains final training and validation sets)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(filepaths, labels)):
        print(f"Creating fold {fold_idx + 1}/{n_splits}...")

        # Get images and labels corresponding to each set
        X_test = filepaths[test_idx]
        y_test = labels[test_idx]

        X_train_val = filepaths[train_idx]
        y_train_val = labels[train_idx]

        # Further split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_size,
            stratify=y_train_val,
            random_state=42,
        )
        print(
            f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}"
        )

        # Read training set to calculate mean and standard deviation for normalization
        training_images = []
        for filepath_i in X_train:
            image_i = np.array(pd.read_csv(filepath_i, sep="\t", header=None))
            training_images.append(image_i)
        training_images = np.dstack(training_images)

        mean_train = np.mean(training_images)
        std_train = np.std(training_images)

        # Save split raw and normalized files
        fold_dir_raw = os.path.join(dst_folder + "_raw", f"fold_{fold_idx}")
        os.makedirs(fold_dir_raw, exist_ok=True)
        fold_dir_norm = os.path.join(dst_folder + "_zscore", f"fold_{fold_idx}")
        os.makedirs(fold_dir_norm, exist_ok=True)

        for set_i, filepaths_phase_i, labels_phase_i in [
            ("train", X_train, y_train),
            ("validation", X_val, y_val),
            ("test", X_test, y_test),
        ]:  # Iterate through all three datasets
            for cls in classes:
                os.makedirs(os.path.join(fold_dir_raw, set_i, cls), exist_ok=True)
                os.makedirs(os.path.join(fold_dir_norm, set_i, cls), exist_ok=True)
            for filepath_i, label_i in zip(
                filepaths_phase_i, labels_phase_i
            ):  # Iterate through each image
                # Open .tsv file
                image_i = np.array(pd.read_csv(filepath_i, sep="\t", header=None))

                # Raw images - save as .PNG
                dst_path_raw = os.path.join(
                    fold_dir_raw,
                    set_i,
                    classes[label_i],
                    os.path.basename(filepath_i.split("/")[-1].split(".")[0] + ".png"),
                )
                image_i_raw = image_i / image_i.max() * 255
                image_i_raw = image_i_raw.astype(np.uint8)
                Image.fromarray(image_i_raw).save(dst_path_raw)

                # Normalized images - save as .tsv, as .PNG would require clipping negative values
                dst_path_norm = os.path.join(
                    fold_dir_norm,
                    set_i,
                    classes[label_i],
                    os.path.basename(filepath_i),
                )

                image_i_norm = (image_i - mean_train) / std_train
                image_i_norm = pd.DataFrame(image_i_norm).to_csv(
                    dst_path_norm, sep="\t", header=False, index=False
                )
    print(f"Created {n_splits} stratified splits under '{dst_folder}'.")

    return


def train_model(
    n_epochs, train_loader, validation_loader, optimizer, model, criterion, device
):
    """
    Trains model for n_epochs epochs.

    Parameters
    ----------
    n_epochs: int
        Number of epochs performed during training.
    train_loader: torch.utils.data.DataLoader
        DataLoader containing training set.
    validation_loader: torch.utils.data.DataLoader
        DataLoader containing validation set.
    optimizer:
        Optimizer used for gradient descent.
    model: nn.Module
        CNN model that will be optimized.
    criterion:
        Metric used for loss quantification.
    device:
        GPU specification.

    Returns
    -------
    train_loss: list
        Training loss.
    val_loss: lists
        Validation loss.

    """
    # Variables to store training metrics
    train_loss = []
    val_loss = []

    for epoch in range(n_epochs):

        if (epoch % 25 == 0) and epoch >= 25:
            print(f"Training epoch {epoch}...")

        # Train using training set
        model.train()

        running_loss = 0.0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)

            loss.backward()
            optimizer.step()

            # Calculate running loss for current batch (take into account batch size)
            running_loss += loss.item() * x.size(0)

        # Calculate loss for entire epoch
        train_loss.append(running_loss / len(train_loader.dataset))

        # Evaluate using validation set
        model.eval()

        running_loss = 0.0
        correct = 0

        with torch.no_grad():  # No need to calculate gradiend descent
            for x, y in validation_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)

                # Calculate running loss for current batch (take into account batch size)
                running_loss += loss.item() * x.size(0)

            # Calculate loss for entire epoch
            val_loss.append(running_loss / len(validation_loader.dataset))

    return train_loss, val_loss


def run_predictions(model, test_loader, device):
    """
    Calculate predictions on test dataset.

    Parameters
    ----------
    model: nn.Module
        Trained model to run predicions on test set.
    test_loader: torch.utils.data.DataLoader
        DataLoader containing test set.
    device:
        GPU specification.

    Returns
    -------
    y_pred_all: list
        List of predicions.
    label_all: list
        List of true labels.

    """

    y_pred_all = []
    y_pred_score_all = []
    label_all = []

    model.eval()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            y_pred_score = model(imgs)
            y_pred = y_pred_score >= 0.5
            y_pred_all.extend(y_pred.long().view(-1).cpu().numpy())
            y_pred_score_all.extend(y_pred_score.view(-1).cpu().numpy())
            label_all.extend(labels.long().view(-1).numpy())

    y_pred_all = [int(x) for x in y_pred_all]
    y_pred_score_all = [float(x) for x in y_pred_score_all]
    label_all = [int(x) for x in label_all]

    return y_pred_all, y_pred_score_all, label_all


def save_losses(training_loss, validation_loss, path):
    """
    Calculate predictions on test dataset.

    Parameters
    ----------
    training_loss: list
        Training losses.
    validation_loss: list
        Validation losses.
    path: str
        Path to saved image

    Returns
    -------

    """

    fig, ax = plt.subplots(1, 1)
    ax.plot(training_loss, label="training loss")
    ax.plot(validation_loss, label="validation_loss")

    ax.set_xlim(0, len(training_loss))
    ax.set_ylim(0, np.max([np.max(validation_loss), np.max(training_loss)]))

    ax.legend()

    fig.savefig(path, format="pdf")
    plt.close()


def save_eval_results(path, split, accuracy, report, cm, cm_norm):
    """
    Saves test set evaluation data.

    Parameters
    ----------
    path: str
        Path to saved image
    split: str
        Current data split
    accuracy: float
        Te st accuracy
    report: str
        Classification report for test set
    cm: numpy.ndarray
        Confusion matrix on test set
    cm_norm: numpy.ndarray
        Normalized confusion matrix on test set

    Returns
    -------

    """

    with open(path, "w") as f:
        f.write("EVALUATION RESULTS\n\n")
        f.write(f"Split: {split}\n\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write(f"Classification report:\n\n")
        f.write(report)
        f.write("\n\nConfusion matrix:\n\n")
        f.write(f"{cm}\n")
        f.write("\n\nNormalized confusion matrix:\n\n")
        f.write(f"{cm_norm}\n")

    f.close()


def save_roc(fpr, tpr, auc, path):
    """
    Saves the ROC and ROC AUC.

    Parameters
    ----------
    fpr: numpy.ndarray
        False positive rate for each threshold
    tpr: numpy.ndarray
        True positive rate for each threshold
    auc: float
        Area under the curve for ROC
    path: str
        Path to saved image

    Returns
    -------

    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    ax.legend()

    fig.savefig(path, format="pdf")
    plt.close()
