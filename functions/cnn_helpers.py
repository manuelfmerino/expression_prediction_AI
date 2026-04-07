### Functions to assist dataset preprocessing and CNN training

# Import packages
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from PIL import Image

import shutil
import os
import random
import torch

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
    return_folder:
        .

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

    #     # Reporting
    #     # train_counts = np.bincount(labels[train_idx], minlength=len(classes))
    #     # test_counts  = np.bincount(labels[test_idx],  minlength=len(classes))
    #     print(f"Fold {fold_idx}:")
    #     # for cls_idx, cls in enumerate(classes):
    #     #    print(f"  {cls}: Train {train_counts[cls_idx]}, Test {test_counts[cls_idx]}")
    #     # use crop_and_rotate_augmentation_training_set for each fold
    #     tv.crop_and_rotate_augmentation_training_set(
    #         os.path.join(fold_dir, "train"),
    #         os.path.join(fold_dir + "_cropped", "train"),
    #         CROP_PERCENT,
    #         AUG_PER_IMAGE,
    #         IMAGE_SIZE,
    #     )
    #     # print number of augmented images and number of test images  per each class
    #     for cls in classes:
    #         aug_train_dir = os.path.join(fold_dir + "_cropped", "train", cls)
    #         train_count = len(os.listdir(aug_train_dir))
    #         test_count = len(os.listdir(os.path.join(fold_dir, "test", cls)))
    #         print(f"  {cls}: Augmented Train {train_count}, Test {test_count}")
    #     # crop test set images
    #     tv.crop_test_set(
    #         os.path.join(fold_dir, "test"),
    #         os.path.join(fold_dir + "_cropped", "test"),
    #         CROP_PERCENT,
    #     )
    #     shutil.rmtree(fold_dir)
    #     shutil.move(
    #         fold_dir + "_cropped", fold_dir
    #     )  # rename cropped folder to original fold dir
    #     return_folder.append(fold_dir)
    # print(f"\nCreated {n_splits} stratified splits under '{dst_folder}'.")
    # # return list of fold directories

    return
