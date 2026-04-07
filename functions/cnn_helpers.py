### Functions to assist dataset preprocessing and CNN training

# Import packages
import random
import torch
import numpy as np

import shutil
import os

from sklearn.model_selection import StratifiedShuffleSplit

# Define functions


def kfold_splits(
    src_folder, dst_folder, test_size=0.2, val_size=0.25, n_splits=5, seed=42
):
    """
    Creates n_splits different splits for k-fold cross-validation.
    Saves images as .PNG in dst_folder}/fold_i/{train,validation,test}/transcription_state.


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
    os.makedirs(dst_folder, exist_ok=True)

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

        # print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        # fold_dir = os.path.join(dst_folder, f"fold_{fold_idx}")
        # for phase, indices in [("train", train_idx), ("test", test_idx)]:
        #     for cls in classes:
        #         os.makedirs(os.path.join(fold_dir, phase, cls), exist_ok=True)
        #     for idx in indices:
        #         src_path = filepaths[idx]
        #         cls = classes[labels[idx]]
        #         dst_path = os.path.join(
        #             fold_dir, phase, cls, os.path.basename(src_path)
        #         )
        #         shutil.copy2(src_path, dst_path)

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

    return splitter, filepaths, labels
