

import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import MNISTTrainDataset, MNISTValDataset, MNISTSubmissionDataset

import numpy as np
from torch.utils.data import DataLoader, Dataset


def get_loaders(train_df_dir, test_df_dir, submission_df_dir, batch_size):
    train_df = pd.read_csv(train_df_dir)
    test_df = pd.read_csv(test_df_dir)
    submission_df = pd.read_csv(submission_df_dir)

    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    train_dataset = MNISTTrainDataset(train_df.iloc[:, 1:].values.astype(np.uint8), train_df.iloc[:, 0].values,
                                      train_df.index.values)
    val_dataset = MNISTValDataset(val_df.iloc[:, 1:].values.astype(np.uint8), val_df.iloc[:, 0].values,
                                  val_df.index.values)
    test_dataset = MNISTSubmissionDataset(test_df.iloc[:, 1:].values.astype(np.uint8), test_df.index.values)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader