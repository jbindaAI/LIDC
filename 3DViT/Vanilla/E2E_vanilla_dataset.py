# LIDC Dataset Module

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np
import glob
import pickle
from typing import Literal
import os

class Vanilla_Dataset(Dataset):
    def __init__(
        self, 
        datadir="data/", 
        transform=None,
        mode: Literal["train","val","test"]="train"
    ):
        self.transform = transform
        self.datadir = datadir
        self.mode = mode

        if self.mode == "train":
            self.X_data = os.listdir(self.datadir + "crops/train")
            self.X_data.sort()
            with open(self.datadir + "y_train_df.pkl", 'rb') as f:
                self.y_data = pickle.load(f)["target"]
                
        elif self.mode == "val":
            # Loading validational subset of data:
            self.X_data = os.listdir(self.datadir + "crops/val")
            self.X_data.sort()
            with open(self.datadir + "y_val_df.pkl", 'rb') as f:
                self.y_data = pickle.load(f)["target"]
                
        elif self.mode == "test":
            # Loading test subset of data:
            self.X_data = os.listdir(self.datadir + "crops/test")
            self.X_data.sort()
            with open(self.datadir + "y_test_df.pkl", 'rb') as f:
                self.y_data = pickle.load(f)["target"]
                
        self.targets = self.y_data.to_numpy()

    
    def __len__(self):
        return len(self.targets)

    
    def __getitem__(self, idx):
        # Label: 0 or 1
        label = self.targets[idx]
        img_id = self.X_data[idx]
        img = torch.load(self.datadir + f"crops/{self.mode}/{img_id}").float()
        # If some image transformations are specified:
        if self.transform is not None:
            img = self.transform(img)
        return [img, label]

    
    def get_target(self, idx):
        target = self.targets[idx]
        return target
    