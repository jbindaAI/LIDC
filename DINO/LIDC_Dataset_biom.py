# LIDC Dataset Module

from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np
import glob
import pickle

class LIDC_Dataset_biom(Dataset):
    def __init__(
        self, 
        datadir,
        transform=None,
        label_transform=None,
        mode="train"
    ):
        
        self.views = ["axial", "coronal", "sagittal"]
        self.transform = transform
        self.label_transform = label_transform
        self.datadir = datadir 

        if mode == "train":
            # Loading training subset of LIDC data:
            with open(self.datadir +"splitted_sets/"+ "X_train.pkl", 'rb') as f:
                self.X_data = pickle.load(f)
            with open(self.datadir +"splitted_sets/" + "y_train.pkl", 'rb') as f:
                self.y_data = pickle.load(f)
                
        elif mode == "val":
            # Loading training subset of LIDC data:
            with open(self.datadir + "splitted_sets/" + "X_val.pkl", 'rb') as f:
                self.X_data = pickle.load(f)
            with open(self.datadir + "splitted_sets/" + "y_val.pkl", 'rb') as f:
                self.y_data = pickle.load(f)
                
        elif mode == "test":
            with open(self.datadir + "splitted_sets/" + "X_test.pkl", 'rb') as f:
                self.X_data = pickle.load(f)
            with open(self.datadir + "splitted_sets/" + "y_test.pkl", 'rb') as f:
                self.y_data = pickle.load(f)

        df_concepts = self.y_data[["subtlety", "calcification", 
                                   "margin", "lobulation", 
                                   "spiculation", "diameter", 
                                   "texture", "sphericity"]].copy()
        self.concepts = df_concepts.to_numpy()
        self.targets = self.y_data["target"].to_numpy()
        
        imgs = []
        for elt in self.X_data:
            crop = torch.load(self.datadir + f"crops/{elt}").float()
            imgs.append(crop)
        self.images = imgs

    
    def __len__(self):
        return len(self.targets)

    
    def process_image(self, nodule_idx, view, slice_=16):
        # Firstly imgs are 3D volumes:
        img = self.images[nodule_idx]
        true_img = True if img.shape[0]==32 else False

        if true_img:
            ## Only true images (which are volumes 32x32x32) needs slice extraction.
            # Then, I extract slice of the volume
            # at the specified view.
            if view == self.views[0]:
                img = img[:, :, slice_]
    
            elif view == self.views[1]:
                img = img[:, slice_, :]
    
            elif view == self.views[2]:
                img = img[slice_, :, :]
            
            img = torch.clamp(img, -1000, 400) # Values in tensor are clamped in range (-1000, 400)
            
            if (len(img.shape) < 3):
                # Extracted slices are of shape: (32, 32).
                # There is need to add third dimmension -> channel.
                # After that we have (1, 32, 32).
                img = img.unsqueeze(0)
     
        # As ViT model requires 3 color channels,
        # code below makes 2 more channels by coping original channel.
        img = img.repeat(3,1,1)
            
        # If some image transformations are specified:
        if self.transform is not None:
            img = self.transform(img)
            
        return img.float()


    def __getitem__(self, idx):
        concepts = self.concepts[idx]
        if (self.label_transform is not None):
            scaler = self.label_transform
            concepts = scaler.transform(np.expand_dims(concepts, axis=0))[0]
        label = torch.tensor(concepts).float()
            
        # Scan:
        ## Generated images has only one slice (1, 32, 32) where
        ## original crops has 32 slices (32, 32, 32).
        ## Generated images are taken as are.
        ## From true crops there is randomly chosen slice in one of the three possible views.
             
        view = random.choice(self.views)
        slices = np.linspace(14, 18, 5).astype(int)
        slice_ = random.choice(slices) # one of the middle slices [14, 15, 16, 17, 18] is chosen.
        img = self.process_image(nodule_idx=idx, view=view, slice_=slice_)
        
        return [img, label]

    def get_target(self, idx):
        target = self.targets[idx]
        return target
    