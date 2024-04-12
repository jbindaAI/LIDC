## Some useful functions

import numpy as np
import torch
import random


def compute_norm_factors(X_train, datapath):
    imgs = []
    for elt in X_train:
        crop = torch.load(datapath+f"/crops/{elt}")
        
        view = random.choice([1, 2, 3])
        slices = np.linspace(14, 18, 5).astype(int)
        slice_ = random.choice(slices)
        
        if view == 1:
            img = crop[:, :, slice_]
        elif view == 2:
            img = crop[:, slice_, :]
        elif view == 3:
            img = crop[slice_, :, :]
    
        img = torch.clamp(img, -1000, 400).float()
    
        if (len(img.shape) < 3):
            img = img.unsqueeze(0)
    
        img = img.repeat(3,1,1)
    
        imgs.append(img)
    
    imgs = torch.stack(imgs, axis=0)
    
    channels_mean = torch.mean(imgs)
    channels_std = torch.std(imgs)
    return [channels_mean, channels_std]