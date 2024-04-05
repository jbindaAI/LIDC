################ CLASS CREATING TRAINING/VAL/TEST PyTorch DATASETS
from torch.utils.data import Dataset
import pandas as pd
import torch
import random
import numpy as np


class LIDC_Dataset(Dataset):
    def __init__(self, 
                data_dir='dataset',
                train_mode = True,
                apply_mask = False,
                transform = None,
                labels = 'targets',
                label_transform = None,
                return_mask = False,
                full_vol = False,
                finetuning = False):
        
        self.data_dir = data_dir
        crop_path = f"{data_dir}/crops"
        mask_path = f"{data_dir}/masks"
        
        df = pd.read_pickle(f"{data_dir}/ALL_annotations_df.pkl")
        self.targets = df['target']
        df_concepts = df[["subtlety", 
                         "calcification", 
                         "margin", 
                         "lobulation", 
                         "spiculation", 
                         "diameter", 
                         "texture", 
                         "sphericity"
                         ]].copy()
        self.concepts = df_concepts.to_numpy()

        imgs = []
        masks = []
        for i in range(len(self.targets)):
            imgs.append(torch.load(f"{crop_path}/{df['path'][i]}").float())
            masks.append(torch.load(f"{mask_path}/{df['path'][i]}").float())
        
        self.images = imgs
        self.masks = masks
        self.views = ["axial", "coronal", "sagittal"]
        
        # Other class attributes:
        self.train_mode = train_mode
        self.apply_mask = apply_mask
        self.transform = transform
        self.labels = labels
        self.label_transform = label_transform
        self.return_mask = return_mask
        self.full_vol = full_vol
        self.finetuning = finetuning
    
    def __len__(self):
        return len(self.targets)
    
    
    def process_image(self, nodule_idx, view, slice_=16):
        # Firstly img, mask are 3D volume:
        img = self.images[nodule_idx]
        mask = self.masks[nodule_idx]
        
        # Then, I extract slice of the volume
        # at the specified view if self.full_vol == False:
        if (not self.full_vol):
            if view == self.views[0]:
                img = img[:, :, slice_]
                mask = mask[:, :, slice_]

            elif view == self.views[1]:
                img = img[:, slice_, :]
                mask = mask[:, slice_, :]

            elif view == self.views[2]:
                img = img[slice_, :, :]
                mask = mask[slice_, :, :]
        
        img = torch.clamp(img, -1000, 400)
        
        if (len(img.shape) < 3):
            img = img.unsqueeze(0)
            mask = mask.unsqueeze(0)
        
        assert img.shape == mask.shape, "Image shape is not equal to mask shape!"
        
        # Rescalling pixel values to range [0, 1].
        # Used only if original ViT mean and std are used for Z-score norm.
        #img -= -1000
        #img = img/1400
        
        if self.apply_mask:
            img = img*mask
        
        # As ViT model requires 3 color channels,
        # code below makes 2 more channels by coping original channel.
        if self.finetuning:
            img = img.repeat(3,1,1)
        
        # If some image transformations are specified:
        if self.transform is not None:
            img = self.transform(img)
        
        return img.float(), mask.float()
    
    
    def __getitem__(self, idx):
        if self.labels == 'targets':
            label = self.targets[idx]
        elif self.labels == 'concepts':
            concepts1 = self.concepts[idx]
            if (self.label_transform is not None):
                scaler = self.label_transform
                concepts1 = scaler.transform(np.expand_dims(concepts1, axis=0))[0]
            label = torch.tensor(concepts1).float()
        else:
            print("Labels may be only: 1) targets, 2) concepts. NOTHING ELSE!")
        
        # For training dataset each nodule is representent by one view: axial, coronal, sagittal.
        if self.train_mode:
            view = random.choice(self.views)
            slices = np.linspace(14, 18, 5).astype(int)
            slice_ = random.choice(slices)
            img, mask = self.process_image(nodule_idx=idx, view=view, slice_=slice_)
            if self.return_mask:
                return [img, label, mask]
            else:
                return [img, label]
            
        else:
            # for testing dataset I take all views of a nodule
            images = []
            masks = []
            for view in self.views:
                img, mask = self.process_image(nodule_idx=idx, view=view, slice_=16)
                images.append(img)
                masks.append(mask)
            if self.return_mask:
                return [images, label, masks]
            else:
                return [images, label]
    
    
    def get_target(self, idx):
        target = self.targets[idx]
        return target