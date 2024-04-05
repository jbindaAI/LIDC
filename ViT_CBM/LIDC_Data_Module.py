# packages
import pytorch_lightning as pl
import torch
from LIDC_Dataset import LIDC_Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from torchvision import transforms
from util.MyRotation import MyRotation
from torch.utils.data import DataLoader
import numpy as np


####### Ligthning Data Module CLASS

class LIDC_Data_Module(pl.LightningDataModule):
    def __init__(self,
                 data_dir="dataset",
                 fold=0,
                 apply_mask=False,
                 batch_size=32,
                 num_workers=8,
                 labels="targets",
                 original_norm=False,
                 return_mask=False,
                 full_vol=False,
                 extract=False,
                 finetuning=False
                ):
        super().__init__()
        self.data_dir = data_dir
        self.fold = fold
        self.apply_mask = apply_mask
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.labels = labels
        self.original_norm = original_norm
        self.finetuning = finetuning
        self.full_vol = full_vol
        self.extract = extract
        self.return_mask = return_mask        
        
    def setup(self, stage=None):
        # full dataset class initialization:
        full_dataset = LIDC_Dataset(data_dir=self.data_dir, 
                                    train_mode=False,
                                    labels=self.labels,
                                    finetuning=self.finetuning
                                   )
        
        # number of examples in the full dataset and their idx:
        num_full = len(full_dataset)
        indices_full = list(range(num_full))
        
        # all labels in the full dataset:
        all_labels = np.array([full_dataset.get_target(i) for i in range(num_full)])
        
        # Using K-Fold Stratified Cross Validation
        # to split dataset into K=5 folds.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        gen_splits = skf.split(indices_full, all_labels)
        
        # Creating lists with idx for train and test folds:
        train_idx_folds = []
        test_idx_folds = []
        for train_idx, test_idx in gen_splits:
            # gen_splits object is iterable object of folds
            # consisted of pairs: train_idx, test_idx
            train_idx_folds.append(train_idx)
            test_idx_folds.append(test_idx)
        
        # choosing one fold:
        train_idx = train_idx_folds[self.fold]
        test_idx = test_idx_folds[self.fold]
        
        if (not self.original_norm):
            # There are two options. Z-Score Normalization of data by manually fitted
            # standard scaler. Or using mean, std on which ViT model was originally
            # trained. Second option requires rescalling voxel values to [0,1] range in
            # LIDC_Dataset.py file.
            # Below fitting standard scaler:
            train_imgs = []
            train_concepts = []
            for idx in train_idx:
                image = full_dataset[idx][0][0]
                train_imgs.append(image)
                if (self.labels == "concepts"):
                    concepts = full_dataset[idx][1]
                    train_concepts.append(concepts)
            scaler = None
            if (self.labels == "concepts"):
                train_concepts = np.stack(train_concepts, axis=0)
                scaler = preprocessing.StandardScaler().fit(train_concepts)
                
            train_imgs = torch.stack(train_imgs, axis=0)
            # imgs have 3 channels but there are the same values.
            # So one mean and std value is enough and each channel
            # of image is normalized by the same value.
            channels_mean = torch.mean(train_imgs)
            channels_std = torch.std(train_imgs)

        else:
            # Below second option for voxels values normalization.
            # I am using mean and std values for data on which ViT_b16 was
            # originally trained. More details in PyTorch ViT_b16 Docs.
            channels_mean = [0.485, 0.456, 0.406]
            channels_std = [0.229, 0.224, 0.225]
        
        train_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=channels_mean, std=channels_std),
                MyRotation([0, 90, 180, 270])
            ]
        )
        
        test_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.Normalize(mean=channels_mean, std=channels_std)
            ]
        )
        
        if (self.extract == True): # it means features extraction from the last ViT layer in bootleneck model
            train_mode = False
        else:
            train_mode = True
        
        train = LIDC_Dataset(
            data_dir=self.data_dir,
            train_mode=train_mode,
            transform=train_transform,
            label_transform=scaler,
            return_mask=self.return_mask,
            apply_mask=self.apply_mask,
            full_vol=self.full_vol,
            finetuning=self.finetuning,
            labels=self.labels
        )
        
        test = LIDC_Dataset(
            data_dir=self.data_dir,
            train_mode=False,
            transform=test_transform,
            label_transform=scaler,
            apply_mask=self.apply_mask,
            full_vol=self.full_vol,
            labels=self.labels,
            finetuning=self.finetuning
        )
        
        self.train_data = torch.utils.data.Subset(train, train_idx)
        self.val_data = torch.utils.data.Subset(test, test_idx)
        self.test_data = torch.utils.data.Subset(test, test_idx)

    def train_dataloader(self):
        if (self.extract == True):
            shuffle=False
        else:
            shuffle=True
        
        train_loader = DataLoader(self.train_data,
                                  batch_size=self.batch_size,
                                  shuffle=shuffle,
                                  num_workers=self.num_workers
                                 )
        return train_loader
    

    def val_dataloader(self):
        val_loader = DataLoader(self.val_data,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers
                               )
        return val_loader
    
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                 num_workers=self.num_workers
                                )
        return test_loader