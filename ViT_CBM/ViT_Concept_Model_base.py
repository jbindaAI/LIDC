# Model used for prediction nodule features like: spiculation, calcification, diameter
# and so on.

# packages
from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
import random

from torch.optim.lr_scheduler import MultiStepLR
import torchvision.models as models




def create_model(freeze_layers, train_layers, patch_size=32):
    if patch_size==32:
        model = models.vit_b_32(weights='IMAGENET1K_V1')
    elif patch_size==16:
        model = models.vit_b_16(weights='IMAGENET1K_V1')
    else:
        raise Exception("Only patch_size=32 or patch_size=16!")
    if freeze_layers:
        num_layers_param = len(list(model.parameters()))
        for i, param in enumerate(model.parameters()):
            # i >= (num_layers_param - train_layers) => requires grad
            if (i < num_layers_param - train_layers):
                param.requires_grad = False
    model.heads = nn.Linear(768, 8) # = nn.Linear(768, 8)
    return model


class ViT_Concept_Model_base(LightningModule):
    def __init__(self, 
                 learning_rate=1e-3, 
                 weight_decay=1e-4,
                 huber_delta=0.5,
                 train_layers=10,
                 freeze_layers=True,
                 patch_size=32
                ):
        
        super().__init__()
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.HuberLoss(delta=huber_delta)
        self.train_layers = train_layers
        self.freeze_layers = freeze_layers
        self.patch_size = patch_size
        self.model = create_model(self.freeze_layers, self.train_layers, self.patch_size)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_views = torch.zeros((3, y.shape[0], 8))
        logits_views = logits_views.type_as(x[0])
        
        for i in range(3):
            logits_views[i] = self(x[i])
        logits = torch.mean(logits_views, axis=0)
        
        loss = self.criterion(logits, y)
        
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        lr_scheduler = {
            'scheduler': MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1),
            'monitor': 'val_loss',
            'name': 'log_lr'
        }
        return [optimizer], [lr_scheduler]