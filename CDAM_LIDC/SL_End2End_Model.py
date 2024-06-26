# from dino_trunc import dino_trunc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision import models

def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.TransformerEncoderLayer):
        # Traverse the encoder layer to find dropout layers
        for child_name, child_module in module.named_children():
            print(child_name)
            if isinstance(child_module, nn.Dropout):
                # Sets dropout probability for dropout layers within encoder blocks
                child_module.p = dropout_p


def create_model(trainable_layers, patch_size=32):
    if patch_size==32:
        model = models.vit_b_32(weights='IMAGENET1K_V1')
    elif patch_size==16:
        model = models.vit_b_16(weights='IMAGENET1K_V1')
    else:
        raise Exception("Only patch_size=32 or patch_size=16!")
    num_layers_param = len(list(model.parameters()))
    for i, param in enumerate(model.parameters()):
        # i >= (num_layers_param - train_layers) => requires grad
         if (i < num_layers_param - trainable_layers):
            param.requires_grad = False
    model.heads = nn.Linear(768, 1)
    return model


class SL_End2End_Model(pl.LightningModule):
    def __init__(self, trainable_layers=0, dropout=0.0, lr_rate=3e-4, patch_size=32):
        super().__init__()
        self.dropout = dropout
        self.lr_rate = lr_rate
        self.model = create_model(trainable_layers, patch_size)
        # changing dropout values:
        if dropout > 0.0:       
            self.model.apply(lambda module: set_encoder_dropout_p(module, dropout_p=self.dropout))
        
        all_layers = len(list(self.model.parameters()))
        for i, p in enumerate(self.model.parameters()):
            if i < (all_layers - trainable_layers):
                p.requires_grad = False    
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.float()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("train_acc", self.accuracy(y_hat, y), prog_bar=False, on_epoch=True)
        self.log("train_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        if len(y_hat.size()) == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        y = y.float()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        self.log("val_acc", self.accuracy(y_hat, y), prog_bar=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        if len(y_hat.size()) == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        y = y.float()
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        self.log("test_acc", self.accuracy(y_hat, y), prog_bar=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
        return [optimizer], [lr_scheduler]