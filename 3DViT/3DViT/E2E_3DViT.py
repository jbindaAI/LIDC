import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
# 3DViT class:
from dino3d import VisionTransformer3D
from functools import partial

def set_dropout_p(module, dropout_p):
    if isinstance(module, nn.Dropout):
        # Sets dropout probability for dropout layers
        module.p = dropout_p


class E2E_3DViT(pl.LightningModule):
    def __init__(self,
                 trainable_layers=0,
                 dropout=0.0,
                 lr_rate=4e-5,
                 max_lr=7e-4,
                 img_size=(7, 224, 224), # (volume depth, height, width)
                 patch_size=8,
                 bootstrap_method="centering",
                 epochs=10,
                 steps_per_epoch=58,
                 train_emb_layer=False,
                 pct_start=0.15
                ):
        super().__init__()
        self.dropout = dropout
        self.lr_rate = lr_rate
        self.max_lr = max_lr
        self.save_hyperparameters()
        self.epochs=epochs
        self.steps_per_epoch=steps_per_epoch
        self.train_emb_layer=train_emb_layer
        self.pct_start=pct_start
        self.dino = VisionTransformer3D(
            img_size=img_size,
            patch_size=(patch_size, patch_size, img_size[0]),
            embed_dim=384,
            depth=12,
            in_chans=1, # grayscale
            num_heads=6, # attention heads
            mlp_ratio=4, # dimensionality increse in MLP layers
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        # changing dropout values:
        if dropout > 0.0:
            self.dino.apply(lambda module: set_dropout_p(module, dropout_p=self.dropout))
        
        all_layers = len(list(self.dino.parameters()))
        for i, p in enumerate(self.dino.parameters()):
            if i < (all_layers - trainable_layers):
                p.requires_grad = False
                
        if self.train_emb_layer:
            for name, param in self.dino.named_parameters():
                k = 0
                if name == "patch_embed.proj.weight" or name == "patch_embed.proj.bias":
                    #print(name)
                    param.requires_grad = True
                    k+=1
                if k == 2:
                    break
        
        self.linear = nn.Linear(384, 1)
        self.accuracy = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        x = self.dino(x)
        x = self.linear(x)
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
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                           max_lr=self.max_lr, 
                                                           steps_per_epoch=self.steps_per_epoch, 
                                                           epochs=self.epochs,
                                                           pct_start=self.pct_start
                                                          )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]