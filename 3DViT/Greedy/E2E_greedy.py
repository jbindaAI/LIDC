import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

def set_dropout_p(module, dropout_p):
    if isinstance(module, nn.Dropout):
        # Sets dropout probability for dropout layers
        module.p = dropout_p


class Greedy_E2E(pl.LightningModule):
    def __init__(self,
                 trainable_layers=0,
                 dropout=0.0,
                 lr_rate=4e-5,
                 max_lr=7e-4,
                 epochs=10,
                 steps_per_epoch=58,
                 pct_start=0.15
                ):
        super().__init__()
        self.dropout = dropout
        self.lr_rate = lr_rate
        self.max_lr = max_lr
        self.save_hyperparameters()
        self.epochs=epochs
        self.steps_per_epoch=steps_per_epoch
        self.pct_start=pct_start
        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        # changing dropout values:
        if dropout > 0.0:
            self.dino.apply(lambda module: set_dropout_p(module, dropout_p=self.dropout))
        
        all_layers = len(list(self.dino.parameters()))
        for i, p in enumerate(self.dino.parameters()):
            if i < (all_layers - trainable_layers):
                p.requires_grad = False    
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
        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                           max_lr=self.max_lr, 
                                                           steps_per_epoch=self.steps_per_epoch, 
                                                           epochs=self.epochs,
                                                           pct_start=self.pct_start
                                                          )
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]