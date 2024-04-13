# from dino_trunc import dino_trunc
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

def set_encoder_dropout_p(module, dropout_p):
    if isinstance(module, nn.TransformerEncoderLayer):
        # Traverse the encoder layer to find dropout layers
        for child_name, child_module in module.named_children():
            print(child_name)
            if isinstance(child_module, nn.Dropout):
                # Sets dropout probability for dropout layers within encoder blocks
                child_module.p = dropout_p


class End2End_Model(pl.LightningModule):
    def __init__(self, trainable_layers=0, dropout=0.0, lr_rate=3e-4):
        super().__init__()
        self.dropout = dropout
        self.lr_rate = lr_rate
        self.save_hyperparameters()
        # self.model = dino_trunc()
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits8")
        # changing dropout values:
        if dropout > 0.0:       
            self.model.apply(lambda module: set_encoder_dropout_p(module, dropout_p=self.dropout))
        
        all_layers = len(list(self.model.parameters()))
        for i, p in enumerate(self.model.parameters()):
            if i < (all_layers - trainable_layers):
                p.requires_grad = False    
        self.linear = nn.Linear(384, 8)

    def forward(self, x):
        x = self.model(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        y = y.float()
        loss = F.Huber_Loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        if len(y_hat.size()) == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        y = y.float()
        loss = F.Huber_Loss(y_hat, y)
        
        self.log("val_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        if len(y_hat.size()) == 0:
            y_hat = y_hat.unsqueeze(dim=0)
        y = y.float()
        loss = F.Huber_Loss(y_hat, y)
        
        self.log("test_loss", loss, prog_bar=False, sync_dist=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_rate)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
        return [optimizer], [lr_scheduler]