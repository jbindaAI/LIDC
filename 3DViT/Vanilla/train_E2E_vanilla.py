import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from E2E_vanilla_dataset import Vanilla_Dataset
from E2E_vanilla import Vanilla_E2E
import pickle
import wandb

wandb_logger = WandbLogger(project='Vanilla_E2E', name="Trial run", job_type='train')

## HYPERPARAMETERS:
TRAINABLE_LAYERS = 10
LR = 3e-4
DROPOUT = 0.05
EPOCHS = 10
BATCH_SIZE = 32
MODEL_NR = 1
LOCAL = True

wandb_logger.experiment.config.update({"trainable_layers": TRAINABLE_LAYERS, 
                                       "lr": LR,
                                       "dropout": DROPOUT,
                                       "epochs": EPOCHS,
                                       "batch_size": BATCH_SIZE
                                      })

if LOCAL:
    data_path="/home/jbinda/INFORM/LIDC/3DViT/Vanilla/data/"
    checkpoints_path = "/home/jbinda/INFORM/LIDC/3DViT/Vanilla/ckpt/End2End/"
else:
    data_path="/home/dzban112/LIDC/dataset/"

with open(data_path+"vanilla_norm_fact.pkl", 'rb') as f:
    dict_ = pickle.load(f)
    
MEAN = dict_["MEAN"]
STD = dict_["STD"]


train_transform = transforms.Compose([
    transforms.Resize(224),
    MyRotation([0, 90, 180, 270]),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize(mean=MEAN, std=STD),
])

# add a checkpoint callback that saves the model with the lowest validation loss
checkpoint_name = f"checkpoint_{MODEL_NR}"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoints_path,
    filename=checkpoint_name,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

ds_train = Vanilla_Dataset(
                datadir=data_path,
                transform=train_transform,
                mode="train"
            )

ds_val = Vanilla_Dataset(
                datadir=data_path,
                transform=val_transform,
                mode="val"
            )

train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
val_loader = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)

torch.set_float32_matmul_precision('medium')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(accelerator="gpu", devices=1, 
                     precision="16-mixed", max_epochs=EPOCHS,
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=wandb_logger
                    )
model = Vanilla_E2E(trainable_layers=TRAINABLE_LAYERS, dropout=DROPOUT, lr_rate=LR)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)