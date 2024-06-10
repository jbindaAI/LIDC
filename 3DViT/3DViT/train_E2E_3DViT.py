import torch
from torchvision import datasets
from torchvision.transforms import v2
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from E2E_3DViT_dataset import ViT_3D_Dataset
from E2E_3DViT import E2E_3DViT
import pickle
import math
import wandb
from typing import Literal

wandb_logger = WandbLogger(project='3DViT_E2E', name="Third cent run", job_type='train')

## HYPERPARAMETERS:
TRAINABLE_LAYERS = 15
LR = 3e-5
MAX_LR = 8e-4
DROPOUT = 0.1
EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = (7,224,224)
BOOTSTRAP_METHOD: Literal["centering", "inflation"]= "centering"
TRAIN_EMBEDDINGS = True
PCT_START = 0.2
MODEL_NR = 3
LOCAL = True

wandb_logger.experiment.config.update({
    "model_nr": MODEL_NR,
    "trainable_layers": TRAINABLE_LAYERS, 
    "lr": LR,
    "max_lr": MAX_LR,
    "dropout": DROPOUT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "bootstrap_method": BOOTSTRAP_METHOD,
    "train_embeddings": TRAIN_EMBEDDINGS,
    "pct_start": PCT_START
})

if LOCAL:
    data_path="/home/jbinda/INFORM/LIDC/3DViT/3DViT/data/"
    checkpoints_path = "/home/jbinda/INFORM/LIDC/3DViT/3DViT/ckpt/End2End/"
else:
    data_path="SOMEDAY..."

with open(data_path+"3dvit_norm_fact.pkl", 'rb') as f:
    dict_ = pickle.load(f)
    
MEAN = dict_["MEAN"]
STD = dict_["STD"]


train_transform = v2.Compose([
    v2.Resize(224),
    MyRotation([0, 90, 180, 270]),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.Normalize(mean=MEAN, std=STD),
])

val_transform = v2.Compose([
    v2.Resize(224),
    v2.Normalize(mean=MEAN, std=STD),
])

# add a checkpoint callback that saves the model with the lowest validation loss
checkpoint_name = f"checkpoint_{BOOTSTRAP_METHOD}_{MODEL_NR}"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoints_path,
    filename=checkpoint_name,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

ds_train = ViT_3D_Dataset(
                datadir=data_path,
                transform=train_transform,
                mode="train"
            )

ds_val = ViT_3D_Dataset(
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
                     logger=wandb_logger,
                     log_every_n_steps=math.ceil(len(ds_train)/BATCH_SIZE)
                    )
model = E2E_3DViT(
    trainable_layers=TRAINABLE_LAYERS,
    dropout=DROPOUT,
    lr_rate=LR,
    max_lr=MAX_LR,
    img_size=IMG_SIZE,
    patch_size=8,
    bootstrap_method=BOOTSTRAP_METHOD,
    epochs=EPOCHS,
    steps_per_epoch=math.ceil(len(ds_train)/BATCH_SIZE),
    train_emb_layer=TRAIN_EMBEDDINGS,
    pct_start=PCT_START
)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)