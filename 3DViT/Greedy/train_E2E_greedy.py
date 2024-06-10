import torch
from torchvision import datasets
from torchvision.transforms import v2
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from E2E_greedy_dataset import Greedy_Dataset
from E2E_greedy import Greedy_E2E
import pickle
import math
import wandb

wandb_logger = WandbLogger(project='Greedy_E2E', name="Trial run", job_type='train')

## HYPERPARAMETERS:
TRAINABLE_LAYERS = 10 # DINO layers, final linear layer is always trained! In particular, when TRAINABLE_LAYERS==0.
LR = 3e-5
MAX_LR = 8e-4
PCT_START = 0.2
DROPOUT = 0.1
EPOCHS = 10
BATCH_SIZE = 32
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
    "pct_start": PCT_START
})

if LOCAL:
    data_path="/home/jbinda/INFORM/LIDC/3DViT/Greedy/data/"
    checkpoints_path = "/home/jbinda/INFORM/LIDC/3DViT/Greedy/ckpt/End2End/"
else:
    data_path="SOMEDAY"

with open(data_path+"greedy_norm_fact.pkl", 'rb') as f:
    dict_ = pickle.load(f)
    
MEAN = dict_["MEAN"]
STD = dict_["STD"]


train_transform = v2.Compose([
    v2.Resize(224),
    MyRotation([0, 90, 180, 270]),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.Normalize(mean=[MEAN], std=[STD]),
])

val_transform = v2.Compose([
    v2.Resize(224),
    v2.Normalize(mean=[MEAN], std=[STD]),
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

ds_train = Greedy_Dataset(
                datadir=data_path,
                transform=train_transform,
                mode="train"
            )

ds_val = Greedy_Dataset(
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

model = Greedy_E2E(trainable_layers=TRAINABLE_LAYERS, 
                    dropout=DROPOUT, 
                    lr_rate=LR,
                    max_lr=MAX_LR,
                    epochs=EPOCHS,
                    steps_per_epoch=math.ceil(len(ds_train)/BATCH_SIZE),
                    pct_start=PCT_START
                  )

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)