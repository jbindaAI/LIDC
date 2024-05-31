import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from LIDC_Dataset_biom import LIDC_Dataset_biom
from Biomarker_Model import Biomarker_Model
import pickle

## HYPERPARAMETERS:
TRAINABLE_LAYERS = 60
LR = 3e-4
LR_DECAY_RATE = 0.95
DROPOUT = 0.003
EPOCHS = 50
BATCH_SIZE = 32
MODEL_NR = 15
LOCAL = True

if LOCAL:
    data_path="/home/jbinda/INFORM/LIDC/dataset/"
    tb_logs_path="/home/jbinda/INFORM/LIDC/DINO/tb_logs/Biomarkers/"
    checkpoints_path = "/home/jbinda/INFORM/LIDC/DINO/checkpoints/Biomarkers/"
else:
    data_path="/home/dzban112/LIDC/dataset/"
    tb_logs_path="/home/dzban112/LIDC/DINO/tb_logs/Biomarkers/"
    checkpoints_path = "/home/dzban112/LIDC/DINO/checkpoints/Biomarkers/"

with open(data_path+"splitted_sets"+"/"+"fitted_mean_std.pkl", 'rb') as f:
    dict_ = pickle.load(f)
with open(data_path+"splitted_sets"+"/"+"scaler.pkl", 'rb') as f:
    SCALER = pickle.load(f)
    
MEAN = dict_["mean"]
STD = dict_["std"]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    MyRotation([0, 90, 180, 270]),
    transforms.Normalize(mean=MEAN, std=STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=MEAN, std=STD),
])

# add a checkpoint callback that saves the model with the lowest validation loss
checkpoint_name = f"best-checkpoint_{MODEL_NR}"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoints_path,
    filename=checkpoint_name,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

ds_train = LIDC_Dataset_biom(
                datadir=data_path,
                transform=train_transform,
                label_transform=SCALER,
                mode="train"
            )

ds_val = LIDC_Dataset_biom(
                datadir=data_path,
                transform=val_transform,
                label_transform=SCALER,
                mode="val"
            )

train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
val_loader = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=BATCH_SIZE, num_workers=8)

torch.set_float32_matmul_precision('medium')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(accelerator="gpu", devices=1, 
                     precision="16-mixed", max_epochs=EPOCHS,
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=TensorBoardLogger(tb_logs_path, name=f"ViT_biom_{MODEL_NR}"),
                     log_every_n_steps=20
                    )
model = Biomarker_Model(trainable_layers=TRAINABLE_LAYERS, dropout=DROPOUT, lr_rate=LR, lr_decay_rate=LR_DECAY_RATE)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)