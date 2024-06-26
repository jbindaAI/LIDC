import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from LIDC_Dataset_E2E import LIDC_Dataset_E2E
from SL_End2End_Model import SL_End2End_Model
import math
import pickle


wandb_logger = WandbLogger(project='ViT_E2E', name="Trial run", job_type='train')


## HYPERPARAMETERS:
PATCH_SIZE = 32
TRAINABLE_LAYERS = 0
FREEZE = False
LR = 3e-9
MAX_LR = 8e-6
DROPOUT = 0.1
EPOCHS = 50
BATCH_SIZE = 16
PCT_START = 0.5
MODEL_NR = 22
LOCAL = True


wandb_logger.experiment.config.update({
    "model_nr": MODEL_NR,
    "trainable_layers": TRAINABLE_LAYERS, 
    "freeze": FREEZE,
    "lr": LR,
    "max_lr": MAX_LR,
    "dropout": DROPOUT,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "pct_start": PCT_START
})


if LOCAL:
    data_path="/home/jbinda/INFORM/LIDC/dataset/"
    checkpoints_path = f"/home/jbinda/INFORM/LIDC/ViT/checkpoints/End2End/p{PATCH_SIZE}"
else:
    data_path="/home/dzban112/LIDC/dataset/"
    checkpoints_path = f"/home/dzban112/LIDC/ViT/checkpoints/End2End/p{PATCH_SIZE}"

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

ds_train = LIDC_Dataset_E2E(
                datadir=data_path,
                transform=train_transform,
                with_gen_imgs=False,
                mode="train"
            )

ds_val = LIDC_Dataset_E2E(
                datadir=data_path,
                transform=val_transform,
                with_gen_imgs=False,
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

model = SL_End2End_Model(trainable_layers=TRAINABLE_LAYERS,
                         freeze=FREEZE,
                         dropout=DROPOUT, 
                         lr_rate=LR,
                         max_lr=MAX_LR,
                         pct_start=PCT_START,
                         epochs=EPOCHS,
                         patch_size=PATCH_SIZE,
                         steps_per_epoch=math.ceil(len(ds_train)/BATCH_SIZE)
                        )

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)