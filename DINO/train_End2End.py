import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from End2End_Model import End2End_Model
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from my_utils.MyRotation import MyRotation
from LIDC_Dataset import LIDC_Dataset
from End2End_Model import End2End_Model


data_path="/home/dzban112/LIDC/dataset/"
tb_logs_path="/home/dzban112/LIDC/DINO/tb_logs/End2End/"
checkpoints_path = "/home/dzban112/LIDC/DINO/checkpoints/"

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    MyRotation([0, 90, 180, 270]),
    transforms.Normalize(mean=-632.900, std=358.664),
])

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=-632.900, std=358.664),
])

# add a checkpoint callback that saves the model with the lowest validation loss
checkpoint_name = "best-checkpoint_7"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath=checkpoints_path,
    filename=checkpoint_name,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

ds_train = LIDC_Dataset(
                datadir=data_path,
                transform=train_transform,
                with_gen_imgs=False,
                mode="train"
            )

ds_val = LIDC_Dataset(
                datadir=data_path,
                transform=val_transform,
                with_gen_imgs=False,
                mode="val"
            )

train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=32, num_workers=8)
val_loader = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=32, num_workers=8)

torch.set_float32_matmul_precision('medium')
lr_monitor = LearningRateMonitor(logging_interval='step')
trainer = pl.Trainer(accelerator="gpu", devices=1, 
                     precision="16-mixed", max_epochs=50,
                     callbacks=[checkpoint_callback, lr_monitor],
                     logger=TensorBoardLogger(tb_logs_path, name=f"ViT_E2E_7")
                    )
model = End2End_Model(trainable_layers=50, dropout=0.08, lr_rate=3e-4)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# lightning deepspeed has saved a directory instead of a file
save_path = f"{checkpoint_name}.ckpt"
output_path = f"{checkpoint_name}.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)