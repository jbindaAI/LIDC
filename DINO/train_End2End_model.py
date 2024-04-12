import torch
from torchvision import datasets, transforms
import pytorch_lightning as pl
from End2End_Model import End2End_Model
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    MyRotation([0, 90, 180, 270]),
    transforms.Normalize(mean=-623.539, std=365.112),
])

val_transform = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=-623.539, std=365.112),
])

# add a checkpoint callback that saves the model with the lowest validation loss
checkpoint_name = "best-checkpoint-full-imgnet-augment-new"
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="checkpoints",
    filename=checkpoint_name,
    save_top_k=1,
    monitor="val_loss",
    mode="min",
)

ds_train = LIDC_Dataset(
                datadir="/home/jbinda/INFORM/LIDC/dataset/",
                transform=train_transform,
                with_gen_imgs=False,
                mode="train"
            )

ds_val = LIDC_Dataset(
                datadir="/home/jbinda/INFORM/LIDC/dataset/",
                transform=val_transform,
                with_gen_imgs=False,
                mode="val"
            )

train_loader = torch.utils.data.DataLoader(ds_train, shuffle=True, batch_size=32, num_workers=8)
val_loader = torch.utils.data.DataLoader(ds_val, shuffle=False, batch_size=bs, num_workers=8)

torch.set_float32_matmul_precision('medium')
trainer = pl.Trainer(accelerator="gpu", devices=2, 
                     precision="16-mixed", max_epochs=30, 
                     strategy="ddp", callbacks=[checkpoint_callback],
                     logger=TensorBoardLogger(f"/home/jbinda/INFORM/LIDC/DINO/End2End/", name=f"ViT_E2E_1")
                    )
model = LitModel(trainable_layers=5)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# lightning deepspeed has saved a directory instead of a file
save_path = f"checkpoints/{checkpoint_name}.ckpt"
output_path = f"{checkpoint_name}.ckpt"
convert_zero_checkpoint_to_fp32_state_dict(save_path, output_path)