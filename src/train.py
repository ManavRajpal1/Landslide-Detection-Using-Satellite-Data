import argparse
import xarray as xr
import numpy as np
import torch
import pytorch_lightning as pl

# Import your data module
from data_module import DualStreamDataModule, crop_to_multiple

# Import your model wrappers
from models.asku_net import ASKULightningModule
# from models.unet import UNetLightningModule  <-- Add others as you build them

def main(args):
    print("Loading data...")
    ds = xr.open_zarr(args.data_path, consolidated=True)
    
    vv_tensor = crop_to_multiple(torch.tensor(ds['vv'][-12:].fillna(0).values, dtype=torch.float32), 256)
    vh_tensor = crop_to_multiple(torch.tensor(ds['vh'][-12:].fillna(0).values, dtype=torch.float32), 256)
    dem_tensor = crop_to_multiple(torch.tensor(ds['dem'].fillna(0).values, dtype=torch.float32), 256)
    label_tensor = crop_to_multiple(torch.tensor(np.nan_to_num(ds['landslides'].compute().data, nan=0.0), dtype=torch.long), 256)

    print("Initializing DataModule...")
    data_module = DualStreamDataModule(
        vv_tensor=vv_tensor, 
        vh_tensor=vh_tensor, 
        dem_tensor=dem_tensor, 
        label_tensor=label_tensor,
        batch_size=args.batch_size
    )

    print(f"Initializing Model: {args.model}...")
    if args.model == "asku_net":
        model = ASKULightningModule(in_channels=3, num_classes=2, deep_supervision=True)
    # elif args.model == "unet":
    #     model = UNetLightningModule(in_channels=3, num_classes=2)
    else:
        raise ValueError(f"Model {args.model} not supported yet.")

    print("Starting training...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=10,
        enable_checkpointing=True,
        accelerator='auto'
    )
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Landslide Segmentation Models")
    parser.add_argument("--model", type=str, default="asku_net", help="Model to train (e.g., asku_net, unet)")
    parser.add_argument("--data_path", type=str, default="data/talakmau_indonesia", help="Path to Zarr dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    
    args = parser.parse_args()
    main(args)
