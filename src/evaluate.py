import argparse
import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import F1Score, AveragePrecision, JaccardIndex, Precision, Recall, ConfusionMatrix

from data_module import DualStreamPatchDataset, crop_to_multiple
from models.asku_net import ASKULightningModule

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.model} model from {args.ckpt_path}...")
    
    if args.model == "asku_net":
        model = ASKULightningModule.load_from_checkpoint(args.ckpt_path)
    # Add other models here as needed
    else:
        raise ValueError("Model not supported for evaluation yet.")
        
    model = model.to(device)
    model.eval()

    print("Loading evaluation data...")
    ds = xr.open_zarr(args.data_path, consolidated=True)
    
    vv_tensor = crop_to_multiple(torch.tensor(ds['vv'][-12:].fillna(0).values, dtype=torch.float32), 256)
    vh_tensor = crop_to_multiple(torch.tensor(ds['vh'][-12:].fillna(0).values, dtype=torch.float32), 256)
    dem_tensor = crop_to_multiple(torch.tensor(ds['dem'].fillna(0).values, dtype=torch.float32), 256)
    label_tensor = crop_to_multiple(torch.tensor(np.nan_to_num(ds['landslides'].compute().data, nan=0.0), dtype=torch.long), 256)

    eval_dataset = DualStreamPatchDataset(
        vv_tensor=vv_tensor, vh_tensor=vh_tensor, dem_tensor=dem_tensor, label_tensor=label_tensor,
        patch_size=256, stride=256, min_percent=0, transform=None
    )
    eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=False, num_workers=2)

    # Initialize metrics
    f1_metric = F1Score(task="multiclass", num_classes=2, average="macro").to(device)
    auprc_metric = AveragePrecision(task="multiclass", num_classes=2).to(device)
    iou_metric = JaccardIndex(task="multiclass", num_classes=2).to(device)
    precision_metric = Precision(task="multiclass", num_classes=2).to(device)
    recall_metric = Recall(task="multiclass", num_classes=2).to(device)
    cm_metric = ConfusionMatrix(task="multiclass", num_classes=2).to(device)

    print("Evaluating...")
    with torch.no_grad():
        for x, y in eval_loader:
            x, y = x.to(device), y.squeeze(1).to(device)
            logits = model(x)
            logits = F.interpolate(logits, size=y.shape[1:], mode='bilinear', align_corners=False)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, 2)
            y_flat = y.reshape(-1)

            f1_metric.update(probs_flat, y_flat)
            auprc_metric.update(probs_flat, y_flat)
            iou_metric.update(probs_flat, y_flat)
            precision_metric.update(probs_flat, y_flat)
            recall_metric.update(probs_flat, y_flat)
            cm_metric.update(preds, y)

    print("\n📊 Evaluation Results:")
    print(f"🔹 F1 Score:     {f1_metric.compute().item():.4f}")
    print(f"🔹 AUPRC:        {auprc_metric.compute().item():.4f}")
    print(f"🔹 IoU:          {iou_metric.compute().item():.4f}")
    print(f"🔹 Precision:    {precision_metric.compute().item():.4f}")
    print(f"🔹 Recall:       {recall_metric.compute().item():.4f}")
    
    print("\n📝 Confusion Matrix:")
    print(cm_metric.compute().cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Landslide Segmentation Models")
    parser.add_argument("--model", type=str, default="asku_net", help="Model type to load")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--data_path", type=str, default="data/talakmau_indonesia", help="Path to evaluation Zarr dataset")
    
    args = parser.parse_args()
    evaluate_model(args)
