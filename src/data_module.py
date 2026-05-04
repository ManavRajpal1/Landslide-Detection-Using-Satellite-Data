import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import albumentations as A

def crop_to_multiple(tensor, multiple=256):
    """Crops tensors so their spatial dimensions are multiples of a given number."""
    if tensor.ndim == 3:
        t, h, w = tensor.shape
        return tensor[:, : (h // multiple) * multiple, : (w // multiple) * multiple]
    elif tensor.ndim == 2:
        h, w = tensor.shape
        return tensor[: (h // multiple) * multiple, : (w // multiple) * multiple]

# Data augmentation pipeline
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
], additional_targets={'dem': 'image', 'mask': 'mask'})

class DualStreamPatchDataset(Dataset):
    def __init__(self, vv_tensor, vh_tensor, dem_tensor, label_tensor,
                 patch_size=256, stride=256, min_percent=0.01, transform=train_transforms):
        self.vv = vv_tensor
        self.vh = vh_tensor
        self.dem = dem_tensor
        self.label = label_tensor
        self.T = vv_tensor.shape[0]
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.coords = []

        H, W = self.label.shape
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                label_patch = self.label[y:y + patch_size, x:x + patch_size]
                percent = (label_patch > 0).float().mean().item() * 100
                if percent >= min_percent:
                    self.coords.append((y, x))
        self.total_samples = len(self.coords) * self.T

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        coord_idx = idx // self.T
        t = idx % self.T
        y, x = self.coords[coord_idx]

        vv_patch = self.vv[t, y:y+self.patch_size, x:x+self.patch_size]
        vh_patch = self.vh[t, y:y+self.patch_size, x:x+self.patch_size]
        dem_patch = self.dem[y:y+self.patch_size, x:x+self.patch_size]
        
        # Stack DEM, VV, VH
        input_patch = torch.stack([dem_patch, vv_patch, vh_patch], dim=0)
        label_patch = self.label[y:y+self.patch_size, x:x+self.patch_size].unsqueeze(0)

        if self.transform:
            augmented = self.transform(
                image=input_patch.permute(1, 2, 0).numpy(),
                mask=label_patch.squeeze(0).numpy()
            )
            input_patch = torch.tensor(augmented['image']).permute(2, 0, 1).float()
            label_patch = torch.tensor(augmented['mask']).unsqueeze(0).long()

        return input_patch, label_patch

class DualStreamDataModule(pl.LightningDataModule):
    def __init__(self, vv_tensor, vh_tensor, dem_tensor, label_tensor,
                 batch_size=4, patch_size=256, stride=128, min_percent=5, transform=train_transforms):
        super().__init__()
        self.vv = vv_tensor
        self.vh = vh_tensor
        self.dem = dem_tensor
        self.label = label_tensor
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride
        self.min_percent = min_percent
        self.transform = transform

    def setup(self, stage=None):
        full_dataset = DualStreamPatchDataset(
            vv_tensor=self.vv, vh_tensor=self.vh, dem_tensor=self.dem,
            label_tensor=self.label, patch_size=self.patch_size,
            stride=self.stride, min_percent=self.min_percent, transform=self.transform
        )
        val_len = int(0.2 * len(full_dataset))
        train_len = len(full_dataset) - val_len
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True)
