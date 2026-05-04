import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import F1Score, AveragePrecision, Precision, Recall, JaccardIndex

# ==========================================
# 1. Raw PyTorch Architecture
# ==========================================
class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.conv(x)

class UpBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class ASKUNPP(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes=2, deep_supervision=False):
        super().__init__()
        ch = base_channels
        self.deep_supervision = deep_supervision

        self.conv00 = ConvBlock(in_channels, ch)        
        self.pool0 = nn.MaxPool2d(2)
        self.conv10 = ConvBlock(ch, ch*2)               
        self.pool1 = nn.MaxPool2d(2)
        self.conv20 = ConvBlock(ch*2, ch*4)             
        self.pool2 = nn.MaxPool2d(2)
        self.conv30 = ConvBlock(ch*4, ch*8)             
        self.pool3 = nn.MaxPool2d(2)
        self.conv40 = ConvBlock(ch*8, ch*16)            

        self.up_40_30 = UpBlock(ch*16, ch*8) 
        self.up_31_21 = UpBlock(ch*8, ch*4)  
        self.up_21_11 = UpBlock(ch*4, ch*2)  
        self.up_11_01 = UpBlock(ch*2, ch)    

        self.conv31 = nn.Sequential(nn.Conv2d(ch*16, ch*8, kernel_size=1, bias=False), nn.BatchNorm2d(ch*8), nn.ReLU(inplace=True))
        self.att31 = AttentionBlock(ch*8)

        self.conv22 = nn.Sequential(nn.Conv2d(ch*8, ch*4, kernel_size=1, bias=False), nn.BatchNorm2d(ch*4), nn.ReLU(inplace=True))
        self.att22 = AttentionBlock(ch*4)

        self.conv12 = nn.Sequential(nn.Conv2d(ch*4, ch*2, kernel_size=1, bias=False), nn.BatchNorm2d(ch*2), nn.ReLU(inplace=True))
        self.att12 = AttentionBlock(ch*2)

        self.conv02 = nn.Sequential(nn.Conv2d(ch*2, ch, kernel_size=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.att02 = AttentionBlock(ch)

        self.out_0 = nn.Conv2d(ch, num_classes, kernel_size=1)
        self.out_1 = nn.Conv2d(ch*2, num_classes, kernel_size=1)
        self.out_2 = nn.Conv2d(ch*4, num_classes, kernel_size=1)
        self.out_3 = nn.Conv2d(ch*8, num_classes, kernel_size=1)

    def forward(self, x):
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        x40_up = self.up_40_30(x40)
        x31 = self.att31(self.conv31(torch.cat([x30, x40_up], dim=1)))

        x31_up = self.up_31_21(x31)
        x21 = self.att22(self.conv22(torch.cat([x20, x31_up], dim=1)))

        x21_up = self.up_21_11(x21)
        x11 = self.att12(self.conv12(torch.cat([x10, x21_up], dim=1)))

        x11_up = self.up_11_01(x11)
        x01 = self.att02(self.conv02(torch.cat([x00, x11_up], dim=1)))

        if self.deep_supervision:
            return self.out_0(x01), self.out_1(x11), self.out_2(x21), self.out_3(x31)
        return self.out_0(x01)

# ==========================================
# 2. PyTorch Lightning Training Wrapper
# ==========================================
class ASKULightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4, in_channels=3, num_classes=2, deep_supervision=True):
        super().__init__()
        self.save_hyperparameters()
        self.net = ASKUNPP(in_channels=in_channels, base_channels=64, num_classes=num_classes, deep_supervision=deep_supervision)
        self.criterion = nn.CrossEntropyLoss()

        metrics = {"f1": F1Score, "auprc": AveragePrecision, "precision": Precision, "recall": Recall, "iou": JaccardIndex}
        for name, Metric in metrics.items():
            setattr(self, f"train_{name}", Metric(task="multiclass", num_classes=num_classes, average="macro" if name != "auprc" else None))
            setattr(self, f"val_{name}", Metric(task="multiclass", num_classes=num_classes, average="macro" if name != "auprc" else None))

    def forward(self, x):
        outputs = self.net(x)
        return outputs[-1] if self.hparams.deep_supervision else outputs

    def compute_loss(self, preds, target):
        if isinstance(preds, (list, tuple)):
            loss = sum(self.criterion(p, F.interpolate(target.unsqueeze(1).float(), size=p.shape[2:], mode='nearest').squeeze(1).long() if p.shape[2:] != target.shape[1:] else target) for p in preds)
            return loss / len(preds)
        
        target = F.interpolate(target.unsqueeze(1).float(), size=preds.shape[2:], mode='nearest').squeeze(1).long() if preds.shape[2:] != target.shape[1:] else target
        return self.criterion(preds, target)

    def _shared_step(self, batch, stage):
        x, y = batch
        preds = self.net(x)
        y_true = y.squeeze(1)
        loss = self.compute_loss(preds, y_true)
        
        preds_last = preds[-1] if isinstance(preds, (list, tuple)) else preds
        probs = torch.softmax(preds_last, dim=1)
        y_true_resized = F.interpolate(y_true.unsqueeze(1).float(), size=preds_last.shape[2:], mode='nearest').squeeze(1).long() if y_true.shape[1:] != preds_last.shape[2:] else y_true
        
        for name in ["f1", "auprc", "precision", "recall", "iou"]:
            getattr(self, f"{stage}_{name}").update(probs, y_true_resized)
            
        self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx): return self._shared_step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._shared_step(batch, "val")

    def on_train_epoch_end(self):
        for name in ["f1", "auprc", "precision", "recall", "iou"]:
            self.log(f"train/{name}", getattr(self, f"train_{name}").compute(), prog_bar=True)
            getattr(self, f"train_{name}").reset()

    def on_validation_epoch_end(self):
        for name in ["f1", "auprc", "precision", "recall", "iou"]:
            self.log(f"val/{name}", getattr(self, f"val_{name}").compute(), prog_bar=True)
            getattr(self, f"val_{name}").reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
