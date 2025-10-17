# classifier_train.py
import os
import random
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ---------------- Dataset ----------------
class CropSensorDataset(Dataset):
    def __init__(self, metadata_csv, label_col="label", sensor_cols=None, transform=None):
        self.df = pd.read_csv(metadata_csv)
        if 'crop_path' not in self.df.columns:
            raise ValueError("metadata csv must contain 'crop_path' column")
        self.transform = transform
        self.label_col = label_col
        self.sensor_cols = sensor_cols if sensor_cols else []
        # fill missing sensor columns with zeros
        for c in self.sensor_cols:
            if c not in self.df.columns:
                self.df[c] = 0.0

        # map labels to ints
        if self.label_col not in self.df.columns:
            raise ValueError(f"Label column '{self.label_col}' not found in metadata")
        self.classes = sorted(self.df[self.label_col].unique().tolist())
        self.class2idx = {c:i for i,c in enumerate(self.classes)}
        self.df['label_idx'] = self.df[self.label_col].map(self.class2idx)

        self.scaler = StandardScaler()

    def fit_scaler(self, indices):
        if len(self.sensor_cols) == 0:
            return
        vals = self.df.loc[indices, self.sensor_cols].fillna(0.0).values.astype(np.float32)
        self.scaler.fit(vals)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['crop_path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        if len(self.sensor_cols) > 0:
            s = row[self.sensor_cols].fillna(0.0).values.astype(np.float32)
            s = self.scaler.transform([s])[0] if hasattr(self.scaler, 'mean_') else s
            sensor = torch.tensor(s, dtype=torch.float32)
        else:
            sensor = torch.zeros(0, dtype=torch.float32)
        label = int(row['label_idx'])
        return img, sensor, label

# ---------------- Model ----------------
class ResNetSensorFusion(nn.Module):
    def __init__(self, backbone='resnet18', num_classes=3, sensor_dim=0, pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            feat_dim = base.fc.in_features
        else:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            feat_dim = base.fc.in_features

        # remove fc
        modules = list(base.children())[:-1]  # remove last fc
        self.backbone = nn.Sequential(*modules)  # outputs (B, feat_dim, 1, 1)
        self.feat_dim = feat_dim
        self.sensor_dim = sensor_dim

        # fusion head
        fusion_dim = feat_dim + sensor_dim
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fusion_dim, fusion_dim//2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim//2, num_classes)
        )

    def forward(self, x, sensor):
        # x: (B,3,H,W)
        f = self.backbone(x)  # (B, feat_dim, 1, 1)
        f = f.view(f.size(0), -1)
        if sensor is not None and sensor.numel() > 0:
            out = torch.cat([f, sensor], dim=1)
        else:
            out = f
        logits = self.classifier(out)
        return logits

# ---------------- Training utilities ----------------
def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return train_tf, val_tf

def make_loaders(metadata_csv, label_col, sensor_cols, batch_size, num_workers, device, val_frac=0.15, seed=42):
    df = pd.read_csv(metadata_csv)
    # split by stratified labels
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(df, test_size=val_frac, stratify=df[label_col], random_state=seed)
    train_csv = Path(metadata_csv).parent / "train_metadata.csv"
    val_csv = Path(metadata_csv).parent / "val_metadata.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    train_tf, val_tf = get_transforms()
    train_ds = CropSensorDataset(str(train_csv), label_col=label_col, sensor_cols=sensor_cols, transform=train_tf)
    val_ds = CropSensorDataset(str(val_csv), label_col=label_col, sensor_cols=sensor_cols, transform=val_tf)

    # fit scaler on train
    train_ds.fit_scaler(train_df.index)
    val_ds.scaler = train_ds.scaler

    # sampler for class balance
    counts = Counter(train_df[label_col].tolist())
    weights = [1.0/counts[l] for l in train_df[label_col]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device.type=='cuda'))
    return train_loader, val_loader, train_ds, val_ds

def train(args):
    device = torch.device(args.device if isinstance(args.device, str) else args.device)
    train_loader, val_loader, train_ds, val_ds = make_loaders(args.metadata_csv, args.label_col, args.sensor_cols,
                                                              args.batch_size, args.num_workers, device,
                                                              val_frac=args.val_frac, seed=args.seed)
    sensor_dim = len(args.sensor_cols)
    model = ResNetSensorFusion(backbone=args.backbone, num_classes=args.num_classes, sensor_dim=sensor_dim, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs)
    scaler = GradScaler() if device.type=='cuda' else None

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Train E{epoch+1}/{args.epochs}")
        for imgs, sensors, labels in pbar:
            imgs = imgs.to(device)
            sensors = sensors.to(device) if sensors.numel()>0 else None
            labels = labels.to(device)
            optimizer.zero_grad()
            if scaler is not None:
                with autocast():
                    outputs = model(imgs, sensors)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs, sensors)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{correct/total:.4f}"})
        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        conf = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
        with torch.no_grad():
            for imgs, sensors, labels in val_loader:
                imgs = imgs.to(device)
                sensors = sensors.to(device) if sensors.numel()>0 else None
                labels = labels.to(device)
                outputs = model(imgs, sensors)
                preds = outputs.argmax(dim=1)
                for t,p in zip(labels.view(-1), preds.view(-1)):
                    conf[t.long(), p.long()] += 1
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(1, val_total)
        print(f"Epoch {epoch+1}: val_acc={val_acc:.4f}")
        print("Confusion:\n", conf.numpy())
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.out_model)
            print("Saved best model:", args.out_model)

    # Export TorchScript (trace)
    try:
        model.load_state_dict(torch.load(args.out_model, map_location=device))
        model.eval()
        example_img = torch.randn(1,3,224,224).to(device)
        example_sensor = torch.randn(1, sensor_dim).to(device) if sensor_dim>0 else torch.randn(1,0).to(device)
        traced = torch.jit.trace(model, (example_img, example_sensor))
        traced.save(args.out_ts)
        print("Saved TorchScript to", args.out_ts)
    except Exception as e:
        print("TorchScript failed:", e)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--metadata-csv", required=True, help="crops_metadata.csv with crop_path,label and sensor columns")
    parser.add_argument("--label-col", default="label", help="column name for label in CSV")
    parser.add_argument("--sensor-cols", nargs='*', default=["r","g","b","temp","dist"], help="sensor column names in CSV")
    parser.add_argument("--out-model", default="models/classifier_best.pt")
    parser.add_argument("--out-ts", default="models/classifier_ts.pt")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18","resnet50"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
