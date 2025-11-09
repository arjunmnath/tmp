import os
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from transformers import AutoImageProcessor, AutoModelForImageClassification, get_scheduler

from PIL import Image

class EmosetJSONDataset(Dataset):
    def __init__(self, json_path, data_root: str, label2idx=None):
        """
        json file expected to be a JSON array of entries: [label, image_path, annotation_path]
        or a newline separated file of such arrays. This loader supports both.
        """
        self.data_root = Path(data_root)
        self.items = []
        p = Path(json_path)
        raw = p.read_text()
        raw = raw.strip()
        try:
            arr = json.loads(raw)
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], list):
                self.items = arr
            else:
                raise ValueError
        except Exception:
            items = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
            self.items = items
        # convert to tuples (label, image_path)
        self.samples = [(it[0], it[1]) for it in self.items]

        # build label mapping if needed (caller may provide)
        if label2idx is None:
            labels = sorted({lab for lab, _ in self.samples})
            self.label2idx = {l: i for i, l in enumerate(labels)}
        else:
            self.label2idx = label2idx

        self.samples = [(self.label2idx[s[0]], s[1]) for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label_idx, img_rel = self.samples[idx]
        p = Path(img_rel)
        if not p.is_absolute():
            p = self.data_root / p
        img = Image.open(p).convert("RGB")
        return img, int(label_idx)

def collate_processor(batch, processor):
    """
    batch: list[(PIL.Image, label_int)]
    returns dict with pixel_values tensor and labels tensor
    """
    images = [b[0] for b in batch]
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    enc = processor(images=images, return_tensors="pt")
    # enc['pixel_values'] shape: (batch, 3, H, W)
    enc["labels"] = labels
    return enc

def train_one_epoch(model, dataloader, optimizer, device, scaler, lr_scheduler=None):
    model.train()
    losses = 0.0
    preds = []
    targs = []
    pbar = tqdm(dataloader, desc="train", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        losses += loss.item() * pixel_values.size(0)
        logits = out.logits.detach().cpu()
        preds.extend(torch.argmax(logits, dim=1).tolist())
        targs.extend(labels.detach().cpu().tolist())
        pbar.set_postfix(loss=losses / (len(preds) + 1e-12))

    avg_loss = losses / len(dataloader.dataset)
    acc = accuracy_score(targs, preds)
    f1 = f1_score(targs, preds, average="macro")
    return avg_loss, acc, f1

def validate(model, dataloader, device):
    model.eval()
    losses = 0.0
    preds = []
    targs = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="val", leave=False)
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            out = model(pixel_values=pixel_values, labels=labels)
            loss = out.loss
            losses += loss.item() * pixel_values.size(0)
            logits = out.logits.detach().cpu()
            preds.extend(torch.argmax(logits, dim=1).tolist())
            targs.extend(labels.detach().cpu().tolist())
    avg_loss = losses / len(dataloader.dataset)
    acc = accuracy_score(targs, preds)
    f1 = f1_score(targs, preds, average="macro")
    return avg_loss, acc, f1

def make_sampler_from_dataset(dataset):
    labels = [lbl for _, lbl in dataset.samples]
    counts = Counter(labels)
    weights = [1.0 / counts[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True, help="root folder for images (where image/ lives)")
    p.add_argument("--train-json", type=str, default="train.json")
    p.add_argument("--val-json", type=str, default="val.json")
    p.add_argument("--model", type=str, default="google/vit-base-patch16-224")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--use-sampler", action="store_true", help="weighted sampler for class imbalance")
    p.add_argument("--save-dir", type=str, default="./checkpoints")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    train_json_path = Path(args.train_json)
    if not train_json_path.exists():
        raise FileNotFoundError(f"train json not found: {train_json_path}")
    raw = train_json_path.read_text().strip()
    try:
        arr = json.loads(raw)
        label_list = [it[0] for it in arr]
    except Exception:
        label_list = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            it = json.loads(line)
            label_list.append(it[0])
    labels_sorted = sorted(set(label_list))
    label2idx = {l: i for i, l in enumerate(labels_sorted)}
    num_labels = len(label2idx)
    print(f"Detected {num_labels} labels")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,  
    )
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    model.to(device)

    train_ds = EmosetJSONDataset(args.train_json, args.data_root, label2idx=label2idx)
    val_ds = EmosetJSONDataset(args.val_json, args.data_root, label2idx=label2idx)

    # dataloaders (use collate to call processor on list of PIL images)
    collate_fn = lambda batch: collate_processor(batch, processor)

    if args.use_sampler:
        sampler = make_sampler_from_dataset(train_ds)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)

    # optimizer + scheduler
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.lr)

    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer,
                                 num_warmup_steps=int(0.03 * num_training_steps),
                                 num_training_steps=num_training_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, device, scaler, lr_scheduler)
        val_loss, val_acc, val_f1 = validate(model, val_loader, device)
        print(f" train loss {train_loss:.4f} acc {train_acc:.4f} f1 {train_f1:.4f}")
        print(f" val   loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path = Path(args.save_dir) / f"best_vit_hf_epoch{epoch}_f1{val_f1:.4f}.pt"
            to_save = {
                "model_state": model.state_dict(),
                "label2idx": label2idx,
                "epoch": epoch,
                "args": vars(args)
            }
            torch.save(to_save, ckpt_path)
            print(" Saved checkpoint to", ckpt_path)

    print("Training complete. Best val F1:", best_val_f1)


if __name__ == "__main__":
    main()
