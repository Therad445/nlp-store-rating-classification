import argparse
import json
import math
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup


THRESHOLDS = 4


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = " ".join(text.split())
    return text


def make_ordinal_targets(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    targets = np.zeros((len(labels), THRESHOLDS), dtype=np.float32)
    for k in range(THRESHOLDS):
        targets[:, k] = (labels > (k + 1)).astype(np.float32)
    return targets


class ReviewDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = list(texts)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.ordinal_targets = None if labels is None else make_ordinal_targets(self.labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
            item["target"] = self.ordinal_targets[idx]
        return item


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts = [x["text"] for x in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            enc["labels"] = torch.tensor([x["label"] for x in batch], dtype=torch.long)
            enc["targets"] = torch.tensor(np.stack([x["target"] for x in batch]), dtype=torch.float32)
        return enc


class OrdinalClassifier(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, THRESHOLDS)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        logits = self.head(self.dropout(pooled))
        return logits


class OrdinalLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None, smoothing: float = 0.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


@torch.no_grad()
def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    thr = torch.sigmoid(logits)
    thr = torch.cummin(thr, dim=1).values
    p1 = 1.0 - thr[:, 0]
    p2 = thr[:, 0] - thr[:, 1]
    p3 = thr[:, 1] - thr[:, 2]
    p4 = thr[:, 2] - thr[:, 3]
    p5 = thr[:, 3]
    probs = torch.stack([p1, p2, p3, p4, p5], dim=1)
    probs = probs.clamp(min=1e-8)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []
    for batch in loader:
        labels = batch.pop("labels", None)
        batch.pop("targets", None)
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)
        probs = logits_to_probs(logits).cpu().numpy()
        all_probs.append(probs)
        if labels is not None:
            all_labels.append(labels.numpy())
    probs = np.vstack(all_probs)
    labels = None if not all_labels else np.concatenate(all_labels)
    return probs, labels


def train_one_fold(model, train_loader, val_loader, loss_fn, optimizer, scheduler, device, epochs, grad_accum, use_amp, patience):
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_state = None
    best_score = -1.0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for step, batch in enumerate(progress, start=1):
            targets = batch.pop("targets").to(device)
            batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(**batch)
                loss = loss_fn(logits, targets) / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        val_probs, val_labels = predict_loader(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1) + 1
        score = f1_score(val_labels, val_pred, average="weighted")
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"val_weighted_f1={score:.6f}")
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_score


def make_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    counts = pd.Series(labels).value_counts().sort_index()
    weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = np.array([weights[int(x)] for x in labels], dtype=np.float64)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_pos_weight(labels: np.ndarray, device: torch.device) -> torch.Tensor:
    ordinal = make_ordinal_targets(labels)
    pos = ordinal.sum(axis=0)
    neg = len(labels) - pos
    ratio = np.where(pos > 0, neg / pos, 1.0)
    return torch.tensor(ratio, dtype=torch.float32, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="train.csv")
    parser.add_argument("--test", type=str, default="test.csv")
    parser.add_argument("--output-dir", type=str, default="runs/rubert_ordinal")
    parser.add_argument("--model-name", type=str, default="DeepPavlov/rubert-base-cased-conversational")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    train_df["text"] = train_df["text"].map(normalize_text)
    test_df["text"] = test_df["text"].map(normalize_text)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collator = Collator(tokenizer, args.max_length)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_probs = np.zeros((len(train_df), 5), dtype=np.float32)
    test_probs = np.zeros((len(test_df), 5), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df["text"], train_df["rate"]), start=1):
        print(f"fold {fold}/{args.folds}")
        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        train_ds = ReviewDataset(tr_df["text"].tolist(), tr_df["rate"].to_numpy())
        val_ds = ReviewDataset(va_df["text"].tolist(), va_df["rate"].to_numpy())
        test_ds = ReviewDataset(test_df["text"].tolist())

        sampler = make_sampler(tr_df["rate"].to_numpy())
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=2, pin_memory=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collator)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collator)

        model = OrdinalClassifier(args.model_name, dropout=args.dropout).to(device)
        no_decay = {"bias", "LayerNorm.weight"}
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(params, lr=args.lr)
        steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
        total_steps = steps_per_epoch * args.epochs
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        pos_weight = build_pos_weight(tr_df["rate"].to_numpy(), device)
        loss_fn = OrdinalLoss(pos_weight=pos_weight, smoothing=args.smoothing)

        model, best_score = train_one_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=args.epochs,
            grad_accum=args.grad_accum,
            use_amp=use_amp,
            patience=args.patience,
        )

        fold_scores.append(best_score)
        fold_val_probs, _ = predict_loader(model, val_loader, device)
        fold_test_probs, _ = predict_loader(model, test_loader, device)
        oof_probs[va_idx] = fold_val_probs
        test_probs += fold_test_probs / args.folds

        torch.save(model.state_dict(), os.path.join(args.output_dir, f"fold_{fold}.pt"))
        np.save(os.path.join(args.output_dir, f"val_probs_fold_{fold}.npy"), fold_val_probs)
        np.save(os.path.join(args.output_dir, f"test_probs_fold_{fold}.npy"), fold_test_probs)
        torch.cuda.empty_cache()

    oof_pred = oof_probs.argmax(axis=1) + 1
    oof_score = f1_score(train_df["rate"], oof_pred, average="weighted")
    print(f"cv_weighted_f1={oof_score:.6f}")

    np.save(os.path.join(args.output_dir, "oof_probs.npy"), oof_probs)
    np.save(os.path.join(args.output_dir, "test_probs.npy"), test_probs)

    metrics = {
        "fold_scores": fold_scores,
        "mean_fold_score": float(np.mean(fold_scores)),
        "oof_weighted_f1": float(oof_score),
        "config": vars(args),
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
