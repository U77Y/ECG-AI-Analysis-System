"""Train and evaluate the PTB-XL 1D CNN baseline."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import DIAGNOSTIC_CLASSES, PTBXLDataset
from .metrics import multilabel_metrics
from .model import ECGCNN1D


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_predictions(model, loader, device):
    model.eval()
    all_targets, all_probabilities = [], []
    with torch.no_grad():
        for signals, targets in loader:
            logits = model(signals.to(device))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.numpy())
    return np.concatenate(all_targets), np.concatenate(all_probabilities)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/ptb-xl")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open("configs/dataset.yaml", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    with open("configs/model.yaml", encoding="utf-8") as handle:
        model_cfg = yaml.safe_load(handle)

    dataset_cfg = data_cfg["dataset"]
    preprocessing_cfg = data_cfg["preprocessing"]
    training_cfg = model_cfg["training"]
    architecture_cfg = model_cfg["model"]
    seed = args.seed if args.seed is not None else model_cfg["experiment"]["seed"]
    epochs = args.epochs if args.epochs is not None else training_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else training_cfg["batch_size"]
    output_dir = Path(args.output_dir or model_cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    common = dict(
        data_dir=args.data_dir,
        sampling_rate=dataset_cfg["sampling_rate"],
        filter_enabled=preprocessing_cfg["bandpass_enabled"],
        lowcut_hz=preprocessing_cfg["lowcut_hz"],
        highcut_hz=preprocessing_cfg["highcut_hz"],
        filter_order=preprocessing_cfg["filter_order"],
    )
    train_set = PTBXLDataset(folds=dataset_cfg["train_folds"], **common)
    val_set = PTBXLDataset(folds=dataset_cfg["validation_folds"], **common)
    test_set = PTBXLDataset(folds=dataset_cfg["test_folds"], **common)

    workers = training_cfg["num_workers"]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECGCNN1D(
        input_channels=architecture_cfg["input_channels"],
        num_classes=architecture_cfg["num_classes"],
        dropout=architecture_cfg["dropout"],
    ).to(device)

    positives = train_set.targets.sum(axis=0)
    negatives = len(train_set) - positives
    pos_weight = torch.tensor(negatives / np.maximum(positives, 1), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    best_val_auc = -float("inf")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for signals, targets in train_loader:
            signals, targets = signals.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(signals), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * signals.size(0)

        val_true, val_prob = collect_predictions(model, val_loader, device)
        val_metrics = multilabel_metrics(val_true, val_prob)
        val_auc = val_metrics["macro_auroc"]
        row = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_set),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, allow_nan=True))
        score = val_auc if np.isfinite(val_auc) else -float("inf")
        if score > best_val_auc:
            best_val_auc = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": list(DIAGNOSTIC_CLASSES),
                    "seed": seed,
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                },
                output_dir / "best_model.pt",
            )

    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_true, test_prob = collect_predictions(model, test_loader, device)
    test_metrics = multilabel_metrics(test_true, test_prob)
    result = {
        "dataset": "PTB-XL",
        "test_folds": dataset_cfg["test_folds"],
        "selected_epoch": checkpoint["epoch"],
        "classes": list(DIAGNOSTIC_CLASSES),
        "test_metrics": test_metrics,
        "note": "Research split metrics only; not clinical validation or diagnostic accuracy.",
    }
    (output_dir / "history.json").write_text(json.dumps(history, indent=2, allow_nan=True), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
