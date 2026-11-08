"""PTB-XL metadata parsing and waveform dataset."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_ecg

DIAGNOSTIC_CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


def _parse_scp_codes(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if pd.isna(value):
        return {}
    parsed = ast.literal_eval(str(value))
    return parsed if isinstance(parsed, dict) else {}


def build_multilabel_targets(
    scp_codes: object,
    scp_statements: pd.DataFrame,
    classes: Sequence[str] = DIAGNOSTIC_CLASSES,
) -> np.ndarray:
    """Map a record's SCP codes to diagnostic-superclass multi-hot labels."""
    codes = _parse_scp_codes(scp_codes)
    targets = np.zeros(len(classes), dtype=np.float32)
    if "diagnostic" not in scp_statements.columns or "diagnostic_class" not in scp_statements.columns:
        raise ValueError("scp_statements.csv must contain diagnostic and diagnostic_class columns")

    for code in codes:
        if code not in scp_statements.index:
            continue
        statement = scp_statements.loc[code]
        if isinstance(statement, pd.DataFrame):
            statement = statement.iloc[0]
        if not bool(statement["diagnostic"]):
            continue
        superclass = statement["diagnostic_class"]
        if pd.notna(superclass) and superclass in classes:
            targets[classes.index(superclass)] = 1.0
    return targets


def load_ptbxl_metadata(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    database_path = root / "ptbxl_database.csv"
    statements_path = root / "scp_statements.csv"
    if not database_path.exists() or not statements_path.exists():
        raise FileNotFoundError(
            f"Expected ptbxl_database.csv and scp_statements.csv inside {root}. "
            "Download and extract PTB-XL from PhysioNet first."
        )
    records = pd.read_csv(database_path, index_col="ecg_id")
    statements = pd.read_csv(statements_path, index_col=0)
    required = {"scp_codes", "strat_fold", "filename_hr"}
    missing = required.difference(records.columns)
    if missing:
        raise ValueError(f"PTB-XL metadata is missing columns: {sorted(missing)}")
    return records, statements


class PTBXLDataset(Dataset):
    """Load one 12-lead PTB-XL waveform at a time to limit RAM use."""

    def __init__(
        self,
        data_dir: str | Path,
        folds: Sequence[int],
        sampling_rate: int = 500,
        filter_enabled: bool = True,
        lowcut_hz: float = 0.5,
        highcut_hz: float = 40.0,
        filter_order: int = 4,
    ) -> None:
        self.root = Path(data_dir)
        records, self.statements = load_ptbxl_metadata(self.root)
        self.records = records[records["strat_fold"].isin(list(folds))].copy()
        if self.records.empty:
            raise ValueError(f"No PTB-XL records found for folds {list(folds)}")
        self.sampling_rate = sampling_rate
        self.filter_enabled = filter_enabled
        self.lowcut_hz = lowcut_hz
        self.highcut_hz = highcut_hz
        self.filter_order = filter_order
        self.targets = np.stack([
            build_multilabel_targets(row.scp_codes, self.statements)
            for row in self.records.itertuples()
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import wfdb

        row = self.records.iloc[index]
        record_path = self.root / str(row["filename_hr"])
        signal, _ = wfdb.rdsamp(str(record_path))
        signal = preprocess_ecg(
            signal,
            sampling_rate=self.sampling_rate,
            filter_enabled=self.filter_enabled,
            lowcut_hz=self.lowcut_hz,
            highcut_hz=self.highcut_hz,
            filter_order=self.filter_order,
        )
        # WFDB returns (samples, leads); PyTorch Conv1d expects (leads, samples).
        x = torch.from_numpy(np.asarray(signal.T, dtype=np.float32))
        y = torch.from_numpy(self.targets[index].copy())
        return x, y
