"""Multi-label evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def multilabel_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    auc_values, ap_values, sensitivities, specificities = [], [], [], []
    for column in range(y_true.shape[1]):
        truth = y_true[:, column]
        pred = predictions[:, column]
        if np.unique(truth).size == 2:
            auc_values.append(float(roc_auc_score(truth, probabilities[:, column])))
            ap_values.append(float(average_precision_score(truth, probabilities[:, column])))
        tp = int(np.sum((truth == 1) & (pred == 1)))
        fn = int(np.sum((truth == 1) & (pred == 0)))
        tn = int(np.sum((truth == 0) & (pred == 0)))
        fp = int(np.sum((truth == 0) & (pred == 1)))
        sensitivities.append(tp / (tp + fn) if tp + fn else float("nan"))
        specificities.append(tn / (tn + fp) if tn + fp else float("nan"))

    return {
        "macro_auroc": float(np.mean(auc_values)) if auc_values else float("nan"),
        "macro_auprc": float(np.mean(ap_values)) if ap_values else float("nan"),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_sensitivity": float(np.nanmean(sensitivities)),
        "macro_specificity": float(np.nanmean(specificities)),
        "per_class_sensitivity": sensitivities,
        "per_class_specificity": specificities,
    }
