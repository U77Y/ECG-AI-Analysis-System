import pandas as pd

from src.ecg_research.data import build_multilabel_targets


def test_scp_codes_map_to_multi_hot_superclasses():
    statements = pd.DataFrame(
        {
            "diagnostic": [1, 1, 0],
            "diagnostic_class": ["MI", "STTC", None],
        },
        index=["IMI", "STTC", "NDT"],
    )
    target = build_multilabel_targets({"IMI": 100, "STTC": 50, "NDT": 10}, statements)
    assert target.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
