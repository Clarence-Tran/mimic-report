from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SplitConfig
from .data import AlignedCohort


def split_for_seed(aligned: AlignedCohort, split_cfg: SplitConfig, seed: int):
    del seed

    subject_id_per_row = aligned.sample_meta["subject_id"].to_numpy()
    subject_anchor_time = (
        pd.DataFrame({"subject_id": subject_id_per_row, "index_time": aligned.study_dates})
        .groupby("subject_id")["index_time"]
        .min()
    )
    order = subject_anchor_time.sort_values().index.to_numpy()

    counts_per_subject = pd.Series(subject_id_per_row).value_counts()
    cum_studies = counts_per_subject.reindex(order).cumsum()
    total = cum_studies.iloc[-1]

    train_cut = total * (1 - split_cfg.test_size) * (1 - split_cfg.val_size_from_temp)
    val_cut = total * (1 - split_cfg.test_size)

    train_subjects = cum_studies.index[cum_studies <= train_cut]
    val_subjects = cum_studies.index[(cum_studies > train_cut) & (cum_studies <= val_cut)]

    is_train = np.isin(subject_id_per_row, train_subjects)
    is_val = np.isin(subject_id_per_row, val_subjects)
    is_test = ~is_train & ~is_val

    return np.where(is_train)[0], np.where(is_val)[0], np.where(is_test)[0]
