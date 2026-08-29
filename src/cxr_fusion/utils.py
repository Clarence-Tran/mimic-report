from __future__ import annotations

import random

import numpy as np
import torch


def id_str(value: object) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_label(name: str) -> str:
    return (
        str(name)
        .lower()
        .replace("y_", "")
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )
