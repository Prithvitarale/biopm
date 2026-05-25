"""PyTorch ``Dataset`` for BioPM inference."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MovementElementDataset(Dataset):
    """Wraps preprocessed arrays and yields one window per ``__getitem__``.

    The arrays are typically produced by
    :func:`biopm.preprocessing.load_preprocessed_h5`.

    ``__getitem__`` returns the 6-tuple

        ``(patches, label, subject_id, raw_acc_window,
           pos_info, additional_embedding)``

    matching the order expected by :func:`biopm.inference.extract_features`.
    The ``raw_acc_window`` field is named for the *raw* acceleration window
    but in practice we use it as the gravity stream — pass the gravity
    array as ``X_grav`` to the constructor.
    """

    def __init__(self,
                 X: np.ndarray,
                 X_grav: np.ndarray,
                 y: np.ndarray,
                 pos_info: np.ndarray,
                 additional_embedding: np.ndarray,
                 pid: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.X_grav = torch.from_numpy(np.ascontiguousarray(X_grav)).float()
        self.y = np.asarray(y)
        self.pid = np.asarray(pid)
        self.pos_info = torch.from_numpy(np.ascontiguousarray(pos_info)).float()
        self.additional_embedding = torch.from_numpy(
            np.ascontiguousarray(additional_embedding)).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.X[idx],
            float(self.y[idx]),
            int(self.pid[idx]) if np.issubdtype(self.pid.dtype, np.integer)
                                else self.pid[idx],
            self.X_grav[idx],
            self.pos_info[idx],
            self.additional_embedding[idx],
        )
