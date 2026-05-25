"""Subject-aware nested cross-validation for downstream classification.

The protocol matches what the paper reports in Table 1 / Table 2:

  * Outer CV
      LOSO (leave-one-subject-out) when the dataset has <= 10 subjects,
      otherwise GroupShuffleSplit with 5 splits and an 80/20 ratio.
  * Inner C search
      A single GroupShuffleSplit on the training subjects, ``val_fraction``
      held out (default 12.5%), grid search over the LogReg ``C`` values.
  * Preprocessing
      StandardScaler refit on the *outer* training fold only.

The single entry point is :func:`logreg_nested_cv`.  It returns a dict of
macro-F1 and accuracy aggregates along with the selected ``C`` mode.

If your dataset has very few subjects per class the inner split may fall
through and we fall back to ``C=1.0`` for that fold (a no-op grid search).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements.txt
    def tqdm(it, **kw):
        return it


DEFAULT_C_GRID: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LOSO_THRESHOLD: int = 10


def _fit_logreg(X: np.ndarray, y: np.ndarray, C: float) -> LogisticRegression:
    # The "lbfgs" solver does multinomial logistic regression automatically
    # for >2 classes in all supported sklearn versions, and the explicit
    # ``multi_class="multinomial"`` kwarg has been deprecated in sklearn 1.5+
    # and removed in 1.7.
    clf = LogisticRegression(
        C=C, solver="lbfgs", tol=1e-3, max_iter=1000, random_state=42,
    )
    clf.fit(X, y)
    return clf


def _inner_C_search(X_train: np.ndarray, y_train: np.ndarray,
                    groups_train: np.ndarray,
                    val_fraction: float,
                    C_grid: Tuple[float, ...]) -> float:
    if len(np.unique(groups_train)) < 3:
        return 1.0
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    tr, va = next(gss.split(X_train, y_train, groups=groups_train))
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train[tr])
    Xva = sc.transform(X_train[va])
    best_C, best_f1 = C_grid[0], -1.0
    for C in C_grid:
        try:
            clf = _fit_logreg(Xtr, y_train[tr], C)
            f1 = f1_score(y_train[va], clf.predict(Xva), average="macro")
            if f1 > best_f1:
                best_f1, best_C = f1, C
        except Exception:
            continue
    return best_C


def logreg_nested_cv(
    features: np.ndarray,
    labels: np.ndarray,
    subject_ids: np.ndarray,
    *,
    val_fraction: float = 0.125,
    n_outer_splits: int = 5,
    outer_test_size: float = 0.2,
    C_grid: Tuple[float, ...] = DEFAULT_C_GRID,
    verbose: bool = True,
) -> Dict[str, float]:
    """Subject-aware nested CV with inner ``C`` selection.

    Args:
        features      : (N, D) BioPM features.
        labels        : (N,) integer labels.
        subject_ids   : (N,) subject ids (must be hashable).
        val_fraction  : fraction of training subjects used for inner C search.
        n_outer_splits: number of outer folds when using GroupShuffleSplit.
        outer_test_size: outer test fraction when using GroupShuffleSplit.
        C_grid        : candidate regularisation strengths.
        verbose       : print progress bar and per-fold scores.

    Returns:
        dict with keys ``macro_f1_mean``, ``macro_f1_std``,
        ``accuracy_mean``, ``accuracy_std``, ``n_folds``, ``C_mode``,
        ``cv_strategy``, ``per_fold_f1``, ``per_fold_acc``,
        ``selected_C``.
    """
    y = np.asarray(labels).astype(int)
    groups = np.asarray(subject_ids)
    n_subjects = len(np.unique(groups))
    use_loso = n_subjects <= LOSO_THRESHOLD
    if use_loso:
        outer = LeaveOneGroupOut()
        n_outer = n_subjects
        cv_name = "LOSO"
    else:
        outer = GroupShuffleSplit(n_splits=n_outer_splits,
                                  test_size=outer_test_size, random_state=42)
        n_outer = n_outer_splits
        cv_name = "GroupShuffleSplit"

    fold_f1: List[float] = []
    fold_acc: List[float] = []
    selected_C: List[float] = []

    iterator = outer.split(features, y, groups=groups)
    if verbose:
        iterator = tqdm(iterator, total=n_outer, desc=cv_name, leave=False)

    for train_idx, test_idx in iterator:
        X_train, y_train, g_train = features[train_idx], y[train_idx], groups[train_idx]
        X_test, y_test = features[test_idx], y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue

        best_C = _inner_C_search(X_train, y_train, g_train, val_fraction, C_grid)
        selected_C.append(best_C)

        sc = StandardScaler()
        Xtr = sc.fit_transform(X_train)
        Xte = sc.transform(X_test)
        clf = _fit_logreg(Xtr, y_train, best_C)
        y_pred = clf.predict(Xte)
        fold_f1.append(f1_score(y_test, y_pred, average="macro"))
        fold_acc.append(accuracy_score(y_test, y_pred))

    C_mode = max(set(selected_C), key=selected_C.count) if selected_C else None
    return {
        "macro_f1_mean": float(np.mean(fold_f1)) if fold_f1 else float("nan"),
        "macro_f1_std":  float(np.std(fold_f1))  if fold_f1 else float("nan"),
        "accuracy_mean": float(np.mean(fold_acc)) if fold_acc else float("nan"),
        "accuracy_std":  float(np.std(fold_acc))  if fold_acc else float("nan"),
        "n_folds": len(fold_f1),
        "C_mode": C_mode,
        "cv_strategy": cv_name,
        "per_fold_f1": fold_f1,
        "per_fold_acc": fold_acc,
        "selected_C": selected_C,
    }
