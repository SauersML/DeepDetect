import os
import sys
import json
import math
import time
import glob
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

import pandas as pd


# Matplotlib only (no seaborn)
import matplotlib
# Detect if we are in a notebook
def _in_notebook() -> bool:
    try:
        from IPython import get_ipython  # type: ignore
        shell = get_ipython()
        if shell is None:
            return False
        return 'IPKernelApp' in sys.modules
    except Exception:
        return False

if _in_notebook():
    # Inline display in notebooks
    from IPython.display import display  # noqa: F401
    matplotlib.use("module://matplotlib_inline.backend_inline")
else:
    # Non-interactive backend for scripts
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Scikit-learn metrics & projections
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_recall_curve,
    roc_curve, average_precision_score, brier_score_loss, confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Hugging Face datasets (for tokenized cache)
try:
    from datasets import load_from_disk as hf_load_from_disk  # noqa: F401
except Exception:
    hf_load_from_disk = None


# -------------------------
# Hard-coded configuration
# -------------------------
RUNS_ROOT = Path("./runs")
PLOTS_ROOT = RUNS_ROOT / "plots"
INCLUDE_TEST = True
MAKE_TSNE = True              # compute t-SNE when val_repr_for_viz.npz is present
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000
PCA_COMPONENTS = 2
RELIABILITY_BINS = 15
DPI = 160

# Aesthetic tweaks (Matplotlib only)
plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "figure.autolayout": True,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
})
# Color cycle (beautiful but accessible)
TAB = plt.cm.tab10.colors
TAB20 = plt.cm.tab20.colors
C_ACCENT = TAB[1]    # blue-ish
C_ACCENT2 = TAB[2]   # orange-ish
C_POS = TAB[3]       # green-ish
C_NEG = TAB[0]       # blue-ish
C_ERR = TAB[2]       # orange
CM_HEAT = "viridis"  # heatmap colormap


# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_metrics_csv(path: Path) -> Optional[pd.DataFrame]:
    if pd is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        # Normalize column names
        df.columns = [c.strip() for c in df.columns]
        # Ensure expected columns exist (fill if missing)
        for col in ("event", "epoch", "step", "loss", "acc", "f1_macro", "auroc", "lr", "timestamp"):
            if col not in df.columns:
                df[col] = np.nan
        return df
    except Exception:
        return None

def find_run_dirs() -> List[Path]:
    if not RUNS_ROOT.exists():
        return []
    dirs = [p for p in RUNS_ROOT.iterdir() if p.is_dir()]
    # exclude known non-run directories
    skip = {"plots", "api_baselines", "cache"}
    dirs = [d for d in dirs if d.name not in skip]
    # keep those with at least metrics.csv or best/preds
    ok = []
    for d in dirs:
        if (d / "metrics.csv").exists() or (d / "best" / "preds").exists():
            ok.append(d)
    return ok

def mtime_of_run(d: Path) -> float:
    candidates = [
        d / "metrics.csv",
        d / "best" / "preds" / "metrics.json",
        d / "best" / "preds" / "metrics_thresh.json",
        d / "best" / "cfg.json",
    ]
    mt = 0.0
    for c in candidates:
        if c.exists():
            mt = max(mt, c.stat().st_mtime)
    return mt

def human_name_for_run(run_dir: Path) -> str:
    return run_dir.name

def load_split_arrays(best_dir: Path, split: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    lab = best_dir / "preds" / f"{split}_labels.npy"
    prob = best_dir / "preds" / (f"{split}_prob_class1.npy" if split != "val" else "val_prob_ai.npy")
    argmax = best_dir / "preds" / (f"{split}_pred_argmax.npy" if split != "val" else "val_pred.npy")

    y = np.load(lab) if lab.exists() else None
    p = np.load(prob) if prob.exists() else None
    yhat = np.load(argmax) if argmax.exists() else None
    return y, p, yhat

def compute_ece_brier(y: np.ndarray, p: np.ndarray, n_bins: int = RELIABILITY_BINS) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Return (ECE, Brier, bin_acc, bin_conf) with equal-width bins."""
    # Guard
    y = y.astype(int)
    p = np.clip(p.astype(float), 0.0, 1.0)
    brier = brier_score_loss(y, p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_acc = np.zeros(n_bins, dtype=float)
    bin_conf = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i+1] if i < n_bins-1 else p <= bins[i+1])
        if not np.any(m):
            continue
        conf = np.mean(p[m])
        acc = np.mean((p[m] >= 0.5).astype(int) == y[m])  # accuracy in bin using 0.5 ref (for calibration intuition)
        bin_acc[i] = acc
        bin_conf[i] = conf
        ece += (np.mean(m) * abs(acc - conf))
    return float(ece), float(brier), bin_acc, bin_conf

def show_or_close(fig: plt.Figure, should_show: bool) -> None:
    if should_show:
        plt.show()
    plt.close(fig)


# -------------------------
# Plot helpers
# -------------------------
def plot_train_loss(df: pd.DataFrame, out_dir: Path, show=False) -> None:
    if df is None or df.empty:
        return
    d = df[df["event"] == "train_step"].copy()
    if d.empty or "loss" not in d.columns:
        return
    x = d["step"].values if "step" in d.columns else np.arange(len(d))
    y = d["loss"].values

    # EMA smoothing
    alpha = 0.1
    ema = []
    s = None
    for v in y:
        s = v if s is None else alpha*v + (1-alpha)*s
        ema.append(s)
    ema = np.array(ema)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, label="Train loss (raw)", color=TAB[0], alpha=0.45)
    ax.plot(x, ema, label="EMA(α=0.1)", color=TAB[3])
    ax.set_title("Training loss vs step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.legend(loc="best", frameon=False)
    fig.savefig(out_dir / "train_loss_vs_step.png")
    show_or_close(fig, show)

def plot_lr(df: pd.DataFrame, out_dir: Path, show=False) -> None:
    if df is None or df.empty:
        return
    d = df[df["event"] == "train_step"].copy()
    if d.empty or "lr" not in d.columns:
        return
    x = d["step"].values if "step" in d.columns else np.arange(len(d))
    y = d["lr"].values
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(x, y, color=TAB[1])
    ax.set_title("Learning rate vs step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("LR")
    fig.savefig(out_dir / "lr_vs_step.png")
    show_or_close(fig, show)

def plot_val_metrics(df: pd.DataFrame, out_dir: Path, show=False) -> None:
    if df is None or df.empty:
        return
    d = df[df["event"] == "eval"].copy()
    if d.empty:
        return
    x = d["step"].values if "step" in d.columns else np.arange(len(d))

    for col, name, fname in [("f1_macro", "F1 (macro)", "val_f1_vs_step.png"),
                             ("acc", "Accuracy", "val_acc_vs_step.png"),
                             ("auroc", "ROC-AUC", "val_auc_vs_step.png")]:
        if col not in d.columns or d[col].isna().all():
            continue
        y = d[col].values
        fig, ax = plt.subplots(figsize=(7.8, 3.8))
        ax.plot(x, y, marker="o", linestyle="-", color=TAB[2], alpha=0.95)
        ax.set_title(f"Validation {name} vs step")
        ax.set_xlabel("Global step")
        ax.set_ylabel(name)
        fig.savefig(out_dir / fname)
        show_or_close(fig, show)

def plot_class_balance(y: np.ndarray, id2label: Dict[int,str], out_path: Path, title: str, show=False) -> None:
    if y is None:
        return
    unique, counts = np.unique(y.astype(int), return_counts=True)
    names = [id2label.get(int(k), str(k)) for k in unique]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    ax.bar(names, counts, color=[TAB[0] if n.upper()=="HUMAN" else TAB[3] for n in names])
    ax.set_title(title)
    ax.set_ylabel("Count")
    for i, c in enumerate(counts):
        ax.text(i, c, str(int(c)), ha="center", va="bottom")
    fig.savefig(out_path)
    show_or_close(fig, show)

def plot_prob_hist(y: np.ndarray, p: np.ndarray, id2label: Dict[int,str], out_path: Path, title: str, show=False) -> None:
    if y is None or p is None:
        return
    y = y.astype(int); p = p.astype(float)
    fig, ax = plt.subplots(figsize=(6.8, 4))
    mask_pos = (y == 1)
    ax.hist(p[~mask_pos], bins=20, range=(0,1), alpha=0.6, label=id2label.get(0,"0"), color=TAB[0])
    ax.hist(p[mask_pos],  bins=20, range=(0,1), alpha=0.6, label=id2label.get(1,"1"), color=TAB[3])
    ax.set_title(title)
    ax.set_xlabel("Predicted P(class==1)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.savefig(out_path)
    show_or_close(fig, show)

def plot_reliability(y: np.ndarray, p: np.ndarray, out_path: Path, title_prefix: str, show=False) -> None:
    if y is None or p is None:
        return
    ece, brier, bin_acc, bin_conf = compute_ece_brier(y, p, RELIABILITY_BINS)
    centers = np.linspace(0, 1, RELIABILITY_BINS, endpoint=False) + (0.5/RELIABILITY_BINS)
    fig, ax = plt.subplots(figsize=(5.8, 5.3))
    ax.plot([0,1],[0,1], "--", color="gray", alpha=0.6, label="Perfect")
    ax.scatter(bin_conf, bin_acc, s=40, color=TAB[2], label="Bins")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title_prefix} — Reliability\nECE={ece:.3f}  Brier={brier:.3f}")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out_path)
    show_or_close(fig, show)

def plot_roc_pr(y: np.ndarray, p: np.ndarray, out_dir: Path, title_prefix: str, t_star: Optional[float], show=False) -> None:
    if y is None or p is None:
        return
    y = y.astype(int); p = p.astype(float)
    # ROC
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        fig, ax = plt.subplots(figsize=(6.2, 4.8))
        ax.plot(fpr, tpr, color=TAB[3], label=f"AUC={auc:.3f}")
        ax.plot([0,1],[0,1], "--", color="gray", alpha=0.5)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"{title_prefix} — ROC")
        ax.legend(frameon=False)
        fig.savefig(out_dir / f"{title_prefix.lower().split()[0]}_roc.png")
        show_or_close(fig, show)
    # PR
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(rec, prec, color=TAB[1], label=f"AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} — Precision-Recall")
    ax.legend(frameon=False)
    fig.savefig(out_dir / f"{title_prefix.lower().split()[0]}_pr.png")
    show_or_close(fig, show)

def plot_f1_vs_threshold(y: np.ndarray, p: np.ndarray, out_path: Path, title: str, t_star: Optional[float], show=False) -> None:
    if y is None or p is None:
        return
    y = y.astype(int); p = p.astype(float)
    ts = np.linspace(0,1,101)
    f1s = []
    for t in ts:
        yhat = (p >= t).astype(int)
        f1s.append(f1_score(y, yhat, average="macro", zero_division=0))
    f1s = np.array(f1s)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ts, f1s, color=TAB[4], label="F1 (macro)")
    if t_star is not None:
        f_at = f1s[int(round(t_star*100))]
        ax.axvline(t_star, color=TAB[2], linestyle="--", label=f"t*={t_star:.2f} (≈F1 {f_at:.3f})")
    ax.set_xlabel("Threshold t on P(class==1)")
    ax.set_ylabel("F1 (macro)")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.savefig(out_path)
    show_or_close(fig, show)

def draw_conf_matrix(cm: np.ndarray, class_names: List[str], title: str, out_path: Path, normalize: bool, show=False) -> None:
    cm_plot = cm.astype(float)
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_plot = cm_plot / cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.nan_to_num(cm_plot)
    fig, ax = plt.subplots(figsize=(4.8, 4.3))
    im = ax.imshow(cm_plot, cmap=CM_HEAT, vmin=0, vmax=(1.0 if normalize else cm_plot.max() or 1.0))
    ax.set_title(title)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    for i in range(2):
        for j in range(2):
            val = cm_plot[i,j]
            txt = f"{val:.2f}" if normalize else f"{int(cm[i,j])}"
            ax.text(j, i, txt, ha="center", va="center", color="white" if val > (0.6 if normalize else 0.6*cm_plot.max()) else "black", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path)
    show_or_close(fig, show)

def plot_confusions(y: np.ndarray, p: np.ndarray, yhat_argmax: np.ndarray, id2label: Dict[int,str], out_dir: Path, split_name: str, t_star: Optional[float], show=False) -> None:
    if y is None: return
    names = [id2label.get(0,"0"), id2label.get(1,"1")]
    # Argmax
    if yhat_argmax is not None:
        cm = confusion_matrix(y, yhat_argmax, labels=[0,1])
        draw_conf_matrix(cm, names, f"{split_name} — Confusion (ARGMAX)", out_dir / f"{split_name.lower()}_confusion_argmax.png", normalize=False, show=show)
        draw_conf_matrix(cm, names, f"{split_name} — Confusion (ARGMAX, normalized)", out_dir / f"{split_name.lower()}_confusion_argmax_norm.png", normalize=True, show=show)
    # Thresholded
    if p is not None:
        t = 0.5 if t_star is None else float(t_star)
        yhat_t = (p >= t).astype(int)
        cm_t = confusion_matrix(y, yhat_t, labels=[0,1])
        draw_conf_matrix(cm_t, names, f"{split_name} — Confusion (t={t:.2f})", out_dir / f"{split_name.lower()}_confusion_thresh.png", normalize=False, show=show)
        draw_conf_matrix(cm_t, names, f"{split_name} — Confusion (t={t:.2f}, normalized)", out_dir / f"{split_name.lower()}_confusion_thresh_norm.png", normalize=True, show=show)

def locate_token_cache(run_dir: Path) -> Optional[Path]:
    cfg = read_json(run_dir / "best" / "cfg.json")
    if not cfg:
        return None
    dataset_id = cfg.get("dataset_id")
    model_id = cfg.get("model_id")
    max_len = cfg.get("max_length")
    if not (dataset_id and model_id and max_len):
        return None
    key = f"{dataset_id}__{model_id}__L{max_len}"
    key = "".join([c if c.isalnum() or c in "._-" else "_" for c in key])
    cache_dir = RUNS_ROOT / "cache" / f"tok_{key}"
    return cache_dir if cache_dir.exists() else None

def load_lengths_and_texts(cache_dir: Path, split="validation") -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
    """Returns (token_lengths, raw_texts, labels) for split."""
    if hf_load_from_disk is None or cache_dir is None or not cache_dir.exists():
        return None, None, None
    try:
        ds = hf_load_from_disk(str(cache_dir))
        if split not in ds:
            return None, None, None
        split_ds = ds[split]
        # token lengths: len(input_ids)
        tok_lens = np.array([len(x) for x in split_ds["input_ids"]], dtype=int)
        raw_texts = split_ds["raw_text"] if "raw_text" in split_ds.column_names else None
        labels = np.array(split_ds["labels"]).astype(int)
        return tok_lens, (list(map(str, raw_texts)) if raw_texts is not None else None), labels
    except Exception:
        return None, None, None

def plot_prob_vs_length(y: np.ndarray, p: np.ndarray, lengths: Optional[np.ndarray], out_path: Path, title: str, show=False) -> None:
    if y is None or p is None or lengths is None:
        return
    y = y.astype(int); p = p.astype(float); L = lengths.astype(int)
    correct = ( ((p >= 0.5).astype(int) == y) )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(L[correct], p[correct], s=12, alpha=0.45, label="Correct", color=TAB[3])
    ax.scatter(L[~correct], p[~correct], s=18, alpha=0.65, label="Error", color=TAB[2], marker="x")
    ax.set_xlabel("Token length")
    ax.set_ylabel("P(class==1)")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out_path)
    show_or_close(fig, show)

def plot_len_hist_and_box(y: np.ndarray, lengths: Optional[np.ndarray], id2label: Dict[int,str], out_dir: Path, title_prefix: str, show=False) -> None:
    if y is None or lengths is None:
        return
    names = [id2label.get(0,"0"), id2label.get(1,"1")]
    fig, ax = plt.subplots(figsize=(6.5,4))
    ax.hist(lengths[y==0], bins=30, alpha=0.6, label=names[0], color=TAB[0])
    ax.hist(lengths[y==1], bins=30, alpha=0.6, label=names[1], color=TAB[3])
    ax.set_title(f"{title_prefix} — Token length hist by class")
    ax.set_xlabel("Token length"); ax.set_ylabel("Count")
    ax.legend(frameon=False)
    fig.savefig(out_dir / f"{title_prefix.lower().split()[0]}_len_hist.png")
    show_or_close(fig, show)

    # Boxplot
    fig2, ax2 = plt.subplots(figsize=(5.5,4))
    data = [lengths[y==0], lengths[y==1]]
    ax2.boxplot(data, labels=names, vert=True, patch_artist=True)
    ax2.set_title(f"{title_prefix} — Token length boxplot by class")
    ax2.set_ylabel("Token length")
    fig2.savefig(out_dir / f"{title_prefix.lower().split()[0]}_len_box_by_label.png")
    show_or_close(fig2, show)

def plot_top_overconfident_errors(y: np.ndarray, p: np.ndarray, texts: Optional[List[str]], id2label: Dict[int,str], out_path: Path, title: str, show=False) -> None:
    if y is None or p is None:
        return
    y = y.astype(int); p = p.astype(float)
    pred = (p >= 0.5).astype(int)
    err_idx = np.where(pred != y)[0]
    if err_idx.size == 0:
        return
    conf = np.maximum(p[err_idx], 1.0 - p[err_idx])  # confidence of wrong prediction
    order = np.argsort(-conf)
    top = err_idx[order][:min(20, err_idx.size)]
    top_conf = conf[order][:min(20, err_idx.size)]
    top_true = y[top]; top_pred = pred[top]

    labels = [f"{id2label.get(int(tt),'?')}→{id2label.get(int(pp),'?')}" for tt,pp in zip(top_true, top_pred)]
    y_pos = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.barh(y_pos, top_conf, color=TAB[2], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"#{i}  {lab}" for i, lab in zip(top.tolist(), labels)])
    ax.invert_yaxis()
    ax.set_xlabel("Confidence of wrong prediction")
    ax.set_title(title)
    fig.savefig(out_path)
    show_or_close(fig, show)

def plot_pca_tsne(repr_path: Path, out_dir: Path, title_prefix: str, show=False) -> None:
    if not repr_path.exists():
        return
    # Load pooled representations and labels
    data = np.load(repr_path, allow_pickle=True)
    X = data["X"]
    y = data["y"].astype(int)
    yhat = data["yhat"].astype(int)
    # PCA 2D
    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    Xp = pca.fit_transform(X)
    errs = (y != yhat)

    def scatter_with_errors(Z, fname, title):
        fig, ax = plt.subplots(figsize=(6.6, 5.6))
        # Classes
        ax.scatter(Z[y==0,0], Z[y==0,1], s=12, alpha=0.7, color=TAB[0], label="Class 0")
        ax.scatter(Z[y==1,0], Z[y==1,1], s=12, alpha=0.7, color=TAB[3], label="Class 1")
        # Errors overlay
        ax.scatter(Z[errs,0], Z[errs,1], s=30, alpha=0.9, facecolors='none', edgecolors=TAB[2], linewidths=1.2, label="Errors")
        ax.set_title(title)
        ax.legend(frameon=False, loc="best")
        ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
        fig.savefig(out_dir / fname)
        show_or_close(fig, show)

    scatter_with_errors(Xp, f"{title_prefix}_pca2.png", f"{title_prefix} — PCA (2D)")

    # t-SNE 2D (optional heavy)
    if MAKE_TSNE:
        tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=TSNE_N_ITER, init="pca", learning_rate="auto", random_state=42)
        Z = tsne.fit_transform(X)
        scatter_with_errors(Z, f"{title_prefix}_tsne2.png", f"{title_prefix} — t-SNE (2D)")

def compute_steps_per_sec(df: Optional[pd.DataFrame]) -> Optional[float]:
    if df is None or df.empty:
        return None
    d = df[df["event"] == "train_step"].copy()
    if d.empty:
        return None
    # Prefer "timestamp" column if present
    if "timestamp" in d.columns and not d["timestamp"].isna().all():
        t = d["timestamp"].astype(float).values
    else:
        # fallback to index monotonic (not ideal)
        t = np.arange(len(d)).astype(float)
    if t.size < 2:
        return None
    dt = float(t[-1] - t[0]) if t[-1] >= t[0] else float(t[0] - t[-1] + 1e-9)
    steps = int(len(d))
    if dt <= 0:
        return None
    return float(steps / dt)


# -------------------------
# Per-run processing
# -------------------------
def process_run(run_dir: Path, latest_display: bool) -> Dict[str, Any]:
    """
    Returns summary dict with key metrics for aggregation.
    """
    out_dir = PLOTS_ROOT / run_dir.name
    ensure_dir(out_dir)

    best_dir = run_dir / "best"
    preds_dir = best_dir / "preds"
    id2label = read_json(best_dir / "id2label.json") or {0: "HUMAN", 1: "AI"}
    # Normalize id2label keys to int
    id2label = {int(k): str(v) for k, v in id2label.items()}

    cfg = read_json(best_dir / "cfg.json") or {}
    thr = read_json(best_dir / "threshold.json") or {}
    t_star = thr.get("best_threshold")
    warm = read_json(run_dir / "warm_start_status.json") or {"used": False}

    metrics_df = read_metrics_csv(run_dir / "metrics.csv")

    # 1) training curves
    if metrics_df is not None:
        plot_train_loss(metrics_df, out_dir, show=latest_display)
        plot_lr(metrics_df, out_dir, show=latest_display)
        plot_val_metrics(metrics_df, out_dir, show=latest_display)

    # Load VAL arrays
    y_val, p_val, yhat_val = load_split_arrays(best_dir, "val")
    # per-split plots: VAL
    if y_val is not None:
        plot_class_balance(y_val, id2label, out_dir / "val_class_balance.png", "Validation — Class balance", show=latest_display)
    if y_val is not None and p_val is not None:
        plot_prob_hist(y_val, p_val, id2label, out_dir / "val_prob_hist.png", "Validation — Probability histogram", show=latest_display)
        plot_reliability(y_val, p_val, out_dir / "val_reliability.png", "Validation", show=latest_display)
        plot_roc_pr(y_val, p_val, out_dir, "Validation", t_star, show=latest_display)
        plot_f1_vs_threshold(y_val, p_val, out_dir / "val_f1_vs_threshold.png", "Validation — F1 vs threshold", t_star, show=latest_display)
    if y_val is not None:
        plot_confusions(y_val, p_val, yhat_val, id2label, out_dir, "Validation", t_star, show=latest_display)

    # Embeddings (PCA/t-SNE)
    repr_path = preds_dir / "val_repr_for_viz.npz"
    if repr_path.exists():
        plot_pca_tsne(repr_path, out_dir, "Validation", show=latest_display)

    # Token cache → lengths + texts
    cache_dir = locate_token_cache(run_dir)
    val_lengths, val_texts, val_labels_from_cache = load_lengths_and_texts(cache_dir, split="validation") if cache_dir else (None, None, None)
    if y_val is not None and p_val is not None:
        plot_prob_vs_length(y_val, p_val, val_lengths, out_dir / "val_prob_vs_length.png", "Validation — P(class==1) vs token length", show=latest_display)
        plot_len_hist_and_box(y_val, val_lengths, id2label, out_dir, "Validation", show=latest_display)
        plot_top_overconfident_errors(y_val, p_val, val_texts, id2label, out_dir / "val_top_overconfident_errors.png", "Validation — Top overconfident errors", show=latest_display)

    # TEST split (optional)
    y_test = p_test = yhat_test = None
    if INCLUDE_TEST:
        # Load arrays if present
        y_test, p_test, yhat_test = load_split_arrays(best_dir, "test")
        if y_test is not None:
            plot_class_balance(y_test, id2label, out_dir / "test_class_balance.png", "Test — Class balance", show=latest_display)
        if y_test is not None and p_test is not None:
            plot_prob_hist(y_test, p_test, id2label, out_dir / "test_prob_hist.png", "Test — Probability histogram", show=latest_display)
            plot_reliability(y_test, p_test, out_dir / "test_reliability.png", "Test", show=latest_display)
            plot_roc_pr(y_test, p_test, out_dir, "Test", t_star, show=latest_display)
            plot_f1_vs_threshold(y_test, p_test, out_dir / "test_f1_vs_threshold.png", "Test — F1 vs threshold (using val t*)", t_star, show=latest_display)
            plot_confusions(y_test, p_test, yhat_test, id2label, out_dir, "Test", t_star, show=latest_display)

        test_lengths, test_texts, _ = load_lengths_and_texts(cache_dir, split="test") if cache_dir else (None, None, None)
        if y_test is not None and p_test is not None:
            plot_prob_vs_length(y_test, p_test, test_lengths, out_dir / "test_prob_vs_length.png", "Test — P(class==1) vs token length", show=latest_display)
            plot_len_hist_and_box(y_test, test_lengths, id2label, out_dir, "Test", show=latest_display)

    # Summaries for aggregation
    # Read summarized metrics if present; else compute from arrays
    preds_metrics = read_json(preds_dir / "metrics.json") or {}
    preds_metrics_thr = read_json(preds_dir / "metrics_thresh.json") or {}
    # Compute if needed
    if (not preds_metrics) and (y_val is not None and p_val is not None and yhat_val is not None):
        try:
            acc = accuracy_score(y_val, yhat_val)
            f1m = f1_score(y_val, yhat_val, average="macro")
            auc = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) == 2 else float("nan")
        except Exception:
            acc=f1m=auc=float("nan")
        preds_metrics = {"acc":acc, "f1_macro":f1m, "auroc":auc}
    if (not preds_metrics_thr) and (y_val is not None and p_val is not None):
        t = 0.5 if t_star is None else float(t_star)
        yhat_t = (p_val >= t).astype(int)
        try:
            acc_t = accuracy_score(y_val, yhat_t)
            f1m_t = f1_score(y_val, yhat_t, average="macro")
        except Exception:
            acc_t=f1m_t=float("nan")
        preds_metrics_thr = {"threshold": t, "acc": acc_t, "f1_macro": f1m_t}

    steps_per_sec = compute_steps_per_sec(metrics_df)

    summary = {
        "run": run_dir.name,
        "model_id": cfg.get("model_id"),
        "dataset_id": cfg.get("dataset_id"),
        "epochs": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "grad_accum": cfg.get("grad_accum"),
        "precision": cfg.get("precision"),
        "use_lora": cfg.get("use_lora"),
        "best_threshold": t_star,
        "val_acc_argmax": preds_metrics.get("acc"),
        "val_f1_argmax": preds_metrics.get("f1_macro"),
        "val_auc": preds_metrics.get("auroc"),
        "val_acc_thresh": preds_metrics_thr.get("acc"),
        "val_f1_thresh": preds_metrics_thr.get("f1_macro"),
        "warm_start_used": bool(warm.get("used", False)),
        "steps_per_sec": steps_per_sec
    }
    return summary


# -------------------------
# Aggregates (across runs)
# -------------------------
def aggregate_across_runs(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    out_dir = PLOTS_ROOT / "_aggregate"
    ensure_dir(out_dir)

    # Save CSV summary
    header = ["run","model_id","dataset_id","epochs","batch_size","grad_accum","precision","use_lora","best_threshold",
              "val_acc_argmax","val_f1_argmax","val_auc","val_acc_thresh","val_f1_thresh","warm_start_used","steps_per_sec"]
    csv_path = out_dir / "agg_scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as cf:
        w = csv.DictWriter(cf, fieldnames=header)
        w.writeheader()
        for s in summaries:
            w.writerow({k: s.get(k) for k in header})

    # Bar charts for F1 (argmax & thresholded)
    runs = [s["run"] for s in summaries]
    f1a = np.array([s.get("val_f1_argmax") if s.get("val_f1_argmax")==s.get("val_f1_argmax") else np.nan for s in summaries], dtype=float)
    f1t = np.array([s.get("val_f1_thresh") if s.get("val_f1_thresh")==s.get("val_f1_thresh") else np.nan for s in summaries], dtype=float)

    idx = np.arange(len(runs))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(7.5, 0.8*len(runs)+4), 4.5))
    ax.bar(idx - width/2, f1a, width, label="Argmax", color=TAB[0], alpha=0.85)
    ax.bar(idx + width/2, f1t, width, label="Thresholded", color=TAB[3], alpha=0.85)
    ax.set_xticks(idx); ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Validation F1 across runs")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "agg_f1_val.png")
    show_or_close(fig, _in_notebook())

    # Steps/sec
    sps = [ (s.get("steps_per_sec") if s.get("steps_per_sec")==s.get("steps_per_sec") else np.nan) for s in summaries ]
    fig2, ax2 = plt.subplots(figsize=(max(7.5, 0.8*len(runs)+4), 4.3))
    ax2.bar(runs, sps, color=TAB[1], alpha=0.9)
    ax2.set_ylabel("Steps per second (proxy)")
    ax2.set_title("Training speed proxy")
    for i, v in enumerate(sps):
        if isinstance(v, float) and not (math.isnan(v)):
            ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    fig2.savefig(out_dir / "agg_steps_per_sec.png")
    show_or_close(fig2, _in_notebook())


# -------------------------
# API Baselines
# -------------------------
def process_api_baselines() -> None:
    base_dir = RUNS_ROOT / "api_baselines"
    if not base_dir.exists():
        return
    out_dir = PLOTS_ROOT / "api_baselines"
    ensure_dir(out_dir)

    mfiles = sorted(base_dir.glob("external_llm_metrics__*.json"))
    if not mfiles:
        return

    import math

    def _to_float_or_nan(x):
        if x is None:
            return float("nan")
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _to_int_or_zero(x):
        if x is None:
            return 0
        try:
            # handle strings / floats cleanly
            v = int(x) if (isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.strip().isdigit())) else int(float(x))
            return v
        except Exception:
            return 0

    names: List[str] = []
    accs: List[float] = []
    f1s: List[float] = []
    cov_used: List[int] = []
    cov_skipped: List[int] = []
    by_model_confmats: List[Optional[np.ndarray]] = []

    for jf in mfiles:
        data = read_json(jf) or {}
        name_raw = data.get("model_name")
        safe_name = (str(name_raw) if name_raw not in (None, "") else jf.stem.replace("external_llm_metrics__", ""))
        names.append(safe_name)

        accs.append(_to_float_or_nan(data.get("accuracy", float("nan"))))
        f1s.append(_to_float_or_nan(data.get("f1_macro", float("nan"))))
        cov_used.append(_to_int_or_zero(data.get("n_used_for_metrics", 0)))
        cov_skipped.append(_to_int_or_zero(data.get("n_skipped", 0)))

        # Try to load the corresponding preds CSV to build a confusion matrix (optional)
        csvp = base_dir / f"external_llm_preds__{safe_name.replace('/','_').replace(':','_')}.csv"
        if not csvp.exists():
            by_model_confmats.append(None)
            continue
        try:
            rows = []
            with open(csvp, "r", encoding="utf-8") as cf:
                reader = csv.DictReader(cf)
                for r in reader:
                    rows.append(r)
            # Keep rows that actually have a parsed prediction
            mask = np.array([ (r.get("pred_int") not in (None, "", "None")) for r in rows ], dtype=bool)
            if not mask.any():
                by_model_confmats.append(None)
                continue
            y_true = np.array([int(r["gold_int"]) for r in np.array(rows, dtype=object)[mask]])
            y_pred = np.array([int(r["pred_int"]) for r in np.array(rows, dtype=object)[mask]])
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            by_model_confmats.append(cm)
        except Exception:
            by_model_confmats.append(None)

    # Convert to numpy arrays for plotting; NaNs are fine for bar heights (bar won't render)
    accs_arr = np.array(accs, dtype=float)
    f1s_arr = np.array(f1s, dtype=float)
    used = np.array(cov_used, dtype=float)   # float avoids dtype issues if stacked with other floats
    skipped = np.array(cov_skipped, dtype=float)

    # --- Accuracy & F1 bars ---
    idx = np.arange(len(names))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * len(names) + 4), 4.6))
    ax.bar(idx - width/2, accs_arr, width, color=TAB[0], alpha=0.9, label="Accuracy")
    ax.bar(idx + width/2, f1s_arr,  width, color=TAB[3], alpha=0.9, label="F1 (macro)")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(n) for n in names], rotation=25, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("API baselines — Accuracy & F1")
    ax.legend(frameon=False)
    fig.savefig(out_dir / "api_acc_f1.png")
    show_or_close(fig, _in_notebook())

    # --- Coverage stacked bar (used vs skipped) ---
    fig2, ax2 = plt.subplots(figsize=(max(7.5, 0.9 * len(names) + 4), 4.3))
    ax2.bar([str(n) for n in names], used,    color=TAB[1], label="Used")
    ax2.bar([str(n) for n in names], skipped, bottom=used, color=TAB[2], label="Skipped")
    ax2.set_ylabel("# examples")
    ax2.set_title("API baselines — Coverage (used vs skipped)")
    ax2.legend(frameon=False)
    fig2.savefig(out_dir / "api_coverage.png")
    show_or_close(fig2, _in_notebook())

    # --- Confusion matrices per model (if available) ---
    for name, cm in zip(names, by_model_confmats):
        if cm is None:
            continue
        try:
            draw_conf_matrix(cm, ["HUMAN", "AI"],
                             f"{name} — Confusion",
                             out_dir / f"api_confusion__{str(name).replace('/','_').replace(':','_')}.png",
                             normalize=False,
                             show=_in_notebook())
        except Exception:
            # Skip quietly if anything goes wrong for a single model
            continue

# -------------------------
# Main
# -------------------------
def main():
    ensure_dir(PLOTS_ROOT)

    run_dirs = find_run_dirs()
    if not run_dirs:
        print("No run directories found under ./runs. Nothing to plot.")
        return

    # Sort by mtime; newest last
    run_dirs.sort(key=mtime_of_run)
    latest_run = run_dirs[-1]

    print("Discovered runs:")
    for d in run_dirs:
        mark = " (latest)" if d == latest_run else ""
        print(f"  - {d}{mark}")

    summaries = []
    for d in run_dirs:
        # Display plots inline only for latest run if in notebook (to avoid flooding)
        latest_display = _in_notebook() and (d == latest_run)
        try:
            s = process_run(d, latest_display=latest_display)
            summaries.append(s)
        except Exception as e:
            print(f"[WARN] Failed to process run {d}: {type(e).__name__}: {e}")

    # Aggregates
    try:
        aggregate_across_runs(summaries)
    except Exception as e:
        print(f"[WARN] Aggregate plotting failed: {type(e).__name__}: {e}")

    # API baselines
    try:
        process_api_baselines()
    except Exception as e:
        print(f"[WARN] API baseline plotting failed: {type(e).__name__}: {e}")

    print("\nAll plots saved under:", PLOTS_ROOT.resolve())
    if _in_notebook():
        print("Inline displays were shown for the latest run only.")


if __name__ == "__main__":
    main()
