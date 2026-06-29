# %%
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt

# %%
"""
Notebook-ready helper to extract embeddings from the MVTec Anomaly Detection
structure, now with timm backbones (e.g., DINOv2 `vit_base_patch14_dinov2.lvd142m`).

Install:
    %pip install -q numpy pandas pyarrow torch torchvision pillow timm

Usage:
    from mvtec_embeddings_notebook_helper import extract_embeddings_mvtec
    df = extract_embeddings_mvtec(
        root="/path/to/mvtec_anomaly_detection",
        backbone="vit_base_patch14_dinov2.lvd142m",  # or "resnet50", "vit_b_16", etc.
        pooling="auto",                               # 'auto'|'mean_patches'|'cls'|'gap'
        splits=("train","test"),
        categories=None,
        include_defects=True,
        flat_columns=True,
        output="mvtec_embeddings.parquet",
        batch_size=128,
        num_workers=4,
    )
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

# Optional timm import (only needed for non-torchvision backbones)
try:
    import timm
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False

# ------------------ Model Builders ------------------
class Identity(nn.Module):
    def forward(self, x):
        return x

class TimmFeatureExtractor(nn.Module):
    """
    Wraps a timm model to emit a single embedding vector per image,
    with sensible pooling across ViT/CNN families.
    """
    def __init__(self, model: nn.Module, pooling: str = "auto"):
        super().__init__()
        self.model = model
        self.pooling = pooling  # 'auto'|'mean_patches'|'cls'|'gap'

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Try forward_features first (richer control), fall back to forward()
        try:
            feats = self.model.forward_features(x)
        except Exception:
            feats = None

        if isinstance(feats, torch.Tensor):
            if feats.ndim == 3:  # [B, N, D] token sequence (ViT-like)
                if self.pooling == "cls":
                    vec = feats[:, 0]
                else:
                    # default to mean over non-CLS tokens if available
                    tokens = feats[:, 1:] if feats.size(1) > 1 else feats
                    if self.pooling in ("mean_patches", "auto"):
                        vec = tokens.mean(dim=1)
                    else:
                        # 'gap' doesn't apply to tokens; fall back to mean
                        vec = tokens.mean(dim=1)
            elif feats.ndim == 4:  # [B, C, H, W] feature map (CNN)
                # global average pooling
                vec = feats.mean(dim=[2, 3])
            else:
                # Unrecognized shape -> fall back to model(x) (pooled if num_classes=0)
                vec = self.model(x)
        else:
            # Some timm models (with num_classes=0) already return pooled feature
            vec = self.model(x)

        # Ensure 2D [B, D]
        if vec.ndim == 1:
            vec = vec.unsqueeze(0)
        return vec

def _build_model(backbone: str, device: str, pooling: str = "auto"):
    """
    Returns (feature_model, preprocess_transform, feature_dim or None, backend_tag)
    backend_tag: 'torchvision' or 'timm'
    """
    bb = (backbone or "").lower()

    # --- Torchvision baselines ---
    if bb in ("resnet50", "resnet"):
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = Identity()           # 2048-D penultimate features
        dim = 2048
        preprocess = weights.transforms()
        model.eval().to(device)
        return model, preprocess, dim, "torchvision"

    if bb in ("vit_b_16", "vit", "vit-b"):
        # Torchvision ViT path: exposes CLS features via heads.head Identity
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        model.heads.head = Identity()   # 768-D (CLS pre-logits)
        dim = 768
        preprocess = weights.transforms()
        model.eval().to(device)
        return model, preprocess, dim, "torchvision"

    # --- timm models (DINOv2, CLIP, ConvNeXt, Swin, etc.) ---
    if not _HAS_TIMM:
        raise ValueError(
            f"backbone='{backbone}' requires timm. Install: pip install timm"
        )

    # Create timm model; num_classes=0 => forward() returns a pooled embedding for many models
    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model.eval().to(device)

    # Build correct preprocessing from timm's pretrained cfg
    data_cfg = timm.data.resolve_data_config(model.pretrained_cfg, model=model)
    preprocess = timm.data.create_transform(**data_cfg)

    # Best-effort feature dim (many timm models expose num_features)
    dim = getattr(model, "num_features", None)

    # Wrap so we can handle ViT tokens vs CNN feature maps consistently
    feature_model = TimmFeatureExtractor(model, pooling=pooling).to(device)

    return feature_model, preprocess, dim, "timm"

# ------------------ MVTec Indexer ------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _iter_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    paths = []
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
    return paths

def build_mvtec_index(
    root: Path | str,
    splits: Tuple[str, ...] = ("train", "test"),
    categories: Optional[Iterable[str]] = None,
    include_defects: bool = True,
) -> pd.DataFrame:
    root = Path(root)
    if categories is None:
        categories = [d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    rows = []
    for cat in sorted(categories):
        croot = root / cat
        if not croot.exists():
            continue
        for split in splits:
            sroot = croot / split
            if not sroot.exists():
                continue
            if split == "train":
                good_dir = sroot / "good"
                for ip in _iter_images(good_dir):
                    rows.append({
                        "category": cat,
                        "split": split,
                        "defect_type": "good",
                        "path": str(ip),
                        "mask_path": None,
                        "label_idx": 0,
                        "label_name": "good",
                    })
            else:  # test split
                good_dir = sroot / "good"
                for ip in _iter_images(good_dir):
                    rows.append({
                        "category": cat,
                        "split": split,
                        "defect_type": "good",
                        "path": str(ip),
                        "mask_path": None,
                        "label_idx": 0,
                        "label_name": "good",
                    })
                if include_defects:
                    for d in sorted([d for d in sroot.iterdir() if d.is_dir() and d.name != "good"]):
                        defect = d.name
                        gt_dir = (croot / "ground_truth" / defect)
                        for ip in _iter_images(d):
                            base = ip.stem  # e.g., '000'
                            mask_candidate = None
                            if gt_dir.exists():
                                cand1 = gt_dir / f"{base}_mask.png"
                                cand2 = gt_dir / f"{base}.png"
                                if cand1.exists():
                                    mask_candidate = cand1
                                elif cand2.exists():
                                    mask_candidate = cand2
                                else:
                                    maybe = list(gt_dir.glob(f"{base}*"))
                                    if maybe:
                                        mask_candidate = maybe[0]
                            rows.append({
                                "category": cat,
                                "split": split,
                                "defect_type": defect,
                                "path": str(ip),
                                "mask_path": str(mask_candidate) if mask_candidate else None,
                                "label_idx": 1,
                                "label_name": defect,
                            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["category", "split", "defect_type", "path"]).reset_index(drop=True)
    return df

# ------------------ Dataset from Paths ------------------
class PathsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, preprocess):
        self.df = df.reset_index(drop=True)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        x = self.preprocess(img)
        y = int(row["label_idx"]) if "label_idx" in self.df.columns else 0
        return x, y

# ------------------ Main Extraction ------------------
def extract_embeddings_mvtec(
    root: str | Path,
    backbone: str = "resnet50",
    splits: Tuple[str, ...] = ("train", "test"),
    categories: Optional[Iterable[str]] = None,
    include_defects: bool = True,
    flat_columns: bool = True,
    output: Optional[str | Path] = "mvtec_embeddings.parquet",
    batch_size: int = 64,
    num_workers: int = 4,
    device: Optional[str] = None,
    verbose: bool = True,
    log_every: int = 20,
    pooling: str = "auto",  # NEW: 'auto'|'mean_patches'|'cls'|'gap' (timm ViTs support token pooling)
):
    """
    Traverse an MVTec-style dataset and extract embeddings for each image.

    Columns: category, split, defect_type, path, mask_path, label_idx, label_name,
             and either emb_0..emb_{D-1} or a single "embedding" list column.

    Notes:
      - Torchvision ViT ('vit_b_16') path exposes CLS features (like original).
        For mean-patch pooled ViT embeddings, supply a **timm ViT** backbone and set pooling.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feature_model, preprocess, dim, backend = _build_model(backbone, device, pooling=pooling)

    index_df = build_mvtec_index(root, splits=splits, categories=categories, include_defects=include_defects)
    if index_df.empty:
        raise ValueError("No images found. Check your 'root' path and structure.")

    ds = PathsDataset(index_df, preprocess)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=str(device).startswith("cuda")
    )

    n_total = len(ds)
    if verbose:
        print(f"Starting embedding extraction: {n_total:,} images | batch_size={batch_size} | device={device}")
        print(f"Backbone: {backbone} [{backend}] | pooling={pooling}")

    feats_list, labels_list = [], []
    n_done = 0
    t0 = time.time()
    with torch.inference_mode():
        for i, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=True)
            f = feature_model(xb)  # unified across torchvision/timm
            if f.dim() == 4:       # very unlikely; safety
                f = torch.flatten(f, 1)
            feats_list.append(f.cpu().numpy())
            labels_list.append(yb.cpu().numpy())

            n_done += xb.size(0)
            if verbose and (i % log_every == 0 or n_done >= n_total):
                elapsed = time.time() - t0
                ips = n_done / max(elapsed, 1e-6)
                remaining = max(n_total - n_done, 0)
                eta = remaining / max(ips, 1e-6)
                print(f"[{n_done:>7,}/{n_total:>7,}]  {ips:6.1f} img/s  ETA {eta:6.1f}s")

    feats = np.concatenate(feats_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    df = index_df.copy()
    if len(df) != len(feats):
        raise RuntimeError("Feature count mismatch with index length.")

    if flat_columns:
        emb_df = pd.DataFrame({f"emb_{i}": feats[:, i] for i in range(feats.shape[1])})
        df = pd.concat([df, emb_df], axis=1)
    else:
        df["embedding"] = list(feats)

    if output:
        out_path = Path(output)
        if out_path.suffix.lower() not in (".parquet", ".parq", ".csv"):
            out_path = out_path.with_suffix(".parquet")
        if out_path.suffix.lower() in (".parquet", ".parq"):
            df.to_parquet(out_path, index=False)
        else:
            dfd = df.copy()
            if not flat_columns:
                dfd["embedding"] = dfd["embedding"].apply(lambda v: ",".join(map(str, v)))
            dfd.to_csv(out_path, index=False)
        if verbose:
            total_t = time.time() - t0
            print(f"Saved {len(df):,} rows to {out_path} (dim={feats.shape[1]}) in {total_t:.1f}s")

    return df

# %%
df = extract_embeddings_mvtec(
    root=r"E:\Computational_Specialty_Projects\MAB_Enhanced_Classifier\mvtec_anomely_dataset",  # fix path & use raw str
    backbone="resnet50",
    pooling="gap",        # <-- key change for timm ViTs (better for kNN/anomaly)
    splits=("train",),             # train has only 'good'
    categories=["capsule"],
    include_defects=False,         # good-only as intended
    flat_columns=True,             # emb_0..emb_{D-1}
    output=None,                   # keep in-memory
    batch_size=128,
    num_workers=0,                 # fine on Windows
    device="cuda"                  # or "cpu" if no GPU
)
df.head(20)


# %%
emb_cols = df.filter(like="emb_").columns
is_constant = df[emb_cols].nunique(dropna=False) <= 1
df = df.drop(columns=is_constant[is_constant].index)
print(df.head(20))

# %%
df_ref = df

# %%
df_eval = extract_embeddings_mvtec(
    root="E:\Computational_Specialty_Projects\MAB_Enhanced_Classifier\mvtec_anomely_dataset",
    backbone="resnet50",           # or "vit_b_16"
    pooling="gap",        # <-- key change for timm ViTs (better for kNN/anomaly)
    splits=("test",),             # train has only 'good'
    categories=["capsule"],
    include_defects=True,         # good-only as intended
    flat_columns=True,             # emb_0..emb_{D-1}
    output=None,                   # keep in-memory
    batch_size=128,
    num_workers=0,                 # fine on Windows
    device="cuda"                  # or "cpu" if no GPU
    )    

df_eval.head(20)

# %%
emb_cols = [c for c in df.columns if c.startswith("emb_")]

# train/good → reference
X_ref_raw = df.loc[df["label_idx"]==0, emb_cols].to_numpy(dtype=np.float32)

# test (all) → eval
X_eval_raw = df_eval[emb_cols].to_numpy(dtype=np.float32)
y_eval     = df_eval["label_idx"].to_numpy(dtype=np.int64)

# %%
df_eval = df_eval[(df_eval["split"] == "test") & (df_eval["category"] == "capsule")].reset_index(drop=True)

# 2) pick embedding columns in correct numeric order
emb_cols = sorted([c for c in df_eval.columns if c.startswith("emb_")],
                  key=lambda x: int(x.split("_")[1]))
assert emb_cols, "No emb_* columns found in df_eval."

# 3) basic sanity: must have labels 0(good)/1(defect)
assert "label_idx" in df_eval.columns, "df_eval needs label_idx (0=good, 1=defect)."
print(df_eval["label_idx"].value_counts())

from joblib import dump, load  # or: from sklearn.decomposition import PCA
from sklearn.decomposition import PCA

# --- assume you've already FIT a PCA on train-good and saved it ---
pca = PCA(n_components=128, random_state=0).fit(X_ref_raw)   # done earlier
dump(pca, "E:\\pca_capsule.joblib")
# pca = load("pca_capsule.joblib")   # <-- load the already-fitted PCA

# 4) build matrices and drop any rows with NaNs (as you had)
X_eval = df_eval[emb_cols].to_numpy(dtype=np.float32)
y_eval = df_eval["label_idx"].to_numpy(np.int32)

mask_ok = ~np.isnan(X_eval).any(axis=1)
if not mask_ok.all():
    df_eval = df_eval[mask_ok].reset_index(drop=True)
    X_eval  = X_eval[mask_ok]
    y_eval  = y_eval[mask_ok]

# --- PCA transform (no refitting here) ---
# X_eval = pca.transform(X_eval).astype(np.float32)

# 5) L2-normalize embeddings (for cosine / FAISS IndexFlatIP)
def l2norm(a, eps=1e-9):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.clip(n, eps, None)

X_eval_n = l2norm(X_eval)

# 6) (optional) quick checks
print(f"Eval rows: {len(df_eval):,} | dims: {X_eval_n.shape[1]}")
assert np.isfinite(X_eval_n).all(), "Non-finite values after normalization."

# %%
# === PREP: build matrices from df_ref (train/good) and df_eval (test all) ===
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.decomposition import PCA
# from sklearn.externals import joblib  # if sklearn <0.23; else use: from joblib import dump, load
from joblib import dump, load

# 1) ensure the splits are what we expect
df_ref  = df_ref[(df_ref["split"] == "train") & (df_ref["label_idx"] == 0)].reset_index(drop=True)
df_eval = df_eval[(df_eval["split"] == "test")].reset_index(drop=True)

# 2) pick embedding columns ONCE (numeric order) and use for both frames
emb_cols = sorted([c for c in df_eval.columns if c.startswith("emb_")],
                  key=lambda x: int(x.split("_")[1]))
assert emb_cols, "No emb_* columns found in df_eval."

# 3) ensure df_ref has those columns too (intersection just in case)
emb_cols = [c for c in emb_cols if c in df_ref.columns]
assert emb_cols, "df_ref is missing emb_* columns; did you embed train/good?"

# 4) build matrices
X_ref  = df_ref[emb_cols].to_numpy(np.float32)
X_eval = df_eval[emb_cols].to_numpy(np.float32)
y_eval = df_eval["label_idx"].to_numpy(np.int32)   # 0=good, 1=defect

# 5) drop NaN rows (if any)
m_ref  = ~np.isnan(X_ref).any(axis=1)
m_eval = ~np.isnan(X_eval).any(axis=1)
X_ref, df_ref  = X_ref[m_ref], df_ref[m_ref].reset_index(drop=True)
X_eval, y_eval, df_eval = X_eval[m_eval], y_eval[m_eval], df_eval[m_eval].reset_index(drop=True)

# 6) L2-normalize (needed for cosine via inner product)
def l2norm(a, eps=1e-9):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    return a / np.maximum(n, eps)

# X_ref_raw: train-good embeddings [N_ref, D]
# X_eval_raw: test embeddings [N_eval, D]

# 1) Fit PCA on train-good only
# d_red = 128
# # or choose by explained variance (see below)
# pca = PCA(n_components=d_red, svd_solver="full", whiten=False, random_state=0)
# pca.fit(X_ref_raw)
pca = load(r"E:\pca_capsule.joblib")
# 2) Transform both sets
X_ref_n  = l2norm(pca.transform(X_ref_raw).astype("float32"))
X_eval_n = l2norm(pca.transform(X_eval_raw).astype("float32"))



print(f"Prepared: X_ref={X_ref_n.shape}, X_eval={X_eval_n.shape}, defects={y_eval.sum()} / total={len(y_eval)}")
# dump(pca, "E:\\pca_mvtec_capsule.joblib")

# %%
import numpy as np
import faiss
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# --- 1) build kNN index on train/good (this is the "training" step) ---
d = X_ref_n.shape[1]
index = faiss.IndexFlatIP(d)          # inner product on L2-normalized vectors = cosine sim
index.add(X_ref_n.astype(np.float32)) # add reference (good) embeddings

# pick k (use <= len(X_ref_n))min(5, len(X_ref_n))
k = min(5, len(X_ref_n))

# --- 2) score eval samples by mean distance to k-NN (higher = more anomalous) ---
S, I = index.search(X_eval_n.astype(np.float32), k)  # S: similarities
scores = (1.0 - S).mean(axis=1).astype(np.float32)   # cosine distance

# --- 3) pick threshold by cost (quick pass; ideally use a small calibration set) ---
def best_threshold_by_cost(y, s, c_fp=1.0, c_fn=50.0):
    idx = np.argsort(s)
    cuts = (s[idx][:-1] + s[idx][1:]) / 2.0
    best_t, best_cost = None, float("inf")
    good = (y == 0); defect = (y == 1)
    for t in np.r_[-np.inf, cuts, np.inf]:
        yhat = (s >= t).astype(int)
        cost = c_fp*np.mean(good & (yhat==1)) + c_fn*np.mean(defect & (yhat==0))
        if cost < best_cost:
            best_t, best_cost = t, cost
    return best_t, best_cost

C_FP, C_FN = 2, 500
t_star, min_cost = best_threshold_by_cost(y_eval, scores, C_FP, C_FN)
y_hat = (scores >= t_star).astype(int)

# # --- 4) report metrics ---
# tn, fp, fn, tp = confusion_matrix(y_eval, y_hat, labels=[0,1]).ravel()
# recall = tp / max(tp+fn, 1)
# fpr    = fp / max(fp+tn, 1)
# auroc  = roc_auc_score(y_eval, scores)
# aupr   = average_precision_score(y_eval, scores)

# print(f"k={k}  |  threshold={t_star:.6f}  |  min_cost={min_cost:.4f}  (C_FP={C_FP}, C_FN={C_FN})")
# print(f"Recall(defect)={recall:.3f}  FPR(good)={fpr:.3f}  AUROC={auroc:.3f}  AUPR={aupr:.3f}")
# print(f"Confusion: TN={tn}  FP={fp}  FN={fn}  TP={tp}")

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def safe_metrics(y_true, scores):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    out = {}

    # Confusion matrix works even for empty if we pass labels, but handle 0-len explicitly
    if y_true.size == 0 or scores.size == 0:
        out.update(dict(auroc=np.nan, aupr=np.nan, recall=np.nan, fpr=np.nan,
                        tn=0, fp=0, fn=0, tp=0))
        return out

    tn, fp, fn, tp = confusion_matrix(y_true, (scores >= 0).astype(int), labels=[0,1]).ravel()
    out.update(dict(tn=tn, fp=fp, fn=fn, tp=tp))

    # Need at least one pos and one neg for AUROC/AUPR
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    if n_pos > 0 and n_neg > 0:
        out["auroc"] = roc_auc_score(y_true, scores)
        out["aupr"]  = average_precision_score(y_true, scores)
        out["recall"] = tp / max(tp + fn, 1)
        out["fpr"]    = fp / max(fp + tn, 1)
    else:
        out.update(dict(auroc=np.nan, aupr=np.nan,
                        recall=(tp / max(tp + fn, 1) if (tp+fn)>0 else np.nan),
                        fpr=(fp / max(fp + tn, 1) if (fp+tn)>0 else np.nan)))
    return out



m = safe_metrics(y_eval, scores)
print(m)




