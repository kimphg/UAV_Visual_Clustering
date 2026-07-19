#!/usr/bin/env python3
"""
General Embedding Evaluation GUI.
Supports SUES-200 and University-1652 datasets.
"""

from collections import Counter
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")

# Cap BLAS thread pools BEFORE numpy/faiss/torch/sklearn first touch their
# backends (these libs read these env vars once at first use, not
# reconfigurable later). This machine's dual Xeon E5-2680 v4 exposes 56
# threads; OpenBLAS defaults to using all of them, and the main process is
# not the only one requesting that many — every DataLoader worker process
# (default up to 8, see loader_kwargs) spins up its OWN full-width OpenBLAS
# thread pool too. Under concurrent load (geo re-rank's K-means codebook fit
# + the eval's DataLoader workers) this multiplication can exhaust the OS's
# thread-local buffer allocations, surfacing as OpenBLAS's own
# "Memory allocation still failed after 10 retries, giving up" — a distinct
# failure mode from a plain CUDA/system OOM. setdefault() so an explicit
# environment override from outside this process still wins.
for _blas_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_blas_var, "8")

import faiss
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget, QHeaderView, QScrollArea, QGridLayout,
)
from torch.utils.data import DataLoader, Dataset

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from model import SwinEmbedding, backbone_img_size
from dataset import make_eval_transform, loader_kwargs

HERE = Path(__file__).parent
SETTINGS_PATH = HERE / "general_eval_gui_settings.json"
FEAT_CACHE_ROOT = HERE / "gallery_feat_cache"

DATASET_SUES200 = "sues200"
DATASET_U1652   = "u1652"
DATASET_DENSEUAV = "denseuav"
DATASET_G4L     = "game4loc_visloc"

DEFAULT_ROOTS = {
    DATASET_SUES200: r"D:\UAV_DATASET\SUES-200-512x512",
    DATASET_U1652:   r"D:\UAV_DATASET\university-1652\University-Release",
    DATASET_DENSEUAV: r"D:\UAV_DATASET\DenseUAV\DenseUAV",  # eval code looks for test/ subdir
    DATASET_G4L:     r"D:\UAV_DATASET\VisLoc",
}
DEFAULT_HISTORIES = {
    DATASET_SUES200: "sues200_eval_history.json",
    DATASET_U1652:   "u1652_eval_history.json",
    DATASET_DENSEUAV: "denseuav_eval_history.json",
    DATASET_G4L:     "g4l_visloc_eval_history.json",
}

SUES_ALTITUDES = [150, 200, 250, 300]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_SAMPLES_DISPLAY = 24


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(label: str, ckpt_path: Path, path_strs: list, backbone: str) -> Path:
    FEAT_CACHE_ROOT.mkdir(exist_ok=True)
    h = hashlib.md5(("".join(path_strs) + backbone).encode()).hexdigest()[:12]
    return FEAT_CACHE_ROOT / f"{label}_{ckpt_path.stem}_{h}.pt"


def _try_load_cache(cache_path: Path, ckpt_path: Path):
    """Return (emb_ndarray, scales_or_None) if valid cache, else None."""
    if not cache_path.exists():
        return None
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if abs(cached.get("checkpoint_mtime", 0) - os.path.getmtime(ckpt_path)) > 1.0:
            return None
        return cached["emb"], cached.get("scales")
    except Exception:
        return None


def _save_cache(cache_path: Path, ckpt_path: Path, emb: np.ndarray, scales=None):
    try:
        data = {"emb": emb, "checkpoint_mtime": os.path.getmtime(ckpt_path)}
        if scales is not None:
            data["scales"] = scales
        torch.save(data, cache_path)
    except Exception:
        pass


def _try_load_generic_cache(cache_path: Path, ckpt_path: Path):
    """Like _try_load_cache but for arbitrary payloads (e.g. quantized
    descriptor IDs + codebook) rather than the fixed emb/scales shape."""
    if not cache_path.exists():
        return None
    try:
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if abs(cached.get("checkpoint_mtime", 0) - os.path.getmtime(ckpt_path)) > 1.0:
            return None
        return cached["data"]
    except Exception:
        return None


def _save_generic_cache(cache_path: Path, ckpt_path: Path, data):
    try:
        torch.save({"data": data, "checkpoint_mtime": os.path.getmtime(ckpt_path)}, cache_path)
    except Exception:
        pass


# ── Shared dataset / metric helpers ──────────────────────────────────────────

class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def quick_eval_denseuav(model, device, train_root, batch_size: int = 64,
                        direction: str = "d2s", stop_event=None, log=print):
    """Lightweight DenseUAV D2S/S2D eval usable mid-training (no Qt, no GUI).

    Embeds the full test gallery+query set with the CURRENT model weights (same
    encode_features → encode_cluster_from_features interface trainer.py uses),
    scores with exact-ID hits over the full confusion gallery (official
    protocol), and returns {r1, r5, r10, ap, n_queries} or None if skipped/
    stopped. Does not touch eval history/cache — for a periodic in-training
    signal only, since training loss alone can plateau while retrieval quality
    keeps improving (see project memory).
    """
    from model import backbone_img_size
    from dataset import make_eval_transform, loader_kwargs
    from torch.utils.data import DataLoader

    train_root = Path(train_root)
    test_dir = train_root.parent / "test"
    if not test_dir.exists():
        log(f"[Quick Eval] DenseUAV test dir not found: {test_dir} — skipping.")
        return None

    if direction == "d2s":
        query_dir, gallery_dir = test_dir / "query_drone", test_dir / "gallery_satellite"
    else:
        query_dir, gallery_dir = test_dir / "query_satellite", test_dir / "gallery_drone"

    def _collect(d):
        paths, ids = [], []
        for loc_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            paths.extend(imgs); ids.extend([loc_dir.name] * len(imgs))
        return paths, ids

    gallery_paths, gallery_ids = _collect(gallery_dir)
    query_paths, query_ids = _collect(query_dir)
    if not gallery_paths or not query_paths:
        log("[Quick Eval] DenseUAV test data not found — skipping.")
        return None

    img_size = backbone_img_size(getattr(model, "backbone_name", "swin_b"))
    transform = make_eval_transform(img_size=img_size)
    was_training = model.training
    model.eval()

    def _embed(paths):
        # pin_memory=False: this loader is spun up ad hoc from a background
        # thread while the main training loop's own pinned-memory DataLoader may
        # still be active in the same process — two concurrent pin-memory
        # threads race on CUDA's pinned allocator and raise cudaErrorAlreadyMapped
        # ("resource already mapped"). Unpinned transfer is slightly slower but
        # this is a periodic, not performance-critical, pass.
        loader = DataLoader(ImageDataset(paths, transform), batch_size=batch_size,
                            shuffle=False, **loader_kwargs(device.type, pin_memory=False))
        chunks = []
        with torch.inference_mode():
            for imgs, _ in loader:
                if stop_event is not None and stop_event.is_set():
                    return None
                feats = model.encode_features(imgs.to(device))
                emb = model.encode_cluster_from_features(feats)
                chunks.append(emb.cpu().float().numpy())
        return np.concatenate(chunks, axis=0) if chunks else None

    gal_emb = _embed(gallery_paths)
    qry_emb = _embed(query_paths) if gal_emb is not None else None
    if was_training:
        model.train()
    if gal_emb is None or qry_emb is None:
        return None

    gal_norm = gal_emb / (np.linalg.norm(gal_emb, axis=1, keepdims=True) + 1e-8)
    qry_norm = qry_emb / (np.linalg.norm(qry_emb, axis=1, keepdims=True) + 1e-8)
    sim = qry_norm @ gal_norm.T
    recalls, ap, n_q = _recall_and_ap(sim, query_ids, gallery_ids, ks=(1, 5, 10),
                                      exclude_junk=False)  # full confusion gallery
    result = {"r1": recalls[1], "r5": recalls[5], "r10": recalls[10], "ap": ap, "n_queries": n_q}
    log(f"[Quick Eval] DenseUAV {direction}  (n={n_q}):  "
        f"R@1={result['r1']*100:.2f}%  R@5={result['r5']*100:.2f}%  "
        f"R@10={result['r10']*100:.2f}%  mAP={result['ap']*100:.2f}%")
    return result


def pil_to_pixmap(img: Image.Image, size: int = 128) -> QPixmap:
    img = img.resize((size, size), Image.LANCZOS)
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _semantic_palette(k: int) -> np.ndarray:
    """Deterministic [K,3] uint8 palette for cluster-ID maps: ceil(K/4) evenly
    spaced hues x 4 (saturation, value) bands. Verified minimum pairwise RGB
    distance ~31 at K=64 (golden-ratio hue stepping was tried first and produced
    near-duplicate colours, min distance 4 — don't switch back without checking).
    Same ID always gets the same colour across runs/images — that's what makes
    two images' maps visually comparable. Beyond K~100 colours are inherently
    too numerous to fully tell apart; the maps stay useful for comparing
    LAYOUTS, just not for naming individual clusters by eye."""
    import colorsys
    n_h = -(-k // 4)
    combos = [(0.90, 0.95), (0.55, 0.95), (0.90, 0.60), (0.62, 0.70)]
    cols = []
    for i in range(k):
        h = (i % n_h) / n_h
        s, v = combos[(i // n_h) % 4]
        cols.append(colorsys.hsv_to_rgb(h, s, v))
    return (np.asarray(cols) * 255).astype(np.uint8)


def _ids_to_semantic_image(ids: np.ndarray, palette: np.ndarray) -> Image.Image:
    """[H,W] cluster-id map -> colour-coded PIL image (1 px per grid cell;
    upscale at display time with NEAREST so cell boundaries stay crisp)."""
    return Image.fromarray(palette[ids], "RGB")


def _semantic_pixmap(ids: np.ndarray, palette: np.ndarray, size: int = 140) -> QPixmap:
    """Colour-coded cluster-ID map as a pixmap, nearest-upscaled (crisp cells)."""
    img = _ids_to_semantic_image(ids, palette).resize((size, size), Image.NEAREST)
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


def _recall_and_ap(sim: np.ndarray, query_ids: list, gallery_ids: list,
                   ks=(1, 5, 10), exclude_junk: bool = True):
    """Compute R@K and mAP. Supports multi-relevant gallery (S→D).

    exclude_junk=True: gallery images whose ID never appears in query_ids are
    treated as distractors (junk) and excluded from the ranking before scoring —
    matching the University-1652 / Game4Loc evaluation protocol.
    exclude_junk=False: full confusion gallery kept (official DenseUAV protocol).

    Returns: ({k: recall_float}, mAP_float, n_queries_int)
    """
    gallery_arr = np.array(gallery_ids)
    query_id_set = set(query_ids)
    # junk: gallery images that have no matching query location
    junk_mask = (np.array([gid not in query_id_set for gid in gallery_ids])
                 if exclude_junk else np.zeros(len(gallery_ids), dtype=bool))

    top_idx = np.argsort(-sim, axis=1)
    r_counts = {k: 0 for k in ks}
    aps = []
    for q_i, qid in enumerate(query_ids):
        ranked_idx = top_idx[q_i]
        ranked_idx = ranked_idx[~junk_mask[ranked_idx]]   # remove distractors
        ranked = gallery_arr[ranked_idx]
        hits = (ranked == qid)
        for k in ks:
            if hits[:k].any():
                r_counts[k] += 1
        n_rel = int(hits.sum())
        if n_rel > 0:
            cum = np.cumsum(hits).astype(float)
            ranks = np.arange(1, len(hits) + 1, dtype=float)
            ap = float((cum / ranks * hits).sum()) / n_rel
            aps.append(ap)
    n_q = len(query_ids)
    recalls = {k: r_counts[k] / n_q for k in ks}
    return recalls, float(np.mean(aps)) if aps else 0.0, n_q


def _recall_and_ap_with_gps(sim: np.ndarray, query_ids: list, gallery_ids: list,
                             query_gps: dict, gallery_gps: dict, ks=(1, 5, 10),
                             gps_threshold: float = 100.0):
    """Compute R@K and mAP for DenseUAV using GPS distance-based hits.

    Locations within gps_threshold meters (default 100m) are treated as hits.
    Gallery images without GPS data are excluded from ranking.

    Returns: ({k: recall_float}, mAP_float, n_queries_int)
    """
    def gps_distance(lat1, lon1, lat2, lon2):
        dlat_m = abs(lat2 - lat1) * 111_320.0
        dlon_m = abs(lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
        return math.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)

    gallery_arr = np.array(gallery_ids)
    top_idx = np.argsort(-sim, axis=1)
    r_counts = {k: 0 for k in ks}
    aps = []

    for q_i, qid in enumerate(query_ids):
        if qid not in query_gps:
            continue  # Skip queries without GPS
        q_lat, q_lon = query_gps[qid]
        ranked_idx = top_idx[q_i]
        hits = np.zeros(len(ranked_idx), dtype=bool)

        for rank, gal_idx in enumerate(ranked_idx):
            gal_id = gallery_arr[gal_idx]
            if gal_id not in gallery_gps:
                continue
            g_lat, g_lon = gallery_gps[gal_id]
            dist = gps_distance(q_lat, q_lon, g_lat, g_lon)
            if dist < gps_threshold:
                hits[rank] = True

        for k in ks:
            if hits[:k].any():
                r_counts[k] += 1
        n_rel = int(hits.sum())
        if n_rel > 0:
            cum = np.cumsum(hits).astype(float)
            ranks = np.arange(1, len(hits) + 1, dtype=float)
            ap = float((cum / ranks * hits).sum()) / n_rel
            aps.append(ap)

    n_q = len([qid for qid in query_ids if qid in query_gps])
    recalls = {k: r_counts[k] / n_q for k in ks} if n_q > 0 else {k: 0.0 for k in ks}
    return recalls, float(np.mean(aps)) if aps else 0.0, n_q


def whiten_embeddings(gal_emb: np.ndarray, qry_emb: np.ndarray,
                      shrinkage: float = 0.1, eig_floor: float = 1e-3):
    """ZCA (Mahalanobis) whitening of embeddings using the gallery covariance.

    Cosine similarity treats every embedding dimension as equally informative, so a
    feature value that is common across the gallery counts as much as a rare one.
    Mahalanobis distance fixes this: d²(x,y) = (x−y)ᵀ Σ⁻¹ (x−y). Whitening the
    features x' = (x−μ) W with W = V·diag(λ^-1/2)·Vᵀ from the gallery covariance
    Σ = V·diag(λ)·Vᵀ makes ‖x'−y'‖² equal to that Mahalanobis distance, so
    high-variance (common) directions are down-weighted and low-variance (rare)
    directions up-weighted. Ranking then uses cosine on the whitened features.

    Regularization prevents low-variance directions from exploding under λ^-1/2
    (the classic Mahalanobis instability):
      • shrinkage — blend Σ toward a scaled identity (Ledoit–Wolf style),
      • eig_floor — clamp eigenvalues to eig_floor·λ_max before inversion.

    Returns (qry_white, gal_white), both L2-normalized float32.
    """
    g = gal_emb.astype(np.float64)
    mu = g.mean(axis=0, keepdims=True)
    gc = g - mu
    n, D = gc.shape
    cov = (gc.T @ gc) / max(1, n - 1)                    # D×D, PSD symmetric
    mean_var = np.trace(cov) / D
    cov = (1.0 - shrinkage) * cov + shrinkage * mean_var * np.eye(D)
    evals, evecs = np.linalg.eigh(cov)                   # ascending eigenvalues
    evals = np.clip(evals, eig_floor * float(evals.max()), None)
    W = (evecs * (evals ** -0.5)) @ evecs.T              # symmetric ZCA whitening
    gw = gc @ W
    qw = (qry_emb.astype(np.float64) - mu) @ W
    gw /= (np.linalg.norm(gw, axis=1, keepdims=True) + 1e-8)
    qw /= (np.linalg.norm(qw, axis=1, keepdims=True) + 1e-8)
    return qw.astype(np.float32), gw.astype(np.float32)


def write_denseuav_kml(path: Path, gt_points: dict, pred_rows: list):
    """2-D map of eval results for Google Earth / GIS tools.

    gt_points:  {location_id: (lat, lon)} — ground-truth query locations.
    pred_rows:  [(query_name, qid, pred_id, (glat, glon) | None,
                  (plat, plon) | None, exact_hit: bool, dist_m | None), ...]
    Markers: GT = blue, correct prediction = green, wrong prediction = red;
    red lines connect each wrong prediction to its ground truth.
    """
    def pm(name, lat, lon, style, desc=""):
        return (f"<Placemark><name>{name}</name>{desc}"
                f"<styleUrl>#{style}</styleUrl><Point><coordinates>"
                f"{lon:.7f},{lat:.7f},0</coordinates></Point></Placemark>")

    gt_xml, ok_xml, bad_xml, line_xml = [], [], [], []
    for qid, (lat, lon) in sorted(gt_points.items()):
        gt_xml.append(pm(f"GT {qid}", lat, lon, "gt"))
    for qname, qid, pred_id, g, p, exact, dist in pred_rows:
        if p is None:
            continue
        desc = (f"<description>query {qname} | gt {qid} | pred {pred_id}"
                + (f" | err {dist:.0f} m" if dist is not None else "")
                + "</description>")
        if exact:
            ok_xml.append(pm(f"OK {qid}", p[0], p[1], "ok", desc))
        else:
            bad_xml.append(pm(f"{qid}->{pred_id}", p[0], p[1], "bad", desc))
            if g is not None:
                line_xml.append(
                    f"<Placemark><styleUrl>#err</styleUrl><LineString><coordinates>"
                    f"{g[1]:.7f},{g[0]:.7f},0 {p[1]:.7f},{p[0]:.7f},0"
                    f"</coordinates></LineString></Placemark>")

    def folder(name, items, visible=1):
        return (f"<Folder><name>{name} ({len(items)})</name>"
                f"<visibility>{visible}</visibility>" + "".join(items) + "</Folder>")

    icon = "http://maps.google.com/mapfiles/kml/paddle/{}.png"
    kml = (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
        f"<name>DenseUAV eval — {datetime.now().strftime('%Y-%m-%d %H:%M')}</name>"
        f"<Style id='gt'><IconStyle><scale>0.7</scale><Icon><href>{icon.format('blu-circle')}</href></Icon></IconStyle></Style>"
        f"<Style id='ok'><IconStyle><scale>0.6</scale><Icon><href>{icon.format('grn-circle')}</href></Icon></IconStyle></Style>"
        f"<Style id='bad'><IconStyle><scale>0.8</scale><Icon><href>{icon.format('red-circle')}</href></Icon></IconStyle></Style>"
        "<Style id='err'><LineStyle><color>7f0000ff</color><width>2</width></LineStyle></Style>"
        + folder("Ground truth", gt_xml)
        + folder("Predictions — correct", ok_xml, visible=0)
        + folder("Predictions — wrong", bad_xml)
        + folder("Error lines", line_xml)
        + "</Document></kml>")
    path.write_text(kml, encoding="utf-8")
    return len(gt_xml), len(ok_xml), len(bad_xml)


# ── Main application ──────────────────────────────────────────────────────────

class GeneralEvalApp(QWidget):
    log_message      = pyqtSignal(str)
    progress_changed = pyqtSignal(int)
    results_ready    = pyqtSignal(dict, str)   # (results_dict, dataset_type)
    samples_ready    = pyqtSignal(list)
    failed_ready     = pyqtSignal(list)         # list of (q_path, pred_path, gt_path, caption)
    eval_finished    = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("General Embedding Evaluation")
        self.resize(1020, 780)
        self.stop_event  = threading.Event()
        self.eval_thread = None

        # ── Settings ──────────────────────────────────────────────────────
        config = QGroupBox("Settings")
        form = QFormLayout()

        # Dataset selector
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("SUES-200  (multi-altitude drone ↔ satellite)", DATASET_SUES200)
        self.dataset_combo.addItem("University-1652  (building geo-localization)",  DATASET_U1652)
        self.dataset_combo.addItem("DenseUAV  (multi-scale drone → satellite)", DATASET_DENSEUAV)
        self.dataset_combo.addItem("Game4Loc VisLoc  (their exact protocol)", DATASET_G4L)
        self.dataset_combo.currentIndexChanged.connect(self._on_dataset_changed)

        # Shared by University-1652 and DenseUAV eval — see the form.addRow
        # site below for why this lives outside any dataset-specific group box.
        self.u1652_direction = QComboBox()
        self.u1652_direction.addItem("Drone → Satellite  (navigation,  Q=drone  G=satellite)", "d2s")
        self.u1652_direction.addItem("Satellite → Drone  (localization, Q=satellite G=drone)", "s2d")

        # Dataset root — single line edit, auto-populated on dataset change
        self.root_input = QLineEdit(DEFAULT_ROOTS[DATASET_SUES200])
        btn_root = QPushButton("Browse")
        btn_root.clicked.connect(lambda: self._browse_dir(self.root_input))
        root_row = QHBoxLayout()
        root_row.addWidget(self.root_input)
        root_row.addWidget(btn_root)

        # Checkpoint
        self.checkpoint_input = QLineEdit("checkpoints/latest_swin_b.pt")
        btn_ckpt = QPushButton("Browse")
        btn_ckpt.clicked.connect(lambda: self._browse_file(
            self.checkpoint_input, "Checkpoint (*.pt *.pth);;All files (*.*)"))
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.checkpoint_input)
        ckpt_row.addWidget(btn_ckpt)

        # Backbone
        self.backbone_input = QComboBox()
        for lbl, key in [
            ("Auto (read from checkpoint)",          "auto"),
            ("Swin-T  (28 M, 224×224)",              "swin_t"),
            ("Swin-B  (88 M, 384×384)",              "swin_b"),
            ("Swin-B no head  (88 M)",               "swin_b_nohead"),
            ("SwinV2-B  (88 M, 384×384)",            "swinv2_b"),
            ("ConvNeXt-B  (89 M, 384×384)",          "convnext_b"),
            ("ConvNeXt-T  (28 M, 384×384)",          "convnext_t"),
            ("ViT-B/16  (86 M, 224×224)",            "vit_b"),
            ("DINOv2-B/14  (86 M, 224×224)",         "dinov2_b"),
            ("Distilled Swin-B  (88 M, 224×224, no head)", "distill_swin_t"),
            ("Routed cluster-expert  (Swin-B, 384×384)", "routed_swin_b"),
        ]:
            self.backbone_input.addItem(lbl, key)

        # Device — enumerate every CUDA GPU (e.g. run eval on a second card
        # while the primary one is busy training).
        self.device_input = QComboBox()
        self.device_input.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            for di in range(torch.cuda.device_count()):
                self.device_input.addItem(
                    f"CUDA:{di} — {torch.cuda.get_device_name(di)}", f"cuda:{di}")
            self.device_input.setCurrentIndex(1)

        # Batch size
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 512)
        self.batch_size_input.setValue(96)

        # Label + history (common)
        self.label_input = QLineEdit()
        self.label_input.setPlaceholderText("optional run label")

        self.history_input = QLineEdit(DEFAULT_HISTORIES[DATASET_SUES200])
        history_browse = QPushButton("…")
        history_browse.setFixedWidth(28)
        history_browse.clicked.connect(self._browse_history)
        history_row_w = QWidget()
        _hl = QHBoxLayout(history_row_w)
        _hl.setContentsMargins(0, 0, 0, 0)
        _hl.addWidget(self.history_input)
        _hl.addWidget(history_browse)

        self.no_history_input = QCheckBox("Skip saving to history")

        form.addRow("Dataset",      self.dataset_combo)
        form.addRow("Dataset root", QWidget())  # spacer row replaced by root_row below
        form.addRow("Checkpoint",   QWidget())
        form.addRow("Backbone",     self.backbone_input)
        form.addRow("Device",       self.device_input)
        form.addRow("Batch size",   self.batch_size_input)
        # Shared query/gallery direction (U-1652 is genuinely bidirectional;
        # DenseUAV's official benchmark ships drone->satellite data ONLY —
        # no gallery_drone/query_satellite exist at all). Lives here, in the
        # always-visible main form, rather than inside the U-1652-only group
        # box: it used to be a child of that box, so on DenseUAV it was
        # simultaneously in effect (its value was read by _eval_denseuav)
        # AND invisible (parent box hidden) — no way to see or fix a stale
        # "Satellite -> Drone" selection left over from a previous U-1652
        # session, which crashed DenseUAV eval looking for a directory that
        # was never going to exist.
        form.addRow("Eval direction", self.u1652_direction)
        form.addRow("Label",        self.label_input)
        form.addRow("History file", history_row_w)
        form.addRow("",             self.no_history_input)

        # Replace placeholder rows with proper widgets
        form.removeRow(1); form.insertRow(1, "Dataset root", _make_hbox(self.root_input, btn_root))
        form.removeRow(2); form.insertRow(2, "Checkpoint",   _make_hbox(self.checkpoint_input, btn_ckpt))

        config.setLayout(form)

        # ── SUES-200 options group ─────────────────────────────────────────
        self.sues_options = QGroupBox("SUES-200 Options")
        sues_form = QFormLayout()

        self.altitude_input = QComboBox()
        self.altitude_input.addItem("All altitudes", None)
        for alt in SUES_ALTITUDES:
            self.altitude_input.addItem(f"{alt} m", alt)

        self.scale_penalty_input = QDoubleSpinBox()
        self.scale_penalty_input.setRange(0.0, 5.0)
        self.scale_penalty_input.setSingleStep(0.1)
        self.scale_penalty_input.setValue(0.0)
        self.scale_penalty_input.setDecimals(2)
        self.scale_penalty_input.setToolTip(
            "Subtract weight × |log_scale_q − log_scale_gallery| from similarity.\n"
            "0 = disabled."
        )
        self.sues_gallery_root_input = QLineEdit()
        self.sues_gallery_root_input.setToolTip(
            "Official SUES-200 protocol (github.com/Reza-Zhu/SUES-200-Benchmark):\n"
            "the gallery must include satellite tiles from ALL 200 locations\n"
            "(120 train + 80 test) as confusion/distractor tiles — only the\n"
            "80 test-location DRONE images (from Dataset root) are queries.\n"
            "Leave empty to fall back to Dataset root's own satellite-view\n"
            "(only the test-split locations — an easier, non-comparable\n"
            "80-way eval, not the official 200-way one).\n"
            "Point this at the FULL unsplit dataset's satellite-view folder,\n"
            "e.g. D:/UAV_DATASET/SUES-200-512x512/satellite-view.")
        sues_gallery_browse = QPushButton("Browse…")
        sues_gallery_browse.clicked.connect(lambda: self._browse_dir(self.sues_gallery_root_input))
        sues_form.addRow("Altitude filter",    self.altitude_input)
        sues_form.addRow("Scale penalty λ",    self.scale_penalty_input)
        sues_form.addRow("Gallery satellite root (all 200 locs)",
                         _make_hbox(self.sues_gallery_root_input, sues_gallery_browse))
        self.sues_options.setLayout(sues_form)

        # ── DenseUAV options group ──────────────────────────────────────────
        self.denseuav_options = QGroupBox("DenseUAV Options")
        denseuav_form = QFormLayout()

        self.denseuav_scale_penalty_width = QDoubleSpinBox()
        self.denseuav_scale_penalty_width.setRange(0.0, 2.0)
        self.denseuav_scale_penalty_width.setSingleStep(0.05)
        self.denseuav_scale_penalty_width.setValue(0.0)   # disabled by default — scale
        # penalty consistently hurts DenseUAV d2s ranking (satellite gallery scales are
        # not a reliable altitude cue). Set >0 only to experiment.
        self.denseuav_scale_penalty_width.setDecimals(2)
        self.denseuav_scale_penalty_width.setToolTip(
            "Scale penalty width for soft tanh-based scale matching.\n"
            "0 = disabled (recommended for DenseUAV — the penalty hurts d2s recall).\n"
            "Smaller = sharper penalty, Larger = softer penalty.\n"
            "Formula: penalty = 1 - tanh(|scale_diff| / width)"
        )
        denseuav_form.addRow("Scale penalty width", self.denseuav_scale_penalty_width)

        self.denseuav_mahalanobis_input = QCheckBox("Mahalanobis (whitened) similarity")
        self.denseuav_mahalanobis_input.setChecked(False)
        self.denseuav_mahalanobis_input.setToolTip(
            "Whiten embeddings by the gallery inverse-covariance before ranking, so\n"
            "common feature directions are down-weighted and rare ones up-weighted\n"
            "(cosine ignores how rare a feature value is; Mahalanobis does not)."
        )
        self.denseuav_maha_shrinkage_input = QDoubleSpinBox()
        self.denseuav_maha_shrinkage_input.setRange(0.0, 1.0)
        self.denseuav_maha_shrinkage_input.setSingleStep(0.05)
        self.denseuav_maha_shrinkage_input.setValue(0.10)
        self.denseuav_maha_shrinkage_input.setDecimals(2)
        self.denseuav_maha_shrinkage_input.setToolTip(
            "Covariance shrinkage toward a scaled identity (Ledoit–Wolf style).\n"
            "0 = pure Mahalanobis (can be unstable in low-variance directions);\n"
            "higher = closer to plain cosine. 0.1 is a safe default."
        )
        denseuav_form.addRow("Mahalanobis whitening", self.denseuav_mahalanobis_input)
        denseuav_form.addRow("  ↳ covariance shrinkage", self.denseuav_maha_shrinkage_input)

        self.denseuav_routed_topk = QSpinBox()
        self.denseuav_routed_topk.setRange(1, 64)
        self.denseuav_routed_topk.setValue(3)
        self.denseuav_routed_topk.setToolTip(
            "Routed model only: number of nearest coarse clusters each query searches.\n"
            "A query only competes against gallery tiles in these clusters; a true match\n"
            "in an unsearched cluster counts as a miss (honest coarse-routing recall).")
        denseuav_form.addRow("Routed: search top-k clusters", self.denseuav_routed_topk)

        # Re-rank controls live in their OWN group box (not denseuav_options)
        # so they're available for SUES-200 and University-1652 evals too.
        # Widget names keep the historical "denseuav_" prefix — renaming them
        # would churn every call site for zero behavior change.
        rerank_form = QFormLayout()
        self.denseuav_georank_input = QCheckBox("Geometric re-rank top-K (DenseUAV only)")
        self.denseuav_georank_input.setChecked(False)
        self.denseuav_georank_input.setToolTip(
            "Take each query's top-K coarse candidates, fit a query<->candidate\n"
            "2D transform from quantized spatial descriptors (geo_verify.py), and\n"
            "use it as a FILTER, not a full re-rank: candidates the fit ACCEPTS\n"
            "keep their original coarse-similarity order; candidates it REJECTS\n"
            "(shift too large, or unfittable) are demoted below every accepted\n"
            "candidate. Produces one additional result set: 'georank_filtered'.\n"
            "(An earlier full-rerank-by-confidence design cratered R@1 81.6%->30%\n"
            "on real DenseUAV data despite R@5/R@10 staying close to baseline — the\n"
            "true match usually stayed in the top-20 but got demoted by a\n"
            "confidence score that saturates near 1.0 too easily on stage 3's\n"
            "coarse 12x12 grid. Filtering trusts coarse similarity's ordering and\n"
            "only uses geometry to catch clear false positives. On real DenseUAV\n"
            "data, stage 3 ALSO showed near-identical accept rates for true\n"
            "matches vs wrong locations (98.6% vs 96.5%) — essentially no real\n"
            "discriminating power — motivating the Stage option below.)\n"
            "Gallery footprint per image comes from the scale head's own\n"
            "prediction (exp(log_scale)) — no manual entry needed. Adds real\n"
            "compute: spatial descriptor extraction for the whole gallery+queries,\n"
            "one shared K-means codebook fit, and a RANSAC fit per (query, top-K\n"
            "candidate) pair. Unavailable for routed models (no per-image scale\n"
            "prediction in that mode).")
        rerank_form.addRow("", self.denseuav_georank_input)

        self.denseuav_vqrank_input = QCheckBox("VQ map re-rank top-K (semantic ID maps)")
        self.denseuav_vqrank_input.setChecked(False)
        self.denseuav_vqrank_input.setToolTip(
            "Re-rank each query's coarse top-K by comparing quantized semantic\n"
            "cell-ID maps (stage-N spatial descriptor -> HxW codebook IDs,\n"
            "144 bytes/image at stage 3): score = coarse_sim + alpha *\n"
            "map_score, where map_score is mean cell-wise agreement maximised\n"
            "over the candidate's 4 cardinal rotations (no flips — mirrored\n"
            "layouts score LOW on purpose). Independent of the geometric\n"
            "re-rank above (RANSAC transform fitting); both can run in one\n"
            "eval and report separate result rows. Motivation: R@10 ~99%\n"
            "means the GT is nearly always in the top-10 — this re-orders\n"
            "within it, so R@10 is unchanged by construction.\n"
            "Validate offline first with vq_rerank_test.py before trusting\n"
            "a given alpha/K.\n"
            "Game4Loc VisLoc: the gallery is a HIERARCHICAL multi-zoom tile\n"
            "pyramid (e.g. zoom 5/6/7 tiles side by side, each covering a\n"
            "different real-world footprint from the same 384x384 input) —\n"
            "a cell-wise semantic map comparison is only meaningful between\n"
            "tiles of the SAME zoom. The re-rank therefore only re-scores a\n"
            "query's top-K candidates that share the coarse top-1's zoom\n"
            "level; other-zoom candidates keep their coarse score untouched\n"
            "rather than being compared against a mismatched physical scale.")
        rerank_form.addRow("", self.denseuav_vqrank_input)

        # Shared re-rank controls — both re-rankers verify the same top-K list
        # from the same quantized descriptors, so stage/top-K/codebook-K are one
        # set of inputs (and one extraction cache), not two.
        _rr_shared_form = QFormLayout()
        _rr_shared_form.setContentsMargins(0, 0, 0, 0)
        self.denseuav_rerank_stage_input = QComboBox()
        self.denseuav_rerank_stage_input.addItem("Stage 2 (24x24; 512-d Swin-B / 384-d ConvNeXt-T)", 2)
        self.denseuav_rerank_stage_input.addItem("Stage 3 (12x12; 1024-d Swin-B / 768-d ConvNeXt-T)", 3)
        self.denseuav_rerank_stage_input.setCurrentIndex(1)   # Stage 3
        self.denseuav_rerank_stage_input.setToolTip(
            "Which backbone stage's spatial grid to use (both re-rankers). For\n"
            "the geometric re-rank, stage 3 showed almost no separation between\n"
            "true matches and wrong locations on real DenseUAV data (see its\n"
            "checkbox tooltip) — its 12x12=144-location grid may be too coarse\n"
            "to carry discriminating layout; stage 2 has 4x the locations (576).\n"
            "Changing this changes the cache key, so switching stages triggers\n"
            "a fresh extraction, not a cache hit.")
        _rr_shared_form.addRow("  ↳ stage", self.denseuav_rerank_stage_input)

        self.denseuav_rerank_topk_input = QSpinBox()
        self.denseuav_rerank_topk_input.setRange(2, 200)
        self.denseuav_rerank_topk_input.setValue(10)
        _rr_shared_form.addRow("  ↳ top-K candidates", self.denseuav_rerank_topk_input)

        self.denseuav_rerank_k_input = QSpinBox()
        self.denseuav_rerank_k_input.setRange(4, 65535)
        self.denseuav_rerank_k_input.setValue(64)
        self.denseuav_rerank_k_input.setToolTip(
            "Shared cell-codebook size K, fit ONCE from a sample of gallery\n"
            "tokens and reused for every image — the production-realistic\n"
            "compression path. Both re-rankers share the cached codebook and\n"
            "quantized ID maps for a given stage+K.")
        _rr_shared_form.addRow("  ↳ quantization codebook K", self.denseuav_rerank_k_input)
        self._rerank_shared_w = QWidget()
        self._rerank_shared_w.setLayout(_rr_shared_form)
        rerank_form.addRow(self._rerank_shared_w)

        # Geometric-re-rank-only controls (hidden unless its checkbox is on).
        _geo_sub_form = QFormLayout()
        _geo_sub_form.setContentsMargins(0, 0, 0, 0)
        self.denseuav_georank_estimate_scale_input = QCheckBox(
            "Estimate scale (variable-altitude datasets, e.g. DenseUAV)")
        self.denseuav_georank_estimate_scale_input.setChecked(False)
        self.denseuav_georank_estimate_scale_input.setToolTip(
            "ON: fit rotation+scale+translation (similarity transform).\n"
            "OFF: fit rotation+translation only, scale locked to 1 (rigid\n"
            "transform) — fewer free parameters, so a low-residual fit through\n"
            "coincidentally-matching tokens is somewhat less trivially achieved.\n"
            "Tested manually via the Geo Verify Test (Full Gallery) tab with this\n"
            "OFF, sim_threshold=0.40, RANSAC threshold=5.0.")
        _geo_sub_form.addRow("  ↳ estimate scale", self.denseuav_georank_estimate_scale_input)

        self.denseuav_georank_sim_input = QDoubleSpinBox()
        self.denseuav_georank_sim_input.setRange(0.0, 1.0)
        self.denseuav_georank_sim_input.setSingleStep(0.05)
        self.denseuav_georank_sim_input.setDecimals(2)
        self.denseuav_georank_sim_input.setValue(0.4)
        self.denseuav_georank_sim_input.setToolTip(
            "0.4 tested manually via the Geo Verify Test (Full Gallery) tab\n"
            "(stage 3, estimate scale OFF, RANSAC threshold=5.0). If you switch\n"
            "the Stage option above to 2, this hasn't been separately tuned.")
        _geo_sub_form.addRow("  ↳ token match sim. threshold", self.denseuav_georank_sim_input)

        self.denseuav_georank_ransac_input = QDoubleSpinBox()
        self.denseuav_georank_ransac_input.setRange(0.1, 20.0)
        self.denseuav_georank_ransac_input.setSingleStep(0.5)
        self.denseuav_georank_ransac_input.setValue(5.0)
        self.denseuav_georank_ransac_input.setToolTip(
            "RANSAC inlier threshold, in GRID units. 5.0 tested manually via the\n"
            "Geo Verify Test (Full Gallery) tab (stage 3, estimate scale OFF,\n"
            "sim_threshold=0.4). Auto-escalates (doubles, up to 4x) if too few\n"
            "inliers, since the real accept/reject gate is the reject threshold\n"
            "below.")
        _geo_sub_form.addRow("  ↳ RANSAC inlier threshold", self.denseuav_georank_ransac_input)

        self.denseuav_georank_ransac_iters_input = QSpinBox()
        self.denseuav_georank_ransac_iters_input.setRange(8, 2000)
        self.denseuav_georank_ransac_iters_input.setValue(64)
        self.denseuav_georank_ransac_iters_input.setToolTip(
            "RANSAC trials per fit attempt (2-point minimal sample). 64 is already\n"
            "generous per standard RANSAC theory (k=ln(0.01)/ln(1-w^2): ~50 trials\n"
            "suffice even at a pessimistic 30% inlier ratio, ~16 at 50%). This is\n"
            "the dominant per-candidate cost at eval scale: total RANSAC trials run\n"
            "= n_queries * top_K * this value * (up to 5x with auto-escalation) —\n"
            "lower it if the 97%->100% verification stage is taking too long,\n"
            "raise it only if you suspect fits are missing genuinely good matches.")
        _geo_sub_form.addRow("  ↳ RANSAC iterations per attempt", self.denseuav_georank_ransac_iters_input)

        self.denseuav_georank_min_inliers_input = QSpinBox()
        self.denseuav_georank_min_inliers_input.setRange(2, 500)
        self.denseuav_georank_min_inliers_input.setValue(10)
        self.denseuav_georank_min_inliers_input.setToolTip(
            "The fitted transform has 4 degrees of freedom (rotation, scale, 2D\n"
            "translation), so a value of 4 here is trivially satisfiable — a\n"
            "minimal-sample RANSAC fit is ALWAYS a near-perfect fit by\n"
            "construction, giving zero real discriminating power. On a real\n"
            "DenseUAV eval, min_inliers=4 produced a 100% accept rate (0 out of\n"
            "46620 candidate checks rejected) — the filter was a complete no-op.\n"
            "Needs to be well above 4 to mean anything; watch the log's inlier-\n"
            "count/n_matches distribution AND the TRUE-match-vs-WRONG-location\n"
            "breakdown after a run to see if this default is achievable on your\n"
            "data (a coarser grid has fewer tokens to match in the first place)\n"
            "and whether it actually separates true matches from wrong ones.")
        _geo_sub_form.addRow("  ↳ min. inliers to accept", self.denseuav_georank_min_inliers_input)

        self.denseuav_georank_check_reflection_input = QCheckBox(
            "Reject mirror-ambiguous matches")
        self.denseuav_georank_check_reflection_input.setChecked(True)
        self.denseuav_georank_check_reflection_input.setToolTip(
            "A genuine match's token positions are almost never ALSO well-\n"
            "explained by their own mirror image. Rejects candidates where a\n"
            "reflected transform explains a comparable fraction of the same\n"
            "matched points as the accepted fit — e.g. a bilaterally-symmetric\n"
            "building complex matched against its own flipped duplicate, which\n"
            "otherwise looks like a confident, small-shift, well-supported fit.\n"
            "Forcing only proper rotations does NOT reliably catch this alone:\n"
            "a 2-point minimal RANSAC sample can't determine handedness.")
        self.denseuav_georank_reflection_ratio_input = QDoubleSpinBox()
        self.denseuav_georank_reflection_ratio_input.setRange(0.1, 2.0)
        self.denseuav_georank_reflection_ratio_input.setSingleStep(0.05)
        self.denseuav_georank_reflection_ratio_input.setDecimals(2)
        self.denseuav_georank_reflection_ratio_input.setValue(0.7)
        _geo_sub_form.addRow("", _make_hbox(self.denseuav_georank_check_reflection_input,
                                            QLabel("ratio:"),
                                            self.denseuav_georank_reflection_ratio_input))

        self.denseuav_georank_scale_min_input = QDoubleSpinBox()
        self.denseuav_georank_scale_min_input.setRange(0.05, 10.0)
        self.denseuav_georank_scale_min_input.setSingleStep(0.05)
        self.denseuav_georank_scale_min_input.setDecimals(2)
        self.denseuav_georank_scale_min_input.setValue(0.7)
        self.denseuav_georank_scale_max_input = QDoubleSpinBox()
        self.denseuav_georank_scale_max_input.setRange(0.05, 10.0)
        self.denseuav_georank_scale_max_input.setSingleStep(0.05)
        self.denseuav_georank_scale_max_input.setDecimals(2)
        self.denseuav_georank_scale_max_input.setValue(1.5)
        _geo_sub_form.addRow("  ↳ scale range", _make_hbox(
            QLabel("min:"), self.denseuav_georank_scale_min_input,
            QLabel("max:"), self.denseuav_georank_scale_max_input))

        self.denseuav_georank_reject_input = QDoubleSpinBox()
        self.denseuav_georank_reject_input.setRange(0.1, 10000.0)
        self.denseuav_georank_reject_input.setValue(20.0)
        self.denseuav_georank_reject_input.setSuffix(" m")
        _geo_sub_form.addRow("  ↳ reject threshold", self.denseuav_georank_reject_input)

        self._georank_sub_w = QWidget()
        self._georank_sub_w.setLayout(_geo_sub_form)
        rerank_form.addRow(self._georank_sub_w)

        # VQ-map-re-rank-only controls (hidden unless its checkbox is on).
        _vq_sub_form = QFormLayout()
        _vq_sub_form.setContentsMargins(0, 0, 0, 0)
        self.denseuav_vqrank_score_input = QComboBox()
        self.denseuav_vqrank_score_input.addItem(
            "soft (centroid-cosine partial credit)", "soft")
        self.denseuav_vqrank_score_input.addItem("exact (ID match only)", "exact")
        _vq_sub_form.addRow("  ↳ map score", self.denseuav_vqrank_score_input)

        self.denseuav_vqrank_alpha_input = QDoubleSpinBox()
        self.denseuav_vqrank_alpha_input.setRange(0.01, 100.0)
        self.denseuav_vqrank_alpha_input.setSingleStep(0.05)
        self.denseuav_vqrank_alpha_input.setDecimals(2)
        self.denseuav_vqrank_alpha_input.setValue(0.5)
        self.denseuav_vqrank_alpha_input.setToolTip(
            "Blend weight: final = coarse_sim + alpha * map_score (within the\n"
            "top-K only). Small alpha = gentle tie-breaking; large alpha\n"
            "approaches ranking by map score alone. Sweep offline with\n"
            "vq_rerank_test.py --alphas before trusting a value here.")
        _vq_sub_form.addRow("  ↳ blend alpha", self.denseuav_vqrank_alpha_input)
        self._vqrank_sub_w = QWidget()
        self._vqrank_sub_w.setLayout(_vq_sub_form)
        rerank_form.addRow(self._vqrank_sub_w)

        def _rerank_vis():
            geo_on = self.denseuav_georank_input.isChecked()
            vq_on = self.denseuav_vqrank_input.isChecked()
            self._rerank_shared_w.setVisible(geo_on or vq_on)
            self._georank_sub_w.setVisible(geo_on)
            self._vqrank_sub_w.setVisible(vq_on)
        self.denseuav_georank_input.toggled.connect(lambda _: _rerank_vis())
        self.denseuav_vqrank_input.toggled.connect(lambda _: _rerank_vis())
        _rerank_vis()

        self.denseuav_options.setLayout(denseuav_form)
        self.denseuav_options.setVisible(False)

        self.rerank_options = QGroupBox("Top-K Re-ranking")
        self.rerank_options.setLayout(rerank_form)
        self.rerank_options.setVisible(False)

        # ── Game4Loc VisLoc options ────────────────────────────────────────
        self.g4l_options = QGroupBox("Game4Loc VisLoc Options")
        g4l_form = QFormLayout()
        self.g4l_json_input = QLineEdit("same-area-drone2sate-test.json")
        self.g4l_json_input.setToolTip(
            "Test pairs JSON (relative to the dataset root). The eval follows\n"
            "Game4Loc's eval_visloc.py EXACTLY: per-query gallery restricted\n"
            "to the query's own map area, positives = pair_pos list only,\n"
            "queries without pos entries dropped from the denominator,\n"
            "sklearn AP, SDM@K = rank-weighted exp(-0.001*meters). A second\n"
            "row reports the pos_semipos variant (all queries, semi-positives\n"
            "counted as positives) for reference.\n"
            "Their published VisLoc same-area fine-tuned reference:\n"
            "R@1=80.20  R@5=96.53  AP=87.83  SDM@3=0.8546 (paper Table 4).")
        g4l_form.addRow("Test JSON", self.g4l_json_input)
        self.g4l_options.setLayout(g4l_form)
        self.g4l_options.setVisible(False)

        # ── Buttons + progress ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Evaluation")
        self.run_btn.clicked.connect(self.start_eval)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_eval)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.progress, stretch=1)

        # ── Tabs ──────────────────────────────────────────────────────────
        tabs = QTabWidget()

        # Settings tab (dataset config + run controls, kept in a tab to save
        # vertical space for the result/preview tabs below)
        settings_w = QWidget()
        settings_v = QVBoxLayout()
        settings_v.addWidget(config)
        settings_v.addWidget(self.sues_options)
        settings_v.addWidget(self.denseuav_options)
        settings_v.addWidget(self.rerank_options)
        settings_v.addWidget(self.g4l_options)
        settings_v.addStretch()
        settings_w.setLayout(settings_v)
        tabs.addTab(settings_w, "Settings")

        # Results tab
        results_w = QWidget()
        rv = QVBoxLayout()
        self.summary_label = QLabel("No results yet.")
        self.summary_label.setAlignment(Qt.AlignCenter)
        sf = QFont(); sf.setPointSize(13); sf.setBold(True)
        self.summary_label.setFont(sf)
        self.results_table = QTableWidget()
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rv.addWidget(self.summary_label)
        rv.addWidget(self.results_table)
        results_w.setLayout(rv)
        tabs.addTab(results_w, "Results")

        # Samples tab
        samples_w = QWidget()
        sv = QVBoxLayout()
        self.samples_hint = QLabel("Query (left) → Top-1 match (right). Green = correct.")
        self.samples_hint.setWordWrap(True)
        self.samples_scroll = QScrollArea()
        self.samples_scroll.setWidgetResizable(True)
        self.samples_container = QWidget()
        self.samples_grid = QGridLayout(self.samples_container)
        self.samples_scroll.setWidget(self.samples_container)
        sv.addWidget(self.samples_hint)
        sv.addWidget(self.samples_scroll)
        samples_w.setLayout(sv)
        tabs.addTab(samples_w, "Sample Matches")

        # Failed Matches tab
        failed_w = QWidget()
        fv = QVBoxLayout()
        fv.setContentsMargins(6, 6, 6, 6)
        self.failed_label = QLabel("Run evaluation to see failed matches.")
        self.failed_label.setStyleSheet("font-weight: bold; padding: 4px 0;")
        self._failed_container = QWidget()
        self._failed_vbox = QVBoxLayout()
        self._failed_vbox.setContentsMargins(0, 0, 0, 0)
        self._failed_vbox.setSpacing(6)
        self._failed_vbox.addStretch()
        self._failed_container.setLayout(self._failed_vbox)
        failed_scroll = QScrollArea()
        failed_scroll.setWidgetResizable(True)
        failed_scroll.setWidget(self._failed_container)
        fv.addWidget(self.failed_label)
        fv.addWidget(failed_scroll, stretch=1)
        failed_w.setLayout(fv)
        tabs.addTab(failed_w, "Failed Matches")

        # Geo Verify Test tabs — quick single-pair test of the geometric
        # verification module (geo_verify.py) before wiring it into the full
        # eval pipeline (full-gallery caching, per-image footprint metadata).
        # One tab per backbone stage (2: 512-d/24x24, 3: 1024-d/12x12 — both
        # come from the SAME backbone forward pass, see model.py's
        # encode_spatial_features(stage=...), so there's no extra inference cost
        # to having both tabs available.
        self.geo_tabs = {}
        for stage, grid_hint, chan_hint in ((2, 24, 512), (3, 12, 1024)):
            tab_w = self._build_geo_verify_tab(stage, grid_hint, chan_hint)
            tabs.addTab(tab_w, f"Geo Verify Test (Stage {stage})")

        fullscan_w = self._build_geo_fullscan_tab()
        tabs.addTab(fullscan_w, "Geo Verify Test (Full Gallery)")

        # Log tab
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Courier New", 9))
        tabs.addTab(self.log, "Log")

        self.main_tabs = tabs

        # ── Main layout ────────────────────────────────────────────────────
        main = QVBoxLayout()
        main.addLayout(btn_row)
        main.addWidget(tabs, stretch=1)
        self.setLayout(main)

        # Signals
        self.log_message.connect(self.log.append)
        self.progress_changed.connect(self.progress.setValue)
        self.results_ready.connect(self.on_results_ready)
        self.samples_ready.connect(self.on_samples_ready)
        self.failed_ready.connect(self.on_failed_ready)
        self.eval_finished.connect(self.on_eval_finished)

        self._load_settings()
        # Explicit sync: if the restored dataset landed on index 0, the
        # currentIndexChanged signal never fired and the per-dataset option
        # groups (incl. rerank_options) would keep their construction-time
        # visibility.
        self._on_dataset_changed()

    # ── Dataset switching ──────────────────────────────────────────────────

    def _on_dataset_changed(self, _=None):
        ds = self.dataset_combo.currentData()
        self.sues_options.setVisible(ds == DATASET_SUES200)
        self.denseuav_options.setVisible(ds == DATASET_DENSEUAV)
        self.rerank_options.setVisible(
            ds in (DATASET_DENSEUAV, DATASET_SUES200, DATASET_U1652, DATASET_G4L))
        self.g4l_options.setVisible(ds == DATASET_G4L)
        # Geo re-rank needs per-image scale predictions + a metric reject
        # threshold — only wired for DenseUAV. VQ re-rank is dataset-agnostic.
        self.denseuav_georank_input.setEnabled(ds == DATASET_DENSEUAV)
        self.u1652_direction.setVisible(ds in (DATASET_U1652, DATASET_DENSEUAV))
        # DenseUAV's official benchmark ships drone->satellite data ONLY —
        # no gallery_drone/query_satellite directories exist at all, so
        # "Satellite -> Drone" isn't just untested here, it's structurally
        # impossible (confirmed: FileNotFoundError trying to open a
        # directory that was never going to exist). Force back to d2s and
        # disable the combo rather than let a leftover U-1652 selection
        # silently carry over and crash at file-open time.
        if ds == DATASET_DENSEUAV:
            d2s_idx = self.u1652_direction.findData("d2s")
            if d2s_idx >= 0:
                self.u1652_direction.setCurrentIndex(d2s_idx)
            self.u1652_direction.setEnabled(False)
        else:
            self.u1652_direction.setEnabled(True)
        # Auto-populate root and history only when the user hasn't edited them
        self.root_input.setText(DEFAULT_ROOTS[ds])
        self.history_input.setText(DEFAULT_HISTORIES[ds])

    # ── File / dir browsing ────────────────────────────────────────────────

    def _browse_dir(self, line_edit: QLineEdit):
        path = QFileDialog.getExistingDirectory(self, "Select directory", line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_file(self, line_edit: QLineEdit, filt: str):
        path, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text(), filt)
        if path:
            line_edit.setText(path)

    def _build_geo_verify_tab(self, stage: int, grid_hint: int, chan_hint: int):
        """Build one 'Geo Verify Test' tab bound to a given backbone stage
        (2: 512-d/24x24 grid, 3: 1024-d/12x12 grid for Swin-B/384 — see
        model.py's encode_spatial_features(stage=...)). Widgets are stored in
        self.geo_tabs[stage] (a dict) rather than as self.geo_* attributes, so
        stage 2 and stage 3 tabs don't collide."""
        w = {}
        geo_w = QWidget()
        geo_v = QVBoxLayout()
        geo_form = QFormLayout()

        w["query_input"] = QLineEdit()
        geo_query_browse = QPushButton("Browse…")
        geo_query_browse.clicked.connect(
            lambda: self._browse_file(w["query_input"],
                                      "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*.*)"))
        geo_form.addRow("Query image", _make_hbox(w["query_input"], geo_query_browse))

        w["gallery_input"] = QLineEdit()
        geo_gallery_browse = QPushButton("Browse…")
        geo_gallery_browse.clicked.connect(
            lambda: self._browse_file(w["gallery_input"],
                                      "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*.*)"))
        geo_form.addRow("Gallery candidate image", _make_hbox(w["gallery_input"], geo_gallery_browse))

        w["footprint_input"] = QDoubleSpinBox()
        w["footprint_input"].setRange(1.0, 100000.0)
        w["footprint_input"].setValue(100.0)
        w["footprint_input"].setSuffix(" m")
        w["footprint_input"].setToolTip(
            "Real-world ground width the GALLERY image's full frame spans.\n"
            "Used to convert the fitted grid-unit shift into metres.\n"
            "(Per-image footprint metadata isn't wired in yet — enter the\n"
            "known/estimated value for this specific gallery image.)")
        geo_form.addRow("Gallery footprint width", w["footprint_input"])

        w["estimate_scale_input"] = QCheckBox(
            "Estimate scale (variable-altitude datasets, e.g. DenseUAV)")
        w["estimate_scale_input"].setChecked(True)
        w["estimate_scale_input"].setToolTip(
            "ON: fit rotation+scale+translation (similarity transform) — use\n"
            "when query/gallery may be at different altitudes.\n"
            "OFF: fit rotation+translation only, scale locked to 1 — use for\n"
            "fixed-altitude datasets (e.g. University-1652, SUES-200 per band).")
        geo_form.addRow("", w["estimate_scale_input"])

        w["scale_min_input"] = QDoubleSpinBox()
        w["scale_min_input"].setRange(0.05, 10.0)
        w["scale_min_input"].setSingleStep(0.05)
        w["scale_min_input"].setDecimals(2)
        w["scale_min_input"].setValue(0.7)
        w["scale_max_input"] = QDoubleSpinBox()
        w["scale_max_input"].setRange(0.05, 10.0)
        w["scale_max_input"].setSingleStep(0.05)
        w["scale_max_input"].setDecimals(2)
        w["scale_max_input"].setValue(1.5)
        scale_range_tip = (
            "Only used when 'Estimate scale' is ON. Bounds the plausible\n"
            "query/gallery scale ratio (e.g. from altitude differences).\n"
            "RANSAC hypotheses whose 2-point scale falls outside this range\n"
            "are discarded before they can compete on inlier count — without\n"
            "this a handful of spurious token matches can lock onto a\n"
            "degenerate near-zero or huge scale that happens to fit them.")
        w["scale_min_input"].setToolTip(scale_range_tip)
        w["scale_max_input"].setToolTip(scale_range_tip)
        geo_form.addRow("Scale range", _make_hbox(
            QLabel("min:"), w["scale_min_input"],
            QLabel("max:"), w["scale_max_input"]))

        w["reject_threshold_input"] = QDoubleSpinBox()
        w["reject_threshold_input"].setRange(0.1, 10000.0)
        w["reject_threshold_input"].setValue(20.0)
        w["reject_threshold_input"].setSuffix(" m")
        geo_form.addRow("Reject threshold", w["reject_threshold_input"])

        w["sim_threshold_input"] = QDoubleSpinBox()
        w["sim_threshold_input"].setRange(0.0, 1.0)
        w["sim_threshold_input"].setSingleStep(0.05)
        w["sim_threshold_input"].setValue(0.5)
        w["sim_threshold_input"].setDecimals(2)
        w["sim_threshold_input"].setToolTip(
            "Minimum mutual-NN cosine similarity to keep a token match\n"
            "before RANSAC transform fitting.")
        geo_form.addRow("Token match sim. threshold", w["sim_threshold_input"])

        w["ransac_thresh_input"] = QDoubleSpinBox()
        w["ransac_thresh_input"].setRange(0.1, 20.0)
        w["ransac_thresh_input"].setSingleStep(0.5)
        w["ransac_thresh_input"].setValue(4.0)
        w["ransac_thresh_input"].setToolTip(
            "RANSAC inlier threshold, in GRID units (tokens), not pixels/metres.\n"
            f"Stage {stage}'s grid is {grid_hint}x{grid_hint}, so each token here\n"
            f"spans ~{24 // grid_hint}x the ground distance of a stage-2 token —\n"
            "the same numeric threshold is proportionally more forgiving in\n"
            "real-world terms on a coarser grid. 4.0 worked well for stage 2 on\n"
            "real DenseUAV pairs (2.0 was too strict); re-tune per stage/dataset.")
        geo_form.addRow("RANSAC inlier threshold", w["ransac_thresh_input"])

        w["ransac_iters_input"] = QSpinBox()
        w["ransac_iters_input"].setRange(8, 2000)
        w["ransac_iters_input"].setValue(64)
        w["ransac_iters_input"].setToolTip(
            "RANSAC trials per fit attempt (2-point minimal sample). 64 is\n"
            "generous per standard RANSAC theory — only matters for speed at\n"
            "full-eval scale (thousands of candidate checks); irrelevant for a\n"
            "single pair here.")
        geo_form.addRow("RANSAC iterations", w["ransac_iters_input"])

        w["auto_escalate_input"] = QCheckBox(
            "Auto-increase RANSAC threshold if too few inliers")
        w["auto_escalate_input"].setChecked(True)
        w["auto_escalate_input"].setToolTip(
            "If the fit above doesn't clear 'Min. inliers to accept', retry with\n"
            "a doubled RANSAC threshold (up to 4 times, capped at half the grid\n"
            "size) instead of rejecting outright as 'too few inliers'. Safe to\n"
            "leave on: the real accept/reject gate is the metre-shift check below\n"
            "— a looser RANSAC tolerance can only let a bad fit's large recovered\n"
            "shift get caught THERE anyway, so there's no downside to giving a\n"
            "sparse-but-genuine match (e.g. stage 3's coarser grid) more room to\n"
            "prove itself before giving up.")
        geo_form.addRow("", w["auto_escalate_input"])

        w["min_inliers_input"] = QSpinBox()
        w["min_inliers_input"].setRange(2, 500)
        w["min_inliers_input"].setValue(4)
        geo_form.addRow("Min. inliers to accept", w["min_inliers_input"])

        w["check_reflection_input"] = QCheckBox("Reject mirror-ambiguous matches")
        w["check_reflection_input"].setChecked(True)
        w["check_reflection_input"].setToolTip(
            "A genuine, correctly-oriented match's token positions are almost\n"
            "never ALSO well-explained by their own mirror image. If a reflected\n"
            "(mirrored) transform explains a comparable fraction of the SAME\n"
            "matched points as the accepted fit, that's a red flag — e.g. a\n"
            "bilaterally-symmetric building complex matched against its own\n"
            "left-right flipped duplicate, which otherwise looks like a\n"
            "confident, small-shift, well-supported fit. Forcing only proper\n"
            "rotations does NOT reliably catch this on its own: with a 2-point\n"
            "minimal RANSAC sample, handedness is undetermined, so it can\n"
            "coincidentally find enough 'inliers' under the wrong handedness.")
        w["reflection_ratio_input"] = QDoubleSpinBox()
        w["reflection_ratio_input"].setRange(0.1, 2.0)
        w["reflection_ratio_input"].setSingleStep(0.05)
        w["reflection_ratio_input"].setDecimals(2)
        w["reflection_ratio_input"].setValue(0.7)
        w["reflection_ratio_input"].setToolTip(
            "Reject if reflected_inlier_count / accepted_inlier_count >= this\n"
            "ratio. Lower = stricter (rejects more borderline-symmetric cases).")
        geo_form.addRow("", _make_hbox(w["check_reflection_input"],
                                       QLabel("ratio:"), w["reflection_ratio_input"]))

        w["quantize_input"] = QCheckBox("Use quantized descriptors (compression test)")
        w["quantize_input"].setToolTip(
            f"Compress [{grid_hint},{grid_hint},{chan_hint}] float -> "
            f"[{grid_hint},{grid_hint}] cluster-id map + small shared codebook. "
            "This GUI test fits an AD-HOC codebook from\n"
            "just THIS query+gallery pair's own tokens — a real\n"
            "deployment codebook would be shared/fit from many images, which is\n"
            "usually HARDER to match well against (less tailored), so this is an\n"
            "optimistic quick check, not a production-accuracy measurement.\n"
            "Synthetic testing showed real accuracy loss in harder (large\n"
            "rotation + scale-mismatch) cases — verify on your own real pairs.")
        w["quantize_k_input"] = QSpinBox()
        w["quantize_k_input"].setRange(4, 65535)
        w["quantize_k_input"].setValue(256)
        w["quantize_k_input"].setToolTip(
            "Number of codebook clusters. IDs are stored as uint8 (K<=256) or\n"
            "uint16 (K>256, up to 65535). K can't exceed the number of tokens\n"
            f"being clustered ({2 * grid_hint * grid_hint} for this pair's ad-hoc "
            "GUI codebook).")
        geo_form.addRow("", _make_hbox(w["quantize_input"], QLabel("K:"), w["quantize_k_input"]))

        geo_form_w = QWidget(); geo_form_w.setLayout(geo_form)
        geo_v.addWidget(geo_form_w)

        w["run_btn"] = QPushButton("Run Geometric Verification Test")
        w["run_btn"].clicked.connect(lambda checked=False, s=stage: self._run_geo_verify_test(s))
        geo_v.addWidget(w["run_btn"])

        w["result_label"] = QLabel("Pick a query + gallery image and run.")
        w["result_label"].setWordWrap(True)
        w["result_label"].setFont(QFont("Courier New", 10))
        w["result_label"].setStyleSheet("padding: 8px; background: #f4f4f4;")
        geo_v.addWidget(w["result_label"])

        def _captioned_thumb(caption):
            col = QVBoxLayout()
            cap = QLabel(caption); cap.setAlignment(Qt.AlignCenter)
            thumb = QLabel(); thumb.setAlignment(Qt.AlignCenter)
            col.addWidget(cap)
            col.addWidget(thumb)
            return col, thumb

        geo_thumb_row = QHBoxLayout()
        q_col, w["query_thumb"] = _captioned_thumb("Query (before)")
        w_col, w["warped_thumb"] = _captioned_thumb("Query warped to gallery frame (after)")
        g_col, w["gallery_thumb"] = _captioned_thumb("Gallery")
        o_col, w["overlay_thumb"] = _captioned_thumb("Overlay (warped query + gallery)")
        geo_thumb_row.addLayout(q_col)
        geo_thumb_row.addLayout(w_col)
        geo_thumb_row.addLayout(g_col)
        geo_thumb_row.addLayout(o_col)
        geo_v.addLayout(geo_thumb_row)
        geo_v.addStretch()
        geo_w.setLayout(geo_v)

        w["_page_widget"] = geo_w
        self.geo_tabs[stage] = w
        return geo_w

    def _run_geo_verify_test(self, stage: int = 2):
        """Quick single-pair test of geo_verify.py's geometric shift verification
        on real images, without needing the full-gallery spatial cache."""
        import geo_verify

        w = self.geo_tabs[stage]
        q_path = w["query_input"].text().strip()
        g_path = w["gallery_input"].text().strip()
        if not q_path or not g_path:
            w["result_label"].setText("Pick both a query and a gallery image first.")
            return
        try:
            ckpt_path = Path(self.checkpoint_input.text())
            device = torch.device(self.device_input.currentData())
            self.log.append(f"[Geo Verify Test (Stage {stage})] Loading model from {ckpt_path}...")
            model, backbone, transform = self._load_model(ckpt_path, device)

            if not hasattr(model, "encode_spatial_features"):
                msg = (f"Backbone '{backbone}' does not support encode_spatial_features "
                      f"(only the standard SwinEmbedding head does — not routed/distilled models).")
                w["result_label"].setText(msg)
                self.log.append(f"[Geo Verify Test (Stage {stage})] {msg}")
                return

            query_img = Image.open(q_path).convert("RGB")
            gallery_img = Image.open(g_path).convert("RGB")
            q_tensor = transform(query_img).unsqueeze(0).to(device)
            g_tensor = transform(gallery_img).unsqueeze(0).to(device)

            model.eval()
            t_feat_start = time.perf_counter()
            with torch.inference_mode():
                q_feats, q_spatial = model.encode_spatial_features(q_tensor, stage=stage)
                g_feats, g_spatial = model.encode_spatial_features(g_tensor, stage=stage)
                q_feats = torch.nn.functional.normalize(q_feats, dim=-1)
                g_feats = torch.nn.functional.normalize(g_feats, dim=-1)
                coarse_sim = float((q_feats[0] * g_feats[0]).sum())
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            feat_time_s = time.perf_counter() - t_feat_start

            q_spatial0 = q_spatial[0].detach().cpu()
            g_spatial0 = g_spatial[0].detach().cpu()

            quantize_note = ""
            t_quant_start = time.perf_counter()
            if w["quantize_input"].isChecked():
                k = w["quantize_k_input"].value()
                q_np = q_spatial0.numpy()
                g_np = g_spatial0.numpy()
                combined = np.concatenate([
                    q_np.reshape(-1, q_np.shape[-1]),
                    g_np.reshape(-1, g_np.shape[-1]),
                ], axis=0)
                codebook = geo_verify.fit_codebook(combined, n_clusters=k)
                q_ids = geo_verify.quantize_spatial(q_spatial0, codebook)
                g_ids = geo_verify.quantize_spatial(g_spatial0, codebook)
                q_spatial0 = geo_verify.reconstruct_from_ids(q_ids, codebook)
                g_spatial0 = geo_verify.reconstruct_from_ids(g_ids, codebook)
                stats = geo_verify.compression_stats(q_np.shape[0], k, n_gallery=1,
                                                      channels=q_np.shape[-1])
                quantize_note = (
                    f"\n[Quantized: K={k} clusters fit ad-hoc from this pair's "
                    f"{combined.shape[0]} tokens, ~{stats['compression_ratio']:.0f}x "
                    f"per-image compression (illustrative only — an ad-hoc "
                    f"per-pair codebook is optimistic vs. a real shared codebook, "
                    f"see checkbox tooltip)]"
                )
            quantize_time_s = time.perf_counter() - t_quant_start

            t_match_start = time.perf_counter()
            res = geo_verify.verify_and_reject(
                q_spatial0, g_spatial0,
                gallery_footprint_m=w["footprint_input"].value(),
                estimate_scale=w["estimate_scale_input"].isChecked(),
                reject_threshold_m=w["reject_threshold_input"].value(),
                sim_threshold=w["sim_threshold_input"].value(),
                ransac_thresh_px=w["ransac_thresh_input"].value(),
                ransac_iters=w["ransac_iters_input"].value(),
                min_inliers=w["min_inliers_input"].value(),
                scale_range=(w["scale_min_input"].value(), w["scale_max_input"].value()),
                auto_escalate_ransac=w["auto_escalate_input"].isChecked(),
                check_reflection=w["check_reflection_input"].isChecked(),
                max_reflection_ratio=w["reflection_ratio_input"].value(),
            )
            match_time_s = time.perf_counter() - t_match_start
            total_time_s = feat_time_s + quantize_time_s + match_time_s

            fit = res["fit"]
            timing_line = (
                f"Timing: feature extraction={feat_time_s*1000:.0f}ms, "
                f"{'quantization=' + format(quantize_time_s*1000, '.0f') + 'ms, ' if w['quantize_input'].isChecked() else ''}"
                f"matching+RANSAC={match_time_s*1000:.0f}ms, total={total_time_s*1000:.0f}ms"
            )
            lines = [
                f"Stage {stage} spatial grid: {q_spatial0.shape[0]}x{q_spatial0.shape[1]}x{q_spatial0.shape[2]}",
                f"Coarse descriptor cosine similarity: {coarse_sim:.4f}",
                timing_line,
                quantize_note,
                "",
                f"Geometric verification: {'ACCEPT' if res['accept'] else 'REJECT'}",
                f"Reason: {res['reason']}",
            ]
            if fit is not None:
                ty, tx = fit["translation"]
                thresh_used = fit.get("ransac_thresh_used", w["ransac_thresh_input"].value())
                escalated_note = (f" (escalated from {w['ransac_thresh_input'].value():.1f})"
                                  if abs(thresh_used - w["ransac_thresh_input"].value()) > 1e-6 else "")
                # fit can be non-None (fit_transform succeeded) while shift_m is
                # still None: verify_and_reject returns early with shift_m=None
                # when inlier_count doesn't clear min_inliers, BEFORE it ever
                # computes the shift — the fit dict itself is still populated.
                shift_line = (f"Estimated shift: {res['shift_m']:.1f} m "
                             f"(grid translation row={ty:.2f}, col={tx:.2f})"
                             if res["shift_m"] is not None else
                             "Estimated shift: n/a (rejected before shift computed — "
                             "too few RANSAC inliers)")
                lines += [
                    shift_line,
                    f"Rotation: {fit['rotation_deg']:.1f}°   Scale: {fit['scale']:.3f}",
                    f"RANSAC inliers: {fit['inlier_count']}/{fit['n_matches']} token matches "
                    f"(threshold={thresh_used:.2f}{escalated_note})",
                    f"Reflection check: {fit['reflection_inlier_count']}/{fit['inlier_count']} "
                    f"points also explained by a mirrored transform "
                    f"(ratio={fit['reflection_inlier_count']/max(fit['inlier_count'],1):.2f})",
                ]
            else:
                lines.append(f"(no transform could be fit — {res['reason']})")
            result_text = "\n".join(lines)
            w["result_label"].setText(result_text)
            self.log.append(f"[Geo Verify Test (Stage {stage})]\n" + result_text)

            w["query_thumb"].setPixmap(pil_to_pixmap(query_img, size=220))
            w["gallery_thumb"].setPixmap(pil_to_pixmap(gallery_img, size=220))

            if fit is not None:
                img_h, img_w = q_tensor.shape[-2], q_tensor.shape[-1]
                query_resized = query_img.resize((img_w, img_h), Image.BILINEAR)
                gallery_resized = gallery_img.resize((img_w, img_h), Image.BILINEAR)
                q_arr = np.array(query_resized)
                gallery_arr = np.array(gallery_resized)
                warped_arr = geo_verify.warp_to_gallery(
                    q_arr, fit, grid_size=q_spatial0.shape[0], gallery_hw=(img_h, img_w))
                warped_img = Image.fromarray(warped_arr)
                overlay_arr = (0.5 * warped_arr.astype(np.float32)
                              + 0.5 * gallery_arr.astype(np.float32)).astype(np.uint8)
                overlay_img = Image.fromarray(overlay_arr)
                w["warped_thumb"].setPixmap(pil_to_pixmap(warped_img, size=220))
                w["overlay_thumb"].setPixmap(pil_to_pixmap(overlay_img, size=220))
            else:
                w["warped_thumb"].clear()
                w["overlay_thumb"].clear()
        except Exception as exc:
            import traceback
            self.log.append(f"[Geo Verify Test (Stage {stage})] ERROR: {exc}\n{traceback.format_exc()}")
            w["result_label"].setText(f"Error: {exc}")

    def _build_geo_fullscan_tab(self):
        """Build the 'Geo Verify Test (Full Gallery)' tab: given ONE query
        image, scan a full gallery directory (DenseUAV-style layout: one
        subfolder per location) for its top-K coarse candidates by pooled
        embedding, then geometrically re-rank those K candidates and preview
        the result against ground truth (the query's own parent folder name,
        matched against each candidate's — same convention _eval_denseuav
        uses for location IDs elsewhere in this file)."""
        w = QWidget()
        v = QVBoxLayout()
        form = QFormLayout()

        self.geo_fs_query_input = QLineEdit()
        query_browse = QPushButton("Browse…")
        query_browse.clicked.connect(
            lambda: self._browse_file(self.geo_fs_query_input,
                                      "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*.*)"))
        form.addRow("Query image", _make_hbox(self.geo_fs_query_input, query_browse))

        self.geo_fs_gallery_root_input = QLineEdit()
        gallery_browse = QPushButton("Browse…")
        gallery_browse.clicked.connect(lambda: self._browse_dir(self.geo_fs_gallery_root_input))
        self.geo_fs_gallery_root_input.setToolTip(
            "Directory containing one subfolder per gallery location (DenseUAV\n"
            "layout, e.g. test/gallery_satellite) — every image inside is scanned.\n"
            "Ground truth is read from the QUERY's own parent folder name, matched\n"
            "against each candidate's parent folder name.")
        form.addRow("Gallery root directory",
                    _make_hbox(self.geo_fs_gallery_root_input, gallery_browse))

        self.geo_fs_stage_input = QComboBox()
        self.geo_fs_stage_input.addItem("Stage 2 (24x24; 512-d Swin-B / 384-d ConvNeXt-T)", 2)
        self.geo_fs_stage_input.addItem("Stage 3 (12x12; 1024-d Swin-B / 768-d ConvNeXt-T)", 3)
        self.geo_fs_stage_input.setCurrentIndex(1)   # default Stage 3
        form.addRow("Stage", self.geo_fs_stage_input)

        self.geo_fs_topk_input = QSpinBox()
        self.geo_fs_topk_input.setRange(1, 100)
        self.geo_fs_topk_input.setValue(10)
        form.addRow("Top-K candidates", self.geo_fs_topk_input)

        self.geo_fs_estimate_scale_input = QCheckBox(
            "Estimate scale (variable-altitude datasets, e.g. DenseUAV)")
        self.geo_fs_estimate_scale_input.setChecked(True)
        form.addRow("", self.geo_fs_estimate_scale_input)

        self.geo_fs_scale_min_input = QDoubleSpinBox()
        self.geo_fs_scale_min_input.setRange(0.05, 10.0)
        self.geo_fs_scale_min_input.setSingleStep(0.05)
        self.geo_fs_scale_min_input.setDecimals(2)
        self.geo_fs_scale_min_input.setValue(0.7)
        self.geo_fs_scale_max_input = QDoubleSpinBox()
        self.geo_fs_scale_max_input.setRange(0.05, 10.0)
        self.geo_fs_scale_max_input.setSingleStep(0.05)
        self.geo_fs_scale_max_input.setDecimals(2)
        self.geo_fs_scale_max_input.setValue(1.5)
        form.addRow("Scale range", _make_hbox(
            QLabel("min:"), self.geo_fs_scale_min_input,
            QLabel("max:"), self.geo_fs_scale_max_input))

        self.geo_fs_reject_input = QDoubleSpinBox()
        self.geo_fs_reject_input.setRange(0.1, 10000.0)
        self.geo_fs_reject_input.setValue(20.0)
        self.geo_fs_reject_input.setSuffix(" m")
        self.geo_fs_reject_input.setToolTip(
            "Gallery footprint per candidate comes from the model's own scale\n"
            "head prediction (exp(log_scale)) — no manual entry needed.")
        form.addRow("Reject threshold", self.geo_fs_reject_input)

        self.geo_fs_sim_input = QDoubleSpinBox()
        self.geo_fs_sim_input.setRange(0.0, 1.0)
        self.geo_fs_sim_input.setSingleStep(0.05)
        self.geo_fs_sim_input.setDecimals(2)
        self.geo_fs_sim_input.setValue(0.2)
        form.addRow("Token match sim. threshold", self.geo_fs_sim_input)

        self.geo_fs_ransac_input = QDoubleSpinBox()
        self.geo_fs_ransac_input.setRange(0.1, 20.0)
        self.geo_fs_ransac_input.setSingleStep(0.5)
        self.geo_fs_ransac_input.setValue(2.0)
        form.addRow("RANSAC inlier threshold", self.geo_fs_ransac_input)

        self.geo_fs_ransac_iters_input = QSpinBox()
        self.geo_fs_ransac_iters_input.setRange(8, 2000)
        self.geo_fs_ransac_iters_input.setValue(64)
        form.addRow("RANSAC iterations", self.geo_fs_ransac_iters_input)

        self.geo_fs_min_inliers_input = QSpinBox()
        self.geo_fs_min_inliers_input.setRange(2, 500)
        self.geo_fs_min_inliers_input.setValue(10)
        self.geo_fs_min_inliers_input.setToolTip(
            "The fitted transform has 4 degrees of freedom (rotation, scale, 2D\n"
            "translation) — a value of 4 is trivially satisfiable (a minimal-\n"
            "sample RANSAC fit is always near-perfect by construction). Needs to\n"
            "be well above 4 to mean anything.")
        form.addRow("Min. inliers to accept", self.geo_fs_min_inliers_input)

        self.geo_fs_auto_escalate_input = QCheckBox(
            "Auto-increase RANSAC threshold if too few inliers")
        self.geo_fs_auto_escalate_input.setChecked(True)
        form.addRow("", self.geo_fs_auto_escalate_input)

        self.geo_fs_check_reflection_input = QCheckBox("Reject mirror-ambiguous matches")
        self.geo_fs_check_reflection_input.setChecked(True)
        self.geo_fs_check_reflection_input.setToolTip(
            "A genuine match's token positions are almost never ALSO well-\n"
            "explained by their own mirror image. Rejects candidates where a\n"
            "reflected transform explains a comparable fraction of the same\n"
            "matched points as the accepted fit — e.g. a symmetric building\n"
            "matched against its own flipped duplicate.")
        self.geo_fs_reflection_ratio_input = QDoubleSpinBox()
        self.geo_fs_reflection_ratio_input.setRange(0.1, 2.0)
        self.geo_fs_reflection_ratio_input.setSingleStep(0.05)
        self.geo_fs_reflection_ratio_input.setDecimals(2)
        self.geo_fs_reflection_ratio_input.setValue(0.7)
        form.addRow("", _make_hbox(self.geo_fs_check_reflection_input,
                                   QLabel("ratio:"), self.geo_fs_reflection_ratio_input))

        self.geo_fs_semantic_input = QCheckBox("Semantic map (whole-gallery codebook)")
        self.geo_fs_semantic_input.setChecked(True)
        self.geo_fs_semantic_input.setToolTip(
            "Fit ONE shared K-means codebook over individual grid-cell\n"
            "descriptors (each 1xC token) sampled across the WHOLE gallery,\n"
            "then quantize every image's spatial descriptor into a small\n"
            "cluster-ID map (e.g. stage 3: 12x12 cells -> 12x12 IDs). The map\n"
            "is rendered colour-coded next to each thumbnail — same colour =\n"
            "same semantic mini-cluster, so corresponding structures in a\n"
            "genuine query/candidate pair should show matching colour layouts.\n"
            "The codebook is cached per checkpoint+gallery+stage+K (first run\n"
            "extracts descriptors for a gallery sample; later runs are instant).")
        self.geo_fs_semantic_k_input = QSpinBox()
        self.geo_fs_semantic_k_input.setRange(8, 256)
        self.geo_fs_semantic_k_input.setValue(64)
        self.geo_fs_semantic_k_input.setToolTip(
            "Number of mini-clusters in the shared cell codebook. 64 gives a\n"
            "readable colour map; more = finer semantic distinctions but\n"
            "harder to tell colours apart visually.")
        form.addRow("", _make_hbox(self.geo_fs_semantic_input,
                                   QLabel("mini-clusters K:"), self.geo_fs_semantic_k_input))

        form_w = QWidget(); form_w.setLayout(form)
        v.addWidget(form_w)

        self.geo_fs_run_btn = QPushButton("Scan Gallery + Geo Re-rank")
        self.geo_fs_run_btn.clicked.connect(self._run_geo_verify_fullscan)
        v.addWidget(self.geo_fs_run_btn)

        self.geo_fs_result_label = QLabel("Pick a query image and gallery root directory, then run.")
        self.geo_fs_result_label.setWordWrap(True)
        self.geo_fs_result_label.setFont(QFont("Courier New", 10))
        self.geo_fs_result_label.setStyleSheet("padding: 8px; background: #f4f4f4;")
        v.addWidget(self.geo_fs_result_label)

        # Query preview row: image + (when semantic maps are on) its cluster-ID map.
        query_row = QHBoxLayout()
        for attr, caption in (("geo_fs_query_thumb", "Query"),
                              ("geo_fs_query_sem_thumb", "Query semantic map")):
            col = QVBoxLayout()
            cap = QLabel(caption); cap.setAlignment(Qt.AlignCenter)
            cap.setStyleSheet("font-size: 9px; color: #555;")
            thumb = QLabel(); thumb.setAlignment(Qt.AlignCenter)
            col.addWidget(cap); col.addWidget(thumb)
            setattr(self, attr, thumb)
            query_row.addLayout(col)
        query_row.addStretch()
        v.addLayout(query_row)

        legend = QLabel("Border colour: green = ground truth location, "
                        "blue = accepted (not GT), red = rejected.")
        legend.setStyleSheet("font-size: 9px; color: #555;")
        v.addWidget(legend)

        self.geo_fs_scroll = QScrollArea()
        self.geo_fs_scroll.setWidgetResizable(True)
        self.geo_fs_grid_container = QWidget()
        self.geo_fs_grid = QGridLayout(self.geo_fs_grid_container)
        self.geo_fs_scroll.setWidget(self.geo_fs_grid_container)
        v.addWidget(self.geo_fs_scroll, stretch=1)

        w.setLayout(v)
        return w

    def _run_geo_verify_fullscan(self):
        """Scan a full gallery directory for a single query's top-K coarse
        candidates (pooled embedding, cosine similarity), then geometrically
        re-rank those K candidates and preview the result against ground truth."""
        import geo_verify

        q_path = self.geo_fs_query_input.text().strip()
        gallery_root = self.geo_fs_gallery_root_input.text().strip()
        if not q_path or not gallery_root:
            self.geo_fs_result_label.setText("Pick a query image and gallery root directory first.")
            return
        try:
            ckpt_path = Path(self.checkpoint_input.text())
            device = torch.device(self.device_input.currentData())
            self.log.append(f"[Geo Verify Full Scan] Loading model from {ckpt_path}...")
            model, backbone, transform = self._load_model(ckpt_path, device)

            if not hasattr(model, "encode_spatial_features"):
                msg = (f"Backbone '{backbone}' does not support encode_spatial_features "
                      f"(only the standard SwinEmbedding head does — not routed/distilled models).")
                self.geo_fs_result_label.setText(msg)
                self.log.append(f"[Geo Verify Full Scan] {msg}")
                return

            gallery_dir = Path(gallery_root)
            gallery_paths = []
            for loc_dir in sorted(d for d in gallery_dir.iterdir() if d.is_dir()):
                imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                gallery_paths.extend(imgs)
            if not gallery_paths:
                msg = (f"No gallery images found under {gallery_dir} "
                      f"(expected one subfolder per location).")
                self.geo_fs_result_label.setText(msg)
                self.log.append(f"[Geo Verify Full Scan] {msg}")
                return

            gt_loc_id = Path(q_path).parent.name
            batch_size = self.batch_size_input.value()
            self.log.append(f"[Geo Verify Full Scan] Embedding gallery "
                            f"({len(gallery_paths)} images)...")
            gal_emb, gal_scales = self._embed_cached(
                "geo_fullscan_gallery", gallery_paths, ckpt_path, backbone,
                model, device, batch_size, transform, with_scale=True)
            if gal_emb is None:
                self.geo_fs_result_label.setText("Stopped.")
                return
            gal_norm = (gal_emb / (np.linalg.norm(gal_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
            gal_footprints = np.exp(np.asarray(gal_scales).reshape(-1))

            stage = self.geo_fs_stage_input.currentData()
            query_img = Image.open(q_path).convert("RGB")
            q_tensor = transform(query_img).unsqueeze(0).to(device)
            model.eval()
            with torch.inference_mode():
                q_feats, _ = model.encode_features_and_scale(q_tensor)
                q_feats_norm = torch.nn.functional.normalize(q_feats, dim=-1)[0].cpu().numpy()
                _, q_spatial = model.encode_spatial_features(q_tensor, stage=stage)
                q_spatial0 = q_spatial[0].cpu()

            sims = gal_norm @ q_feats_norm
            top_k = min(self.geo_fs_topk_input.value(), len(gallery_paths))
            top_idx = np.argsort(-sims)[:top_k]

            # Whole-gallery cell codebook for the semantic maps: every grid cell
            # (one 1xC token) across the gallery is a point in the same space,
            # so ONE shared K-means over a gallery-wide sample gives cluster IDs
            # that mean the same thing in every image — that's what makes two
            # images' colour maps comparable. Cached per checkpoint+gallery+stage+K.
            sem_codebook = sem_palette = q_sem_ids = None
            if self.geo_fs_semantic_input.isChecked():
                sem_k = self.geo_fs_semantic_k_input.value()
                cb_cp = _cache_path(f"geo_fullscan_codebook_s{stage}_k{sem_k}", ckpt_path,
                                    [str(p) for p in gallery_paths], backbone)
                sem_codebook = _try_load_generic_cache(cb_cp, ckpt_path)
                if sem_codebook is None:
                    rng = np.random.RandomState(0)
                    n_sample = min(len(gallery_paths), 300)
                    sample_paths = [gallery_paths[i] for i in
                                    rng.choice(len(gallery_paths), size=n_sample, replace=False)]
                    self.log.append(
                        f"[Geo Verify Full Scan] Fitting shared K={sem_k} cell codebook "
                        f"from {n_sample} gallery images (one-time; cached after this)...")
                    sample_spatial = self._extract_spatial_descriptors(
                        model, sample_paths, device, batch_size, transform, stage)
                    if sample_spatial is None:
                        self.geo_fs_result_label.setText("Stopped.")
                        return
                    tokens = sample_spatial.reshape(-1, sample_spatial.shape[-1])
                    if tokens.shape[0] > 40000:
                        tokens = tokens[rng.choice(tokens.shape[0], size=40000, replace=False)]
                    sem_codebook = geo_verify.fit_codebook(tokens, n_clusters=sem_k)
                    del sample_spatial
                    _save_generic_cache(cb_cp, ckpt_path, sem_codebook)
                    self.log.append(f"[Geo Verify Full Scan] Codebook cached ({cb_cp.name}).")
                sem_palette = _semantic_palette(sem_k)
                q_sem_ids = geo_verify.quantize_spatial(q_spatial0, sem_codebook)

            self.geo_fs_query_thumb.setPixmap(pil_to_pixmap(query_img, size=140))
            if q_sem_ids is not None:
                self.geo_fs_query_sem_thumb.setPixmap(
                    _semantic_pixmap(q_sem_ids, sem_palette, size=140))
            else:
                self.geo_fs_query_sem_thumb.clear()

            self.log.append(f"[Geo Verify Full Scan] Verifying top-{top_k} candidates "
                            f"(stage {stage})...")
            candidates = []
            with torch.inference_mode():
                for rank, ci in enumerate(top_idx):
                    g_path = gallery_paths[ci]
                    g_img = Image.open(g_path).convert("RGB")
                    g_tensor = transform(g_img).unsqueeze(0).to(device)
                    _, g_spatial = model.encode_spatial_features(g_tensor, stage=stage)
                    res = geo_verify.verify_and_reject(
                        q_spatial0, g_spatial[0].cpu(), float(gal_footprints[ci]),
                        estimate_scale=self.geo_fs_estimate_scale_input.isChecked(),
                        reject_threshold_m=self.geo_fs_reject_input.value(),
                        sim_threshold=self.geo_fs_sim_input.value(),
                        ransac_thresh_px=self.geo_fs_ransac_input.value(),
                        ransac_iters=self.geo_fs_ransac_iters_input.value(),
                        min_inliers=self.geo_fs_min_inliers_input.value(),
                        scale_range=(self.geo_fs_scale_min_input.value(),
                                    self.geo_fs_scale_max_input.value()),
                        auto_escalate_ransac=self.geo_fs_auto_escalate_input.isChecked(),
                        check_reflection=self.geo_fs_check_reflection_input.isChecked(),
                        max_reflection_ratio=self.geo_fs_reflection_ratio_input.value(),
                    )
                    candidates.append({
                        "path": g_path, "img": g_img, "coarse_rank": rank,
                        "coarse_sim": float(sims[ci]), "accept": res["accept"],
                        "shift_m": res["shift_m"], "fit": res["fit"], "reason": res["reason"],
                        "is_gt": (g_path.parent.name == gt_loc_id),
                        "sem_ids": (geo_verify.quantize_spatial(g_spatial[0].cpu(), sem_codebook)
                                    if sem_codebook is not None else None),
                    })

            # Filter re-rank: accepted first (by original coarse sim desc), then
            # rejected (by original coarse sim desc) — same semantics as
            # _geo_rerank_denseuav's margin-based demotion, simplified since
            # this only ever holds one query's own top-K in memory at a time.
            reranked = sorted(candidates, key=lambda c: (0 if c["accept"] else 1, -c["coarse_sim"]))

            coarse_top1 = candidates[0]
            geo_top1 = reranked[0]
            n_accepted = sum(1 for c in candidates if c["accept"])
            summary = [
                f"Query: {Path(q_path).name}   Ground truth location: {gt_loc_id}",
                f"Gallery scanned: {len(gallery_paths)} images",
                f"Accepted: {n_accepted}/{top_k} candidates",
                "",
                f"Coarse top-1: {'CORRECT' if coarse_top1['is_gt'] else 'WRONG'} "
                f"({coarse_top1['path'].parent.name}, sim={coarse_top1['coarse_sim']:.4f})",
                f"Geo-filtered top-1: {'CORRECT' if geo_top1['is_gt'] else 'WRONG'} "
                f"({geo_top1['path'].parent.name})",
            ]
            self.geo_fs_result_label.setText("\n".join(summary))
            self.log.append("[Geo Verify Full Scan]\n" + "\n".join(summary))

            self._populate_geo_fullscan_grid(reranked, sem_palette=sem_palette)
        except Exception as exc:
            import traceback
            self.log.append(f"[Geo Verify Full Scan] ERROR: {exc}\n{traceback.format_exc()}")
            self.geo_fs_result_label.setText(f"Error: {exc}")

    def _populate_geo_fullscan_grid(self, reranked: list, sem_palette=None):
        while self.geo_fs_grid.count():
            item = self.geo_fs_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        show_sem = sem_palette is not None
        THUMB = 140; COLS = 3 if show_sem else 5
        for i, c in enumerate(reranked):
            if c["is_gt"]:
                border = "#2ca02c"      # green: ground truth location
            elif c["accept"]:
                border = "#1f77b4"      # blue: accepted, not GT
            else:
                border = "#d62728"      # red: rejected

            pix = pil_to_pixmap(c["img"], size=THUMB)
            thumb = QLabel(); thumb.setPixmap(pix); thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet(f"border: 3px solid {border}; border-radius: 3px;")
            thumb.setToolTip(str(c["path"]))

            sem_thumb = None
            if show_sem and c.get("sem_ids") is not None:
                sem_thumb = QLabel()
                sem_thumb.setPixmap(_semantic_pixmap(c["sem_ids"], sem_palette, size=THUMB))
                sem_thumb.setAlignment(Qt.AlignCenter)
                sem_thumb.setToolTip("Semantic map (shared whole-gallery cell codebook) — "
                                     "compare colour layout against the query's map")

            fit = c["fit"]
            fit_str = (f"inliers {fit['inlier_count']}/{fit['n_matches']}"
                      if fit is not None else "no fit")
            mirror_str = ""
            if fit is not None:
                ref_ratio = fit["reflection_inlier_count"] / max(fit["inlier_count"], 1)
                if ref_ratio >= 0.4:   # worth flagging even if below the reject ratio
                    mirror_str = f"  mirror-ratio={ref_ratio:.2f}"
            shift_str = f"{c['shift_m']:.1f}m" if c["shift_m"] is not None else "n/a"
            gt_str = " [GT]" if c["is_gt"] else ""
            caption = (
                f"geo #{i+1}  (coarse #{c['coarse_rank']+1}, sim={c['coarse_sim']:.3f}){gt_str}\n"
                f"{'ACCEPT' if c['accept'] else 'REJECT'}  shift={shift_str}  {fit_str}{mirror_str}"
            )
            cap_lbl = QLabel(caption)
            cap_lbl.setAlignment(Qt.AlignCenter)
            cap_lbl.setStyleSheet(f"color: {border}; font-size: 9px;")

            cell = QWidget(); cl = QVBoxLayout()
            cl.setContentsMargins(2, 2, 2, 2); cl.setSpacing(1)
            if sem_thumb is not None:
                pair = QHBoxLayout(); pair.setSpacing(2)
                pair.addWidget(thumb); pair.addWidget(sem_thumb)
                cl.addLayout(pair)
            else:
                cl.addWidget(thumb)
            cl.addWidget(cap_lbl)
            cell.setLayout(cl)
            self.geo_fs_grid.addWidget(cell, i // COLS, i % COLS)

    def _browse_history(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Select history file", self.history_input.text(),
            "JSON (*.json);;All files (*.*)")
        if path:
            self.history_input.setText(path)

    # ── Eval lifecycle ─────────────────────────────────────────────────────

    def start_eval(self):
        if self.eval_thread and self.eval_thread.is_alive():
            return
        self._save_settings()
        self.stop_event.clear()
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.log.clear()
        self.eval_thread = threading.Thread(target=self.eval_loop, daemon=True)
        self.eval_thread.start()

    def stop_eval(self):
        self.stop_event.set()

    def on_eval_finished(self, status: str):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message.emit(f"— Evaluation {status}. —")

    # ── Results display ────────────────────────────────────────────────────

    def on_results_ready(self, results: dict, dataset_type: str):
        self.results_table.clear()
        if dataset_type == DATASET_SUES200:
            self._show_results_sues(results)
        elif dataset_type == DATASET_G4L:
            self._show_results_g4l(results)
        else:
            self._show_results_u1652(results)

    def _show_results_g4l(self, results: dict):
        cols = ["Protocol", "Queries", "Gallery", "R@1 (%)", "R@5 (%)",
                "R@10 (%)", "AP (%)", "SDM@1", "SDM@3"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(0)
        labels = {"pos": "pos (official)", "pos_semipos": "pos_semipos (ref.)",
                 "pos_vq": "pos + VQ re-rank", "pos_semipos_vq": "pos_semipos + VQ re-rank"}
        for key in ("pos", "pos_semipos", "pos_vq", "pos_semipos_vq"):
            if key not in results:
                continue
            r = results[key]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            vals = [labels[key], str(r["n_queries"]), str(r["n_gallery"]),
                    f"{r['r1']*100:.2f}", f"{r['r5']*100:.2f}",
                    f"{r['r10']*100:.2f}", f"{r['ap']*100:.2f}",
                    f"{r['sdm1']:.4f}", f"{r['sdm3']:.4f}"]
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if key in ("pos", "pos_vq"):
                    f = item.font(); f.setBold(True); item.setFont(f)
                self.results_table.setItem(row, c, item)
        if "pos" in results:
            r = results["pos"]
            self.summary_label.setText(
                f"[their protocol]  R@1: {r['r1']*100:.2f}%    "
                f"AP: {r['ap']*100:.2f}%    SDM@3: {r['sdm3']:.4f}    "
                f"(target: 80.20 / 87.83 / 0.8546)")

    def _show_results_sues(self, results: dict):
        cols = ["Altitude", "Queries", "R@1 (%)", "R@5 (%)", "R@10 (%)",
                "Scale MAE (m)", "Scale Rel (%)", "Pred log-scale"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(0)

        # isinstance guard: results may also carry non-altitude string keys
        # ("overall", "vq_overall") which can't be sorted against ints.
        for alt in sorted(k for k in results if isinstance(k, (int, float))):
            r = results[alt]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{alt} m"))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(r["queries"])))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{r['r1']*100:.1f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{r.get('r5',0)*100:.1f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{r.get('r10',0)*100:.1f}"))
            mae = r.get("scale_mae_m"); rel = r.get("scale_rel_err_pct"); mls = r.get("mean_log_scale")
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{mae:.1f}"  if mae is not None else ""))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{rel:.1f}"  if rel is not None else ""))
            self.results_table.setItem(row, 7, QTableWidgetItem(f"{mls:.3f}"  if mls is not None else ""))
            for c in range(len(cols)):
                item = self.results_table.item(row, c)
                if item: item.setTextAlignment(Qt.AlignCenter)

        if "overall" in results:
            ov = results["overall"]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem("Overall"))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(ov["queries"])))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{ov['r1']*100:.1f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{ov.get('r5',0)*100:.1f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{ov.get('r10',0)*100:.1f}"))
            mae = ov.get("scale_mae_m"); rel = ov.get("scale_rel_err_pct")
            disc = ov.get("scale_discrimination")
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{mae:.1f}" if mae is not None else ""))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{rel:.1f}" if rel is not None else ""))
            self.results_table.setItem(row, 7, QTableWidgetItem(
                f"{disc:.1f}% discr." if disc is not None else ""))
            for c in range(len(cols)):
                item = self.results_table.item(row, c)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)
                    f = item.font(); f.setBold(True); item.setFont(f)
            self.summary_label.setText(
                f"R@1: {ov['r1']*100:.1f}%    R@5: {ov.get('r5',0)*100:.1f}%    "
                f"R@10: {ov.get('r10',0)*100:.1f}%    ({ov['queries']} queries)")

        if "vq_overall" in results:
            vq = results["vq_overall"]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem("Overall (VQ re-rank)"))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(vq["queries"])))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{vq['r1']*100:.1f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{vq.get('r5',0)*100:.1f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{vq.get('r10',0)*100:.1f}"))
            for c in range(self.results_table.columnCount()):
                item = self.results_table.item(row, c)
                if item:
                    item.setTextAlignment(Qt.AlignCenter)
                    f = item.font(); f.setBold(True); item.setFont(f)

    def _show_results_u1652(self, results: dict):
        cols = ["Direction", "Queries", "Gallery", "R@1 (%)", "R@5 (%)", "R@10 (%)", "mAP (%)"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(0)
        for key in ("d2s", "s2d"):
            if key not in results:
                continue
            r = results[key]
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            direction_lbl = "Drone→Sat" if key == "d2s" else "Sat→Drone"
            self.results_table.setItem(row, 0, QTableWidgetItem(direction_lbl))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(r["n_queries"])))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(r["n_gallery"])))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{r['r1']*100:.2f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{r['r5']*100:.2f}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{r['r10']*100:.2f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{r['ap']*100:.2f}"))
            for c in range(len(cols)):
                item = self.results_table.item(row, c)
                if item: item.setTextAlignment(Qt.AlignCenter)

        # Any additional result sets beyond the two base directions (e.g. DenseUAV's
        # re-rank rows: "d2s_georank_filtered", "d2s_vqrank") — shown as extra rows
        # so they don't silently vanish from the table.
        rerank_labels = {"georank_filtered": "Geo re-rank (filtered)",
                         "vqrank": "VQ map re-rank"}
        for key in sorted(k for k in results if k not in ("d2s", "s2d")):
            r = results[key]
            base_dir, _, suffix = key.partition("_")
            label = rerank_labels.get(suffix, key)
            dir_lbl = "Drone→Sat" if base_dir == "d2s" else "Sat→Drone" if base_dir == "s2d" else base_dir
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(f"{dir_lbl}: {label}"))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(r["n_queries"])))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(r["n_gallery"])))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{r['r1']*100:.2f}"))
            self.results_table.setItem(row, 4, QTableWidgetItem(f"{r['r5']*100:.2f}"))
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{r['r10']*100:.2f}"))
            self.results_table.setItem(row, 6, QTableWidgetItem(f"{r['ap']*100:.2f}"))
            for c in range(len(cols)):
                item = self.results_table.item(row, c)
                if item: item.setTextAlignment(Qt.AlignCenter)

        # Summary from whichever direction was run
        key = "d2s" if "d2s" in results else "s2d"
        if key in results:
            r = results[key]
            self.summary_label.setText(
                f"R@1: {r['r1']*100:.2f}%    R@5: {r['r5']*100:.2f}%    "
                f"R@10: {r['r10']*100:.2f}%    mAP: {r['ap']*100:.2f}%    "
                f"({r['n_queries']} queries / {r['n_gallery']} gallery)")

    # ── Sample matches display ─────────────────────────────────────────────

    def on_samples_ready(self, samples: list):
        while self.samples_grid.count():
            item = self.samples_grid.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        THUMB = 112; COLS = 6
        for i, (q_path, g_path, correct, caption) in enumerate(samples):
            color = "#2ca02c" if correct else "#d62728"

            def _thumb(p):
                try:
                    return pil_to_pixmap(Image.open(p).convert("RGB"), THUMB)
                except Exception:
                    pix = QPixmap(THUMB, THUMB); pix.fill(); return pix

            def _lbl(pix, tooltip, col):
                l = QLabel(); l.setPixmap(pix); l.setAlignment(Qt.AlignCenter)
                l.setToolTip(str(tooltip))
                l.setStyleSheet(f"border: 2px solid {col}; border-radius: 3px;")
                return l

            q_lbl = _lbl(_thumb(q_path),  q_path,  color)
            g_lbl = _lbl(_thumb(g_path),  g_path,  color)
            cap_lbl = QLabel(caption)
            cap_lbl.setAlignment(Qt.AlignCenter)
            cap_lbl.setStyleSheet(f"color: {color}; font-size: 9px;")

            cell = QWidget(); cl = QVBoxLayout()
            cl.setContentsMargins(2, 2, 2, 2); cl.setSpacing(1)
            pr = QHBoxLayout(); pr.setSpacing(2)
            pr.addWidget(q_lbl); pr.addWidget(g_lbl)
            cl.addLayout(pr); cl.addWidget(cap_lbl)
            cell.setLayout(cl)
            self.samples_grid.addWidget(cell, i // COLS, i % COLS)

    def on_failed_ready(self, failed: list):
        """Populate the Failed Matches tab.

        Each entry: (query_path, pred_path, gt_path, caption).
        gt_path may be None when the ground-truth gallery image cannot be identified.
        """
        while self._failed_vbox.count():
            item = self._failed_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        MAX_SHOW = 100
        shown = failed[:MAX_SHOW]
        extra = f" — showing first {MAX_SHOW}" if len(failed) > MAX_SHOW else ""
        self.failed_label.setText(
            f"Failed matches (Hit@1=No): {len(failed)} total{extra}"
        )

        if not shown:
            placeholder = QLabel("No failed matches — all queries correct at rank 1.")
            placeholder.setAlignment(Qt.AlignCenter)
            self._failed_vbox.addWidget(placeholder)
            self._failed_vbox.addStretch()
            return

        THUMB = 256

        def _thumb_label(path, title, inspect_query_path=None):
            """Return a QWidget with a title and thumbnail. Hover to see full path.
            If inspect_query_path is given, this thumbnail is a candidate (not
            the query itself) — add 'Inspect S2'/'Inspect S3' buttons that jump
            to the corresponding Geo Verify Test tab with (query, this image)
            pre-filled and run immediately."""
            w = QWidget()
            v = QVBoxLayout()
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)
            t = QLabel(title)
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet("font-size: 10px; color: #444;")
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFixedSize(THUMB, THUMB)
            if path is not None:
                try:
                    pix = pil_to_pixmap(Image.open(path).convert("RGB"), THUMB)
                    img_lbl.setPixmap(pix)
                    # Extract location ID from path (parent dir for DenseUAV)
                    loc_id_text = f"loc_id: {Path(path).parent.name}"
                    img_lbl.setToolTip(f"{path}\n{loc_id_text}")
                except Exception as e:
                    img_lbl.setText("(error)")
                    img_lbl.setToolTip(str(path) if path else "None")
            else:
                img_lbl.setText("(none)")
                img_lbl.setToolTip("No ground truth image")
            v.addWidget(t)
            v.addWidget(img_lbl)
            if inspect_query_path is not None and path is not None:
                btn_row = QHBoxLayout()
                btn_row.setSpacing(2)
                for stage in (2, 3):
                    btn = QPushButton(f"Inspect S{stage}")
                    btn.setStyleSheet("font-size: 9px; padding: 1px 4px;")
                    btn.clicked.connect(
                        lambda checked=False, qp=inspect_query_path, cp=path, s=stage:
                        self._inspect_in_geo_verify(qp, cp, s))
                    btn_row.addWidget(btn)
                v.addLayout(btn_row)
            w.setLayout(v)
            return w

        for q_path, pred_path, gt_path, caption, pred_id, gt_id in shown:
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setStyleSheet("QFrame { background: #fafafa; border-radius: 4px; }")
            fh = QVBoxLayout()
            fh.setContentsMargins(8, 6, 8, 6)
            fh.setSpacing(4)

            # Show location IDs extracted from paths
            q_loc = Path(q_path).parent.name if q_path else "?"
            p_loc = Path(pred_path).parent.name if pred_path else "?"
            g_loc = Path(gt_path).parent.name if gt_path else "?"

            hdr = QLabel(f"<b>{caption}</b><br>Query loc={q_loc} | Pred loc={p_loc} (id {pred_id}) | GT loc={g_loc} (id {gt_id})")
            hdr.setStyleSheet("font-family: monospace; font-size: 9px; color: #333;")
            fh.addWidget(hdr)

            row = QHBoxLayout()
            row.setSpacing(10)
            row.addWidget(_thumb_label(q_path,   "Query"))
            row.addWidget(_thumb_label(pred_path, f"Predicted (Top-1)<br><b>id {pred_id}</b>",
                                       inspect_query_path=q_path))
            row.addWidget(_thumb_label(gt_path,   f"Ground Truth<br><b>id {gt_id}</b>",
                                       inspect_query_path=q_path))
            row.addStretch()
            fh.addLayout(row)
            frame.setLayout(fh)
            self._failed_vbox.addWidget(frame)

        self._failed_vbox.addStretch()

    def _inspect_in_geo_verify(self, query_path, gallery_path, stage: int):
        """Jump to the Geo Verify Test (Stage N) tab with (query, gallery)
        pre-filled from a Failed Matches entry and run it immediately —
        the "Inspect S2"/"Inspect S3" buttons under each Predicted/Ground
        Truth thumbnail call this."""
        if query_path is None or gallery_path is None:
            self.log.append("[Inspect] Missing query or gallery image path — nothing to inspect.")
            return
        w = self.geo_tabs[stage]
        w["query_input"].setText(str(query_path))
        w["gallery_input"].setText(str(gallery_path))
        self.main_tabs.setCurrentWidget(w["_page_widget"])
        self._run_geo_verify_test(stage)

    # ── Embedding helpers ──────────────────────────────────────────────────

    def _embed_distill(self, model, paths, device, batch_size, transform,
                       prog_range=None) -> np.ndarray | None:
        """Embed with distilled Swin-T: returns CLS token [N, 1024], no cluster/scale head."""
        ds = ImageDataset(paths, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            **loader_kwargs(device.type))
        n = max(1, len(loader)); chunks = []
        model.eval()
        with torch.inference_mode():
            for bi, (imgs, _) in enumerate(loader):
                if self.stop_event.is_set():
                    return None
                cls, _ = model(imgs.to(device, non_blocking=True))
                chunks.append(cls.cpu().float().numpy())
                if prog_range is not None:
                    p = prog_range[0] + (bi + 1) / n * (prog_range[1] - prog_range[0])
                    self.progress_changed.emit(int(p))
        return np.concatenate(chunks, axis=0) if chunks else None

    def _embed(self, model, paths, device, batch_size, transform,
               prog_range=None) -> np.ndarray | None:
        """Embed paths via cluster head. Returns float32 [N, D] or None if stopped."""
        ds = ImageDataset(paths, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            **loader_kwargs(device.type))
        n = max(1, len(loader)); chunks = []
        model.eval()
        with torch.inference_mode():
            for bi, (imgs, _) in enumerate(loader):
                if self.stop_event.is_set():
                    return None
                emb = model.encode_cluster_from_features(
                    model.encode_features(imgs.to(device, non_blocking=True)))
                chunks.append(emb.cpu().float().numpy())
                if prog_range is not None:
                    p = prog_range[0] + (bi + 1) / n * (prog_range[1] - prog_range[0])
                    self.progress_changed.emit(int(p))
        return np.concatenate(chunks, axis=0) if chunks else None

    def _embed_and_scale(self, model, paths, device, batch_size, transform,
                         prog_range=None):
        """Like _embed but also returns log_scale predictions."""
        ds = ImageDataset(paths, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            **loader_kwargs(device.type))
        n = max(1, len(loader)); emb_chunks, scl_chunks = [], []
        model.eval()
        with torch.inference_mode():
            for bi, (imgs, _) in enumerate(loader):
                if self.stop_event.is_set():
                    return None, None
                imgs = imgs.to(device, non_blocking=True)
                raw, log_scale = model.encode_features_and_scale(imgs)
                emb_chunks.append(model.encode_cluster_from_features(raw).cpu().float().numpy())
                scl_chunks.append(log_scale.cpu().float().numpy())
                if prog_range is not None:
                    p = prog_range[0] + (bi + 1) / n * (prog_range[1] - prog_range[0])
                    self.progress_changed.emit(int(p))
        if not emb_chunks:
            return None, None
        return np.concatenate(emb_chunks, axis=0), np.concatenate(scl_chunks, axis=0)

    def _embed_cached(self, tag: str, paths: list[Path], ckpt_path: Path,
                      backbone: str, model, device, batch_size, transform,
                      prog_range=None, with_scale: bool = False):
        """Embed with checkpoint-keyed cache. Returns (emb, scales_or_None)."""
        cp = _cache_path(tag, ckpt_path, [str(p) for p in paths], backbone)
        hit = _try_load_cache(cp, ckpt_path)
        if hit is not None:
            emb, scales = hit
            if with_scale and scales is None and backbone != "distill_swin_t":
                self.log_message.emit(f"  [{tag}] cache missing scales — re-extracting")
            else:
                self.log_message.emit(f"  [{tag}] loaded from cache ({len(emb)} embs)")
                if prog_range:
                    self.progress_changed.emit(int(prog_range[1]))
                return emb, scales

        if backbone == "distill_swin_t":
            emb    = self._embed_distill(model, paths, device, batch_size, transform, prog_range)
            scales = None
        elif with_scale:
            emb, scales = self._embed_and_scale(model, paths, device, batch_size, transform, prog_range)
        else:
            emb = self._embed(model, paths, device, batch_size, transform, prog_range)
            scales = None

        if emb is not None:
            _save_cache(cp, ckpt_path, emb, scales)
            self.log_message.emit(f"  [{tag}] cached → {cp.name}")
        return emb, scales

    def _extract_spatial_descriptors(self, model, paths, device, batch_size, transform,
                                     stage, prog_range=None):
        """Batched spatial descriptor extraction at a given backbone stage (see
        model.py's encode_spatial_features(stage=...) — free, shares the
        backbone pass already used for the pooled embedding/scale regardless
        of stage). Returns float32 [N,H,W,C] (e.g. [N,24,24,512] for stage 2,
        [N,12,12,1024] for stage 3, Swin-B/384) held temporarily in full
        precision — caller quantizes and discards it, not meant for long-term
        storage of a large gallery."""
        ds = ImageDataset(paths, transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            **loader_kwargs(device.type))
        n = max(1, len(loader)); chunks = []
        model.eval()
        with torch.inference_mode():
            for bi, (imgs, _) in enumerate(loader):
                if self.stop_event.is_set():
                    return None
                _, spatial = model.encode_spatial_features(
                    imgs.to(device, non_blocking=True), stage=stage)
                chunks.append(spatial.cpu().float().numpy())
                if prog_range is not None:
                    p = prog_range[0] + (bi + 1) / n * (prog_range[1] - prog_range[0])
                    self.progress_changed.emit(int(p))
        return np.concatenate(chunks, axis=0) if chunks else None

    def _vq_codebook_and_ids(self, model, device, batch_size, transform,
                             gallery_paths, query_paths, stage, k,
                             ckpt_path, backbone, log_tag="VQ"):
        """Shared K=k cell codebook + quantized [H,W] ID maps for the whole
        gallery and query sets, checkpoint-keyed-cached (same scheme as
        _embed_cached) — extraction (stage-N forward pass for the WHOLE
        gallery+queries), the K-means codebook fit, and quantization are all
        re-done from scratch on every eval run otherwise, which is expensive
        and pointless if nothing about the model/images/K/stage has changed.
        stage MUST be part of the cache key — stage 2 (24x24x512) and stage 3
        (12x12x1024) descriptors are incompatible; without this a switch from
        one stage to the other would silently load the wrong cached data.
        Cache labels keep the historical "denseuav_georank" prefix so existing
        caches stay valid; both the geo re-rank and the VQ map re-rank share
        these artifacts. Returns (codebook, gal_ids, qry_ids) or None if
        stopped."""
        import geo_verify

        codebook_cp = _cache_path(f"denseuav_georank_codebook_s{stage}_k{k}", ckpt_path,
                                  [str(p) for p in gallery_paths], backbone)
        gal_ids_cp = _cache_path(f"denseuav_georank_gal_ids_s{stage}_k{k}", ckpt_path,
                                 [str(p) for p in gallery_paths], backbone)
        qry_ids_cp = _cache_path(f"denseuav_georank_qry_ids_s{stage}_k{k}", ckpt_path,
                                 [str(p) for p in query_paths], backbone)
        codebook = _try_load_generic_cache(codebook_cp, ckpt_path)
        gal_ids = _try_load_generic_cache(gal_ids_cp, ckpt_path)
        qry_ids = _try_load_generic_cache(qry_ids_cp, ckpt_path)

        if codebook is not None and gal_ids is not None and qry_ids is not None:
            self.log_message.emit(
                f"{log_tag}: loaded stage-{stage} codebook (K={k}) + "
                f"quantized descriptors from cache")
            self.progress_changed.emit(97)
        else:
            self.log_message.emit(
                f"{log_tag}: extracting stage-{stage} spatial descriptors for "
                f"gallery ({len(gallery_paths)}) + queries ({len(query_paths)})...")
            gal_spatial = self._extract_spatial_descriptors(
                model, gallery_paths, device, batch_size, transform, stage, prog_range=(90, 94))
            if gal_spatial is None:
                return None
            qry_spatial = self._extract_spatial_descriptors(
                model, query_paths, device, batch_size, transform, stage, prog_range=(94, 96))
            if qry_spatial is None:
                return None

            n_g = gal_spatial.shape[0]
            self.log_message.emit(f"{log_tag}: fitting shared K={k} codebook "
                                  f"from a gallery sample...")
            rng = np.random.RandomState(0)
            sample_imgs = rng.choice(n_g, size=min(n_g, 300), replace=False)
            sample_tokens = gal_spatial[sample_imgs].reshape(-1, gal_spatial.shape[-1])
            if sample_tokens.shape[0] > 40000:
                idx = rng.choice(sample_tokens.shape[0], size=40000, replace=False)
                sample_tokens = sample_tokens[idx]
            codebook = geo_verify.fit_codebook(sample_tokens, n_clusters=k)

            self.log_message.emit(f"{log_tag}: quantizing gallery + query descriptors...")
            gal_ids = np.stack([
                geo_verify.quantize_spatial(torch.from_numpy(gal_spatial[i]), codebook)
                for i in range(n_g)])
            qry_ids = np.stack([
                geo_verify.quantize_spatial(torch.from_numpy(qry_spatial[i]), codebook)
                for i in range(qry_spatial.shape[0])])
            del gal_spatial, qry_spatial   # done with full-precision descriptors
            self.progress_changed.emit(97)

            _save_generic_cache(codebook_cp, ckpt_path, codebook)
            _save_generic_cache(gal_ids_cp, ckpt_path, gal_ids)
            _save_generic_cache(qry_ids_cp, ckpt_path, qry_ids)
            self.log_message.emit(f"{log_tag}: cached codebook + quantized descriptors "
                                  f"({gal_ids_cp.name}, {qry_ids_cp.name})")
        return codebook, gal_ids, qry_ids

    def _vq_map_rerank_denseuav(self, model, device, batch_size, transform,
                                gallery_paths, query_paths, sim, opts,
                                ckpt_path, backbone, query_ids, gallery_ids):
        """Blend-re-rank each query's coarse top-K using quantized semantic
        cell-ID maps (stage-N spatial descriptor -> [H,W] codebook IDs).

        map_score(query, candidate) = mean cell-wise agreement, maximised over
        the candidate map's 4 cardinal rotations (drone yaw is arbitrary vs a
        north-up satellite tile). NO flips in the rotation search — a mirrored
        layout must score LOW (mirror lookalikes are a known DenseUAV failure
        mode this should punish, not reward). "soft" scoring gives partial
        credit S[q_id, g_id] from the KxK centroid-cosine matrix (matters at
        small K, where near-identical content can straddle two clusters);
        "exact" counts only identical IDs.

        Final score within the top-K only: coarse_sim + alpha * map_score.
        The additive term is >= 0 and applied only to top-K entries, so no
        candidate outside the top-K can be displaced INTO it — R@10 is
        unchanged by construction; this is purely an R@1/R@5 play (offline
        validation: vq_rerank_test.py). Returns the adjusted [n_q, n_g] sim
        matrix (a sim.copy() — NOT -inf-initialized, see _geo_rerank_denseuav's
        mAP-corruption note), or None if stopped."""
        got = self._vq_codebook_and_ids(model, device, batch_size, transform,
                                        gallery_paths, query_paths,
                                        opts["stage"], opts["quantize_k"],
                                        ckpt_path, backbone, log_tag="VQ map re-rank")
        if got is None:
            return None
        codebook, gal_ids, qry_ids = got

        S = codebook @ codebook.T                    # [K,K] centroid cosine
        use_soft = opts["score"] == "soft"
        alpha = opts["alpha"]
        top_k = min(opts["top_k"], sim.shape[1])
        n_q = sim.shape[0]
        sim_rerank = sim.copy()
        top_idx = np.argsort(-sim, axis=1)[:, :top_k]
        gallery_arr = np.asarray(gallery_ids)

        self.log_message.emit(
            f"VQ map re-rank: blending top-{top_k} with "
            f"{opts['score']} map score (alpha={alpha:g}, {n_q} queries)...")
        tp_scores, fp_scores = [], []
        for qi in range(n_q):
            if self.stop_event.is_set():
                return None
            q_map = qry_ids[qi]
            q_flat = q_map.ravel()
            for ci in top_idx[qi]:
                best = -1.0
                for r in range(4):
                    g = np.rot90(gal_ids[ci], r)
                    if use_soft:
                        v = float(S[q_flat, g.ravel()].mean())
                    else:
                        v = float((q_map == g).mean())
                    best = max(best, v)
                sim_rerank[qi, ci] = sim[qi, ci] + alpha * best
                (tp_scores if gallery_arr[ci] == query_ids[qi]
                 else fp_scores).append(best)
            if qi % 200 == 0:
                self.progress_changed.emit(int(97 + (qi + 1) / n_q * 3))

        # TP/FP separation is the go/no-go diagnostic (lesson from the geo
        # re-rank attempts: an accept/score signal that doesn't separate true
        # from wrong locations can't help no matter how it's blended).
        for label, vals in (("TRUE match (same location)", tp_scores),
                            ("WRONG location", fp_scores)):
            if vals:
                v = np.asarray(vals)
                self.log_message.emit(
                    f"  [{label}] n={len(v)}, map_score mean={v.mean():.4f} "
                    f"median={np.median(v):.4f} p10={np.percentile(v, 10):.4f}")
        return sim_rerank

    def _vq_rerank_g4l(self, model, device, batch_size, transform,
                       gallery_paths, query_paths, gallery_names, gallery_seq,
                       sim, ckpt_path, backbone, query_names, pos_lookup):
        """VQ map re-rank for Game4Loc VisLoc, zoom-aware.

        Unlike DenseUAV/SUES/U-1652 (every gallery image the same physical
        footprint), VisLoc's gallery is a HIERARCHICAL multi-zoom tile
        pyramid: tile name encodes {area}_{zoom}_{x}_{y}, and tiles at
        different zoom levels cover different real-world extents even
        though every tile is resized to the same 384x384 input. A stage-3
        12x12 grid cell therefore represents a different physical patch
        size depending on the tile's zoom — comparing cell-wise agreement
        between a query and a candidate at a MISMATCHED zoom is comparing
        different spatial scales, not a meaningful signal.

        Fix: only apply the map-score bonus to candidates that share the
        SAME zoom level as the query's own coarse top-1 hit (a reasonable
        reference scale, since coarse cosine similarity already tends to
        prefer the zoom whose footprint best matches the query). Other-zoom
        candidates inside the top-K keep their coarse score untouched —
        still ranked, just not re-scored against an incommensurate grid.
        """
        opts_stage = self.denseuav_rerank_stage_input.currentData()
        opts_k = self.denseuav_rerank_k_input.value()
        top_k = min(self.denseuav_rerank_topk_input.value(), sim.shape[1])
        use_soft = self.denseuav_vqrank_score_input.currentData() == "soft"
        alpha = self.denseuav_vqrank_alpha_input.value()

        got = self._vq_codebook_and_ids(
            model, device, batch_size, transform, gallery_paths, query_paths,
            opts_stage, opts_k, ckpt_path, backbone, log_tag="VQ VisLoc")
        if got is None:
            return None
        codebook, gal_ids, qry_ids = got
        S = codebook @ codebook.T

        gallery_zoom = np.array([int(n.split("_")[1]) for n in gallery_names])
        sim_rerank = sim.copy()
        n_q = sim.shape[0]
        same_zoom_n = cross_zoom_n = 0
        tp_scores, fp_scores = [], []

        self.log_message.emit(
            f"VQ VisLoc re-rank: blending top-{top_k} with {'soft' if use_soft else 'exact'} "
            f"map score (alpha={alpha:g}), zoom-gated to each query's "
            f"coarse-top-1 zoom level ({n_q} queries)...")
        # Reference zoom per query = zoom of its own best-scoring candidate,
        # ignoring area restriction (this method isn't given the query's
        # area code directly — `sim` is passed in already un-area-masked,
        # matching how the final scoring separately re-applies the area
        # mask). Using the unmasked top-1's zoom as the reference is a
        # reasonable proxy: same-area retrieval dominates R@1 in practice,
        # so the globally-best-scoring tile is usually already in the
        # correct area, and even when it isn't, the ZOOM it favors is still
        # informative about the query's own apparent footprint scale.
        for qi in range(n_q):
            if self.stop_event.is_set():
                return None
            order = np.argsort(-sim[qi])[:top_k]
            ref_zoom = gallery_zoom[order[0]]
            q_map = qry_ids[qi]
            q_flat = q_map.ravel()
            qname = query_names[qi]
            positives = pos_lookup.get(qname, ())
            for ci in order:
                if gallery_zoom[ci] != ref_zoom:
                    cross_zoom_n += 1
                    continue
                same_zoom_n += 1
                best = -1.0
                for r in range(4):
                    g = np.rot90(gal_ids[ci], r)
                    if use_soft:
                        v = float(S[q_flat, g.ravel()].mean())
                    else:
                        v = float((q_map == g).mean())
                    best = max(best, v)
                sim_rerank[qi, ci] = sim[qi, ci] + alpha * best
                (tp_scores if gallery_names[ci] in positives
                 else fp_scores).append(best)
            if qi % 200 == 0:
                self.progress_changed.emit(int(90 + (qi + 1) / n_q * 8))

        total = same_zoom_n + cross_zoom_n
        self.log_message.emit(
            f"  Zoom gating: {same_zoom_n}/{total} candidate slots re-scored "
            f"(same zoom as coarse top-1), {cross_zoom_n} left at coarse "
            f"score (different zoom, scale-mismatched — not compared).")
        for label, vals in (("TRUE match (pos/semi-pos)", tp_scores),
                            ("WRONG location", fp_scores)):
            if vals:
                v = np.asarray(vals)
                self.log_message.emit(
                    f"  [{label}] n={len(v)}, map_score mean={v.mean():.4f} "
                    f"median={np.median(v):.4f} p10={np.percentile(v, 10):.4f}")
        return sim_rerank

    def _geo_rerank_denseuav(self, model, device, batch_size, transform,
                             gallery_paths, query_paths, sim, gal_scales, opts,
                             ckpt_path, backbone, query_ids, gallery_ids):
        """Geometric verification of each query's top-K coarse candidates using
        quantized spatial descriptors (opts["stage"], 2 or 3), used as a FILTER
        rather than a full re-rank: candidates the transform fit ACCEPTS keep
        their original coarse-similarity order; candidates it REJECTS (shift
        too large, or unfittable) are demoted below every accepted candidate,
        but keep their relative coarse order among themselves.

        This is deliberately conservative — an earlier version that ranked
        purely by RANSAC inlier ratio ("match confidence") or by shift_m
        catastrophically hurt R@1 (81.6%→~30%) on real DenseUAV data despite
        R@5/R@10 staying close to baseline: the true match was usually still
        in the top-20, just getting demoted by a confidence score that
        saturates near 1.0 too easily on stage 3's coarse 12x12 grid (few
        token matches + auto-escalating RANSAC threshold means almost any
        visually-similar-but-wrong candidate can look "confidently fit").
        Coarse similarity is already informative; geometric verification
        should refine it (catch clear false positives), not replace it.

        Returns sim_filtered — a sim.copy() with rejected top-K candidates
        demoted below the accepted ones; entries outside each query's top-K
        keep their exact coarse values (see the mAP-corruption note at the
        sim.copy() below). Returns None if stopped or extraction failed.
        """
        import geo_verify

        got = self._vq_codebook_and_ids(model, device, batch_size, transform,
                                        gallery_paths, query_paths,
                                        opts["stage"], opts["quantize_k"],
                                        ckpt_path, backbone, log_tag="Geo re-rank")
        if got is None:
            return None
        codebook, gal_ids, qry_ids = got

        gal_footprints = np.exp(np.asarray(gal_scales).reshape(-1))
        top_k = min(opts["top_k"], sim.shape[1])
        n_q = sim.shape[0]
        # Start from an EXACT copy of coarse sim (not -inf): this is a filter,
        # not an independent re-rank, so candidates never in a query's top-K
        # must keep their original coarse rank untouched — otherwise every
        # non-verified candidate collapses to a tied score, silently corrupting
        # mAP (which integrates precision over the WHOLE ranked gallery, not
        # just R@1/5/10's bounded top-20 window) even when R@1/5/10 look
        # unaffected. Caught via a real run where mAP shifted despite literally
        # every top-20 candidate being accepted (i.e. sim_filtered should have
        # been byte-identical to sim, but wasn't).
        sim_filtered = sim.copy()
        top_idx = np.argsort(-sim, axis=1)[:, :top_k]

        self.log_message.emit(f"Geo re-rank (filter): verifying top-{top_k} "
                              f"candidates per query ({n_q} queries)...")
        verify_kwargs = dict(
            estimate_scale=opts["estimate_scale"], reject_threshold_m=opts["reject_threshold_m"],
            sim_threshold=opts["sim_threshold"], ransac_thresh_px=opts["ransac_thresh_px"],
            min_inliers=opts["min_inliers"], scale_range=opts["scale_range"],
            auto_escalate_ransac=True, ransac_iters=opts["ransac_iters"],
            check_reflection=opts["check_reflection"],
            max_reflection_ratio=opts["max_reflection_ratio"])
        n_accepted_total = 0
        reason_counts = Counter()
        inlier_counts, n_matches_list, accepted_shifts = [], [], []
        # Stratify by ground truth (same location ID as the query, vs a
        # genuinely different/wrong location) — this is the real question:
        # does geometric verification actually tell true matches apart from
        # wrong ones, or does it accept/reject both at roughly the same rate?
        # (A near-identical accept rate + inlier/shift distribution between
        # the two groups would mean the filter has no real discriminating
        # power for this task, regardless of threshold tuning.)
        gallery_arr = np.asarray(gallery_ids)
        tp_stats = {"total": 0, "accepted": 0, "inliers": [], "shifts": []}
        fp_stats = {"total": 0, "accepted": 0, "inliers": [], "shifts": []}
        for qi in range(n_q):
            if self.stop_event.is_set():
                return None
            q_desc = geo_verify.reconstruct_from_ids(qry_ids[qi], codebook)
            cand_idx = top_idx[qi]
            cand_descs = [geo_verify.reconstruct_from_ids(gal_ids[ci], codebook)
                         for ci in cand_idx]
            cand_footprints = [float(gal_footprints[ci]) for ci in cand_idx]
            results = geo_verify.rerank_candidates(
                q_desc, cand_descs, cand_footprints, **verify_kwargs)
            cand_sims = sim[qi, cand_idx]
            # Margin guarantees every rejected candidate's adjusted score is
            # strictly below every accepted candidate's UNCHANGED coarse sim,
            # regardless of the sim matrix's scale (cosine [-1,1], Mahalanobis-
            # whitened, scale-penalized, etc) — computed per query, not a fixed
            # magic constant.
            margin = float(cand_sims.max() - cand_sims.min()) + 1e-3
            qid = query_ids[qi]
            for local_i, ci in enumerate(cand_idx):
                r = results[local_i]
                bucket = tp_stats if gallery_arr[ci] == qid else fp_stats
                bucket["total"] += 1
                if r["fit"] is not None:
                    inlier_counts.append(r["fit"]["inlier_count"])
                    n_matches_list.append(r["fit"]["n_matches"])
                    bucket["inliers"].append(r["fit"]["inlier_count"])
                if r["accept"]:
                    n_accepted_total += 1
                    bucket["accepted"] += 1
                    if r["shift_m"] is not None:
                        accepted_shifts.append(r["shift_m"])
                        bucket["shifts"].append(r["shift_m"])
                else:
                    sim_filtered[qi, ci] = sim[qi, ci] - margin  # demote below all accepted
                if r["reason"] == "ok":
                    reason_counts["ok (accepted)"] += 1
                elif r["reason"] == "too few token matches":
                    reason_counts["too few token matches"] += 1
                elif r["reason"].startswith("transform fit failed"):
                    reason_counts["too few inliers / fit failed"] += 1
                elif r["reason"].startswith("required shift"):
                    reason_counts["shift exceeded threshold"] += 1
                elif r["reason"].startswith("mirror-ambiguous"):
                    reason_counts["mirror-ambiguous (rejected)"] += 1
                else:
                    reason_counts[r["reason"]] += 1
            if qi % 20 == 0:
                self.progress_changed.emit(int(97 + (qi + 1) / n_q * 3))

        total_checks = n_q * top_k
        self.log_message.emit(
            f"Geo re-rank (filter): {n_accepted_total}/{total_checks} candidate "
            f"checks accepted ({n_accepted_total / max(1, total_checks) * 100:.1f}%)")
        for reason, count in reason_counts.most_common():
            self.log_message.emit(f"  - {reason}: {count} ({count / total_checks * 100:.1f}%)")
        if inlier_counts:
            ic, nm = np.array(inlier_counts), np.array(n_matches_list)
            self.log_message.emit(
                f"  RANSAC inlier_count (fits attempted, min_inliers={opts['min_inliers']}): "
                f"mean={ic.mean():.1f} median={np.median(ic):.0f} "
                f"p10={np.percentile(ic,10):.0f} max={ic.max()}")
            self.log_message.emit(
                f"  Mutual-NN n_matches (before RANSAC): mean={nm.mean():.1f} "
                f"median={np.median(nm):.0f} p10={np.percentile(nm,10):.0f} max={nm.max()}")
        if accepted_shifts:
            sv = np.array(accepted_shifts)
            self.log_message.emit(
                f"  Accepted shift_m (reject_threshold_m={opts['reject_threshold_m']:.0f}): "
                f"mean={sv.mean():.1f} median={np.median(sv):.1f} "
                f"p90={np.percentile(sv,90):.1f} max={sv.max():.1f}")
        for label, bucket in (("TRUE match (same location)", tp_stats),
                              ("WRONG location", fp_stats)):
            if bucket["total"] == 0:
                continue
            acc_rate = bucket["accepted"] / bucket["total"] * 100
            ic_mean = float(np.mean(bucket["inliers"])) if bucket["inliers"] else float("nan")
            sh_mean = float(np.mean(bucket["shifts"])) if bucket["shifts"] else float("nan")
            self.log_message.emit(
                f"  [{label}] n={bucket['total']}, accept_rate={acc_rate:.1f}%, "
                f"mean_inlier_count={ic_mean:.1f}, mean_accepted_shift_m={sh_mean:.2f}")
        return sim_filtered

    # ── Main eval loop ────────────────────────────────────────────────────

    def eval_loop(self):
        ds = self.dataset_combo.currentData()
        try:
            if ds == DATASET_SUES200:
                self._eval_sues200()
            elif ds == DATASET_U1652:
                self._eval_u1652()
            elif ds == DATASET_DENSEUAV:
                self._eval_denseuav()
            elif ds == DATASET_G4L:
                self._eval_g4l_visloc()
        except Exception as exc:
            import traceback
            self.log_message.emit(f"Error: {exc}\n{traceback.format_exc()}")
            self.eval_finished.emit("failed")

    # ── Game4Loc VisLoc evaluation (their exact protocol) ─────────────────

    def _eval_g4l_visloc(self):
        """Replicates Game4Loc's eval_visloc.py exactly (verified against
        their code 2026-07-18): per-query gallery restricted to the query's
        own map area, positives = pair_pos list only (queries without pos
        entries excluded from the denominator, matching their pos-mode
        loader), sklearn average-precision, SDM@K = rank-weighted
        exp(-0.001 * geodesic meters). Also reports a pos_semipos variant
        (all queries, semi-positives counted) for reference — that row is
        NOT comparable to their published numbers."""
        import visloc_eval as VE
        from sklearn.metrics import average_precision_score

        data_root = Path(self.root_input.text())
        test_json = self.g4l_json_input.text().strip() or "same-area-drone2sate-test.json"
        ckpt_path = Path(self.checkpoint_input.text())
        device = torch.device(self.device_input.currentData())
        batch_size = self.batch_size_input.value()

        meta = json.loads((data_root / test_json).read_text(encoding="utf-8"))
        entries = [e for e in meta if e.get("pair_pos_semipos_sate_img_list")]
        if not entries:
            raise ValueError(f"No pos_semipos entries in {test_json}")
        q_names = [e["drone_img_name"] for e in entries]
        q_locs = [tuple(e["drone_loc_lat_lon"]) for e in entries]
        q_paths = [data_root / e["drone_img_dir"] / e["drone_img_name"]
                   for e in entries]
        pos_lists = {e["drone_img_name"]: e.get("pair_pos_sate_img_list", [])
                     for e in entries}
        semi_lists = {e["drone_img_name"]: e["pair_pos_semipos_sate_img_list"]
                      for e in entries}

        sate_dir = data_root / (entries[0].get("sate_img_dir") or "satellite")
        g_names, g_locs = [], []
        for t in sorted(p.name for p in sate_dir.iterdir()
                        if p.suffix.lower() == ".png"):
            if t.split("_")[0] not in VE._SATE_LATLON:
                continue
            try:
                g_locs.append(VE._tile2sate(t))
            except Exception:
                continue
            g_names.append(t)
        g_paths = [sate_dir / n for n in g_names]
        g_seq = np.array([n.split("_")[0] for n in g_names])
        n2i = {n: i for i, n in enumerate(g_names)}
        self.log_message.emit(
            f"Game4Loc VisLoc protocol: {len(q_names)} queries "
            f"({sum(1 for q in q_names if pos_lists[q])} with pos entries) / "
            f"{len(g_names)} gallery tiles")

        model, backbone, transform = self._load_model(ckpt_path, device)
        g_emb, _ = self._embed_cached("g4l_gal", g_paths, ckpt_path, backbone,
                                      model, device, batch_size, transform,
                                      prog_range=(5, 55))
        if g_emb is None:
            self.eval_finished.emit("stopped"); return
        q_emb, _ = self._embed_cached("g4l_qry", q_paths, ckpt_path, backbone,
                                      model, device, batch_size, transform,
                                      prog_range=(55, 90))
        if q_emb is None:
            self.eval_finished.emit("stopped"); return
        gn = (g_emb / (np.linalg.norm(g_emb, axis=1, keepdims=True) + 1e-8))
        qn = (q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-8))
        sim = (qn @ gn.T).astype(np.float32)

        def _hav(a, b):
            la1, lo1 = a; la2, lo2 = b
            R = 6371000.0
            p1, p2 = math.radians(la1), math.radians(la2)
            dp = math.radians(la2 - la1); dl = math.radians(lo2 - lo1)
            h = (math.sin(dp/2)**2
                 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2)
            return 2 * R * math.asin(math.sqrt(h))

        def _protocol_eval(lists, subset):
            cmc = np.zeros(len(g_names))
            aps, s1, s3, s5 = [], [], [], []
            for i in subset:
                qname = q_names[i]
                score = sim[i] * (g_seq == qname.split("_")[0])
                order = np.argsort(-score)
                pos = np.array([n2i[p] for p in lists[qname] if p in n2i])
                good = np.isin(order, pos)
                if good.any():
                    aps.append(average_precision_score(
                        good.astype(int), np.arange(len(good), 0, -1)))
                    cmc[np.flatnonzero(good)[0]:] += 1
                for k, acc in ((1, s1), (3, s3), (5, s5)):
                    nom = den = 0.0
                    for r in range(k):
                        d = _hav(q_locs[i], g_locs[order[r]])
                        nom += (k - r) / np.exp(0.001 * d)
                        den += (k - r)
                    acc.append(nom / den)
            cmc = cmc / len(subset)
            return {"r1": float(cmc[0]), "r5": float(cmc[4]),
                    "r10": float(cmc[9]), "ap": float(np.mean(aps)),
                    "sdm1": float(np.mean(s1)), "sdm3": float(np.mean(s3)),
                    "sdm5": float(np.mean(s5)),
                    "n_queries": len(subset), "n_gallery": len(g_names)}

        pos_subset = [i for i, qname in enumerate(q_names) if pos_lists[qname]]
        final = {"pos": _protocol_eval(pos_lists, pos_subset),
                 "pos_semipos": _protocol_eval(semi_lists,
                                               list(range(len(q_names))))}
        for key, lbl in (("pos", "pos (their protocol)"),
                         ("pos_semipos", "pos_semipos (reference)")):
            r = final[key]
            self.log_message.emit(
                f"[{lbl}] n={r['n_queries']}  R@1={r['r1']*100:.2f}  "
                f"R@5={r['r5']*100:.2f}  R@10={r['r10']*100:.2f}  "
                f"AP={r['ap']*100:.2f}  SDM@1={r['sdm1']:.4f}  "
                f"SDM@3={r['sdm3']:.4f}")
        self.log_message.emit(
            "Game4Loc paper reference (VisLoc same-area FT, Table 4): "
            "R@1=80.20  R@5=96.53  AP=87.83  SDM@3=0.8546")

        if self.denseuav_vqrank_input.isChecked():
            if not hasattr(model, "encode_spatial_features"):
                self.log_message.emit(
                    "VQ map re-rank skipped: backbone doesn't support "
                    "encode_spatial_features (routed/distilled models).")
            else:
                sim_vq = self._vq_rerank_g4l(
                    model, device, batch_size, transform,
                    g_paths, q_paths, g_names, g_seq, sim, ckpt_path, backbone,
                    q_names, semi_lists)
                if sim_vq is None:
                    self.eval_finished.emit("stopped"); return
                sim_backup = sim
                sim = sim_vq   # _protocol_eval closes over `sim` by name
                final["pos_vq"] = _protocol_eval(pos_lists, pos_subset)
                final["pos_semipos_vq"] = _protocol_eval(
                    semi_lists, list(range(len(q_names))))
                sim = sim_backup
                for key, lbl in (("pos_vq", "pos + VQ re-rank"),
                                 ("pos_semipos_vq", "pos_semipos + VQ re-rank")):
                    r = final[key]
                    self.log_message.emit(
                        f"[{lbl}] n={r['n_queries']}  R@1={r['r1']*100:.2f}  "
                        f"R@5={r['r5']*100:.2f}  R@10={r['r10']*100:.2f}  "
                        f"AP={r['ap']*100:.2f}  SDM@1={r['sdm1']:.4f}  "
                        f"SDM@3={r['sdm3']:.4f}")

        self.progress_changed.emit(95)
        self.results_ready.emit(final, DATASET_G4L)

        if not self.no_history_input.isChecked() and self.history_input.text().strip():
            hp = Path(self.history_input.text().strip())
            if not hp.is_absolute(): hp = HERE / hp
            self._save_history(final, backbone, ckpt_path,
                               self.label_input.text().strip(), hp, DATASET_G4L)
        self.samples_ready.emit([])
        self.failed_ready.emit([])
        self.progress_changed.emit(100)
        self.eval_finished.emit("complete")

    # ── SUES-200 evaluation ───────────────────────────────────────────────

    def _eval_sues200(self):
        dataset_root = Path(self.root_input.text())
        gallery_override = self.sues_gallery_root_input.text().strip()
        # Official protocol (github.com/Reza-Zhu/SUES-200-Benchmark): the
        # gallery is ALL 200 locations' satellite tiles (120 train + 80 test,
        # train locations acting as confusion/distractor tiles), while
        # queries are only the 80 test-location drone images. Without this
        # override the gallery is silently just the test-split's own 80
        # satellite tiles — an easier, non-comparable 80-way eval instead of
        # the official 200-way one.
        sat_dir      = Path(gallery_override) if gallery_override else dataset_root / "satellite-view"
        drone_dir    = dataset_root / "drone_view_512"
        ckpt_path    = Path(self.checkpoint_input.text())
        device       = torch.device(self.device_input.currentData())
        batch_size   = self.batch_size_input.value()
        alt_filter   = self.altitude_input.currentData()

        model, backbone, transform = self._load_model(ckpt_path, device)

        # Collect gallery (satellite, 1 per location)
        sat_loc_dirs = sorted(d for d in sat_dir.iterdir() if d.is_dir())
        sat_paths, sat_ids = [], []
        for loc_dir in sat_loc_dirs:
            imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if imgs:
                sat_paths.append(imgs[0])
                sat_ids.append(int(loc_dir.name))
        if not sat_paths:
            raise FileNotFoundError(f"No satellite images in {sat_dir}")
        self.log_message.emit(
            f"Gallery: {len(sat_paths)} satellite images from {sat_dir}"
            + (" (official 200-location protocol)" if gallery_override else
               " (WARNING: test-split only — not the official 200-location gallery, "
               "set 'Gallery satellite root' for comparable numbers)"))
        self.progress_changed.emit(5)

        sat_emb, _ = self._embed_cached(
            "sues_gallery", sat_paths, ckpt_path, backbone,
            model, device, batch_size, transform, prog_range=(5, 20))
        if sat_emb is None:
            self.eval_finished.emit("stopped"); return
        sat_norm = (sat_emb / (np.linalg.norm(sat_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        self.log_message.emit(f"Gallery embedded: {sat_norm.shape}")

        # Collect drone queries
        altitudes_to_eval = [alt_filter] if alt_filter else SUES_ALTITUDES
        all_drone_paths, all_drone_labels, all_drone_alts = [], [], []
        for alt in altitudes_to_eval:
            count = 0
            for loc_dir in sorted(drone_dir.iterdir()):
                if not loc_dir.is_dir(): continue
                try: lid = int(loc_dir.name)
                except ValueError: continue
                alt_d = loc_dir / str(alt)
                if not alt_d.is_dir(): continue
                imgs = sorted(p for p in alt_d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                all_drone_paths.extend(imgs)
                all_drone_labels.extend([lid] * len(imgs))
                all_drone_alts.extend([alt] * len(imgs))
                count += len(imgs)
            self.log_message.emit(f"Altitude {alt} m: {count} drone images")

        total = len(all_drone_paths)
        if total == 0:
            raise ValueError("No drone images found.")
        self.log_message.emit(f"Total queries: {total}")

        alt_tag = f"alt{alt_filter or 'all'}"
        drone_emb, drone_scales = self._embed_cached(
            f"sues_drone_{alt_tag}", all_drone_paths, ckpt_path, backbone,
            model, device, batch_size, transform, prog_range=(25, 85), with_scale=True)
        if drone_emb is None:
            self.eval_finished.emit("stopped"); return

        drone_norm = (drone_emb / (np.linalg.norm(drone_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        # Full [n_queries, n_gallery] similarity instead of a faiss top-10
        # search: the gallery is at most 200 tiles, and the full matrix is
        # what the VQ re-rank operates on.
        sim = drone_norm @ sat_norm.T
        I = np.argsort(-sim, axis=1)[:, :10]
        self.progress_changed.emit(90)

        # Per-altitude recall
        results_per_alt = {alt: {"r1": 0, "r5": 0, "r10": 0, "queries": 0}
                           for alt in altitudes_to_eval}
        sample_correct, sample_wrong = [], []
        failed_entries = []
        sat_id_to_path = {sid: sp for sid, sp in zip(sat_ids, sat_paths)}

        for i, (label, alt) in enumerate(zip(all_drone_labels, all_drone_alts)):
            pred_ids = [sat_ids[idx] for idx in I[i] if idx >= 0]
            r = results_per_alt[alt]; r["queries"] += 1
            if pred_ids and pred_ids[0] == label: r["r1"] += 1
            if label in pred_ids[:5]:  r["r5"]  += 1
            if label in pred_ids[:10]: r["r10"] += 1
            top1_path = sat_id_to_path.get(pred_ids[0]) if pred_ids else None
            correct = bool(pred_ids and pred_ids[0] == label)
            caption = f"{all_drone_paths[i].parts[-3]} / {alt}m"
            entry = (all_drone_paths[i], top1_path, correct, caption)
            (sample_correct if correct else sample_wrong).append(entry)
            if not correct:
                gt_path = sat_id_to_path.get(label)
                pred_id = pred_ids[0] if pred_ids else None
                failed_entries.append((all_drone_paths[i], top1_path, gt_path, caption, pred_id, label))

        final = {}
        total_r1 = total_r5 = total_r10 = total_q = 0
        for alt in altitudes_to_eval:
            r = results_per_alt[alt]; q = r["queries"]
            if q == 0: continue
            final[alt] = {"queries": q, "r1": r["r1"]/q, "r5": r["r5"]/q, "r10": r["r10"]/q}
            total_r1 += r["r1"]; total_r5 += r["r5"]; total_r10 += r["r10"]; total_q += q
            self.log_message.emit(
                f"Alt {alt} m — R@1: {r['r1']/q*100:.1f}%  R@5: {r['r5']/q*100:.1f}%  "
                f"R@10: {r['r10']/q*100:.1f}%  ({q} queries)")
        if total_q > 0:
            final["overall"] = {"queries": total_q,
                                "r1": total_r1/total_q, "r5": total_r5/total_q,
                                "r10": total_r10/total_q}

        if self.denseuav_vqrank_input.isChecked():
            if not hasattr(model, "encode_spatial_features"):
                self.log_message.emit(
                    "VQ map re-rank skipped: backbone doesn't support "
                    "encode_spatial_features (routed/distilled models).")
            else:
                vq_opts = dict(
                    stage=self.denseuav_rerank_stage_input.currentData(),
                    quantize_k=self.denseuav_rerank_k_input.value(),
                    top_k=self.denseuav_rerank_topk_input.value(),
                    score=self.denseuav_vqrank_score_input.currentData(),
                    alpha=self.denseuav_vqrank_alpha_input.value(),
                )
                sim_vq = self._vq_map_rerank_denseuav(
                    model, device, batch_size, transform,
                    sat_paths, all_drone_paths, sim, vq_opts,
                    ckpt_path, backbone, all_drone_labels, sat_ids)
                if sim_vq is None:
                    self.eval_finished.emit("stopped"); return
                I_vq = np.argsort(-sim_vq, axis=1)[:, :10]
                vq_alt = {alt: {"r1": 0, "r5": 0, "r10": 0, "q": 0}
                          for alt in altitudes_to_eval}
                for i, (label, alt) in enumerate(zip(all_drone_labels, all_drone_alts)):
                    pred_ids = [sat_ids[idx] for idx in I_vq[i]]
                    r = vq_alt[alt]; r["q"] += 1
                    if pred_ids[0] == label:       r["r1"] += 1
                    if label in pred_ids[:5]:      r["r5"] += 1
                    if label in pred_ids[:10]:     r["r10"] += 1
                vr1 = vr5 = vr10 = vq_q = 0
                for alt in altitudes_to_eval:
                    r = vq_alt[alt]
                    if r["q"] == 0: continue
                    self.log_message.emit(
                        f"[VQ re-rank] Alt {alt} m — R@1: {r['r1']/r['q']*100:.1f}%  "
                        f"R@5: {r['r5']/r['q']*100:.1f}%  R@10: {r['r10']/r['q']*100:.1f}%")
                    vr1 += r["r1"]; vr5 += r["r5"]; vr10 += r["r10"]; vq_q += r["q"]
                if vq_q > 0:
                    final["vq_overall"] = {"queries": vq_q, "r1": vr1/vq_q,
                                           "r5": vr5/vq_q, "r10": vr10/vq_q}
                    self.log_message.emit(
                        f"[VQ map re-rank: {vq_opts['score']}, a={vq_opts['alpha']:g}, "
                        f"K={vq_opts['quantize_k']}, s{vq_opts['stage']}]  Overall "
                        f"R@1={vr1/vq_q*100:.2f}%  R@5={vr5/vq_q*100:.2f}%  "
                        f"R@10={vr10/vq_q*100:.2f}%")

        # Scale prediction accuracy
        if drone_scales is not None:
            _SUES_FOV_H = 84.0
            _tan_hfov = math.tan(math.radians(_SUES_FOV_H) / 2.0)
            alts_arr = np.array(all_drone_alts)
            for alt in altitudes_to_eval:
                mask = alts_arr == alt
                if mask.any() and alt in final:
                    s = drone_scales[mask]
                    gt_log = math.log(2.0 * alt * _tan_hfov)
                    gt_m   = math.exp(gt_log)
                    pred_m = np.exp(s)
                    mae_m  = float(np.abs(pred_m - gt_m).mean())
                    rel_err = float((np.abs(pred_m - gt_m) / gt_m).mean()) * 100.0
                    final[alt].update({"mean_log_scale": float(s.mean()),
                                       "gt_log_scale": gt_log,
                                       "scale_mae_m": mae_m,
                                       "scale_rel_err_pct": rel_err})
            alts_with_scale = [a for a in sorted(altitudes_to_eval)
                               if "mean_log_scale" in final.get(a, {})]
            if len(alts_with_scale) >= 2:
                pairs_ok = sum(final[a1]["mean_log_scale"] < final[a2]["mean_log_scale"]
                               for i, a1 in enumerate(alts_with_scale)
                               for a2 in alts_with_scale[i+1:])
                n_pairs = len(alts_with_scale) * (len(alts_with_scale) - 1) // 2
                pred_means = [final[a]["mean_log_scale"] for a in alts_with_scale]
                gt_logs    = [final[a]["gt_log_scale"]   for a in alts_with_scale]
                disc = (max(pred_means)-min(pred_means)) / (max(gt_logs)-min(gt_logs)) * 100
                if "overall" in final:
                    final["overall"].update({
                        "scale_ordinal_acc":   pairs_ok / n_pairs,
                        "scale_mae_m":         float(np.mean([final[a]["scale_mae_m"] for a in alts_with_scale])),
                        "scale_rel_err_pct":   float(np.mean([final[a]["scale_rel_err_pct"] for a in alts_with_scale])),
                        "scale_discrimination": disc,
                    })

        self.results_ready.emit(final, DATASET_SUES200)

        if not self.no_history_input.isChecked() and self.history_input.text().strip():
            hp = Path(self.history_input.text().strip())
            if not hp.is_absolute(): hp = HERE / hp
            self._save_history(final, backbone, ckpt_path, self.label_input.text().strip(),
                               hp, DATASET_SUES200)

        random.shuffle(sample_correct); random.shuffle(sample_wrong)
        half = MAX_SAMPLES_DISPLAY // 2
        samples = sample_wrong[:half] + sample_correct[:half]
        random.shuffle(samples)
        self.samples_ready.emit(samples[:MAX_SAMPLES_DISPLAY])
        random.shuffle(failed_entries)
        self.failed_ready.emit(failed_entries)
        self.progress_changed.emit(100)
        self.eval_finished.emit("complete")

    # ── University-1652 evaluation ────────────────────────────────────────

    def _eval_u1652(self):
        dataset_root = Path(self.root_input.text())
        test_dir     = dataset_root / "test"
        ckpt_path    = Path(self.checkpoint_input.text())
        device       = torch.device(self.device_input.currentData())
        batch_size   = self.batch_size_input.value()
        direction    = self.u1652_direction.currentData()   # "d2s" or "s2d"

        model, backbone, transform = self._load_model(ckpt_path, device)

        if direction == "d2s":
            query_dir   = test_dir / "query_drone"
            gallery_dir = test_dir / "gallery_satellite"
        else:
            query_dir   = test_dir / "query_satellite"
            gallery_dir = test_dir / "gallery_drone"

        dir_label = "Drone→Satellite" if direction == "d2s" else "Satellite→Drone"
        self.log_message.emit(f"University-1652  {dir_label}")

        # Collect gallery
        gallery_paths, gallery_ids = [], []
        for loc_dir in sorted(d for d in gallery_dir.iterdir() if d.is_dir()):
            imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            gallery_paths.extend(imgs)
            gallery_ids.extend([loc_dir.name] * len(imgs))
        if not gallery_paths:
            raise FileNotFoundError(f"No gallery images in {gallery_dir}")
        self.log_message.emit(f"Gallery: {len(gallery_paths)} images from "
                              f"{len(set(gallery_ids))} locations")
        self.progress_changed.emit(5)

        # Collect queries
        query_paths, query_ids = [], []
        for loc_dir in sorted(d for d in query_dir.iterdir() if d.is_dir()):
            imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            query_paths.extend(imgs)
            query_ids.extend([loc_dir.name] * len(imgs))
        if not query_paths:
            raise FileNotFoundError(f"No query images in {query_dir}")
        self.log_message.emit(f"Queries: {len(query_paths)} images from "
                              f"{len(set(query_ids))} locations")

        # Embed gallery (progress 5 → 40)
        gal_emb, _ = self._embed_cached(
            f"u1652_gal_{direction}", gallery_paths, ckpt_path, backbone,
            model, device, batch_size, transform, prog_range=(5, 40))
        if gal_emb is None:
            self.eval_finished.emit("stopped"); return
        gal_norm = (gal_emb / (np.linalg.norm(gal_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        # Embed queries (progress 40 → 85)
        qry_emb, _ = self._embed_cached(
            f"u1652_qry_{direction}", query_paths, ckpt_path, backbone,
            model, device, batch_size, transform, prog_range=(40, 85))
        if qry_emb is None:
            self.eval_finished.emit("stopped"); return
        qry_norm = (qry_emb / (np.linalg.norm(qry_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

        self.log_message.emit("Computing similarity and metrics...")
        sim = qry_norm @ gal_norm.T   # [Q, G]
        self.progress_changed.emit(90)

        recalls, ap, n_q = _recall_and_ap(sim, query_ids, gallery_ids, ks=(1, 5, 10))
        n_g = len(gallery_paths)
        self.log_message.emit(
            f"{dir_label}  ({n_q} queries / {n_g} gallery)  "
            f"R@1={recalls[1]*100:.2f}%  R@5={recalls[5]*100:.2f}%  "
            f"R@10={recalls[10]*100:.2f}%  mAP={ap*100:.2f}%")

        final = {direction: {"r1": recalls[1], "r5": recalls[5], "r10": recalls[10],
                             "ap": ap, "n_queries": n_q, "n_gallery": n_g}}

        if self.denseuav_vqrank_input.isChecked():
            if not hasattr(model, "encode_spatial_features"):
                self.log_message.emit(
                    "VQ map re-rank skipped: backbone doesn't support "
                    "encode_spatial_features (routed/distilled models).")
            else:
                vq_opts = dict(
                    stage=self.denseuav_rerank_stage_input.currentData(),
                    quantize_k=self.denseuav_rerank_k_input.value(),
                    top_k=self.denseuav_rerank_topk_input.value(),
                    score=self.denseuav_vqrank_score_input.currentData(),
                    alpha=self.denseuav_vqrank_alpha_input.value(),
                )
                sim_vq = self._vq_map_rerank_denseuav(
                    model, device, batch_size, transform,
                    gallery_paths, query_paths, sim, vq_opts,
                    ckpt_path, backbone, query_ids, gallery_ids)
                if sim_vq is None:
                    self.eval_finished.emit("stopped"); return
                # exclude_junk=True — same distractor handling as the baseline
                # row above, so the two rows are directly comparable.
                rec_v, ap_v, _ = _recall_and_ap(sim_vq, query_ids, gallery_ids,
                                                ks=(1, 5, 10))
                self.log_message.emit(
                    f"[VQ map re-rank: {vq_opts['score']}, a={vq_opts['alpha']:g}, "
                    f"K={vq_opts['quantize_k']}, s{vq_opts['stage']}]  "
                    f"R@1={rec_v[1]*100:.2f}%  R@5={rec_v[5]*100:.2f}%  "
                    f"R@10={rec_v[10]*100:.2f}%  mAP={ap_v*100:.2f}%")
                final[f"{direction}_vqrank"] = {
                    "r1": rec_v[1], "r5": rec_v[5], "r10": rec_v[10],
                    "ap": ap_v, "n_queries": n_q, "n_gallery": n_g}

        self.results_ready.emit(final, DATASET_U1652)

        if not self.no_history_input.isChecked() and self.history_input.text().strip():
            hp = Path(self.history_input.text().strip())
            if not hp.is_absolute(): hp = HERE / hp
            self._save_history(final, backbone, ckpt_path, self.label_input.text().strip(),
                               hp, DATASET_U1652)

        # Sample matches — pick query images and their top-1 gallery match
        top_idx = np.argsort(-sim, axis=1)
        gal_arr = np.array(gallery_ids)
        query_id_set = set(query_ids)
        junk_mask = np.array([gid not in query_id_set for gid in gallery_ids])
        sample_correct, sample_wrong = [], []
        failed_entries = []
        for q_i in range(len(query_paths)):
            ranked_idx = top_idx[q_i][~junk_mask[top_idx[q_i]]]  # remove distractors
            top1_gal_idx = ranked_idx[0]
            best_gal = gallery_paths[top1_gal_idx]
            correct  = (gal_arr[top1_gal_idx] == query_ids[q_i])
            caption  = f"id {query_ids[q_i]}"
            entry = (query_paths[q_i], best_gal, correct, caption)
            (sample_correct if correct else sample_wrong).append(entry)
            if not correct:
                gt_indices = np.where(gal_arr == query_ids[q_i])[0]
                gt_path = gallery_paths[gt_indices[0]] if len(gt_indices) > 0 else None
                pred_id = gal_arr[top1_gal_idx]
                failed_entries.append((query_paths[q_i], best_gal, gt_path, caption, pred_id, query_ids[q_i]))

        random.shuffle(sample_correct); random.shuffle(sample_wrong)
        half = MAX_SAMPLES_DISPLAY // 2
        samples = sample_wrong[:half] + sample_correct[:half]
        random.shuffle(samples)
        self.samples_ready.emit(samples[:MAX_SAMPLES_DISPLAY])
        random.shuffle(failed_entries)
        self.failed_ready.emit(failed_entries)
        self.progress_changed.emit(100)
        self.eval_finished.emit("complete")

    # ── DenseUAV evaluation ───────────────────────────────────────────────

    def _eval_denseuav(self):
        dataset_root = Path(self.root_input.text())
        test_dir = dataset_root / "test"
        ckpt_path = Path(self.checkpoint_input.text())
        device = torch.device(self.device_input.currentData())
        batch_size = self.batch_size_input.value()
        direction = self.u1652_direction.currentData()

        model, backbone, transform = self._load_model(ckpt_path, device)

        if direction == "d2s":
            query_dir = test_dir / "query_drone"
            gallery_dir = test_dir / "gallery_satellite"
        else:
            query_dir = test_dir / "query_satellite"
            gallery_dir = test_dir / "gallery_drone"

        dir_label = "Drone→Satellite" if direction == "d2s" else "Satellite→Drone"
        self.log_message.emit(f"DenseUAV  {dir_label}")

        gallery_paths, gallery_ids = [], []
        for loc_dir in sorted(d for d in gallery_dir.iterdir() if d.is_dir()):
            imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            gallery_paths.extend(imgs)
            gallery_ids.extend([loc_dir.name] * len(imgs))
        if not gallery_paths:
            raise FileNotFoundError(f"No gallery images in {gallery_dir}")
        self.log_message.emit(f"Gallery: {len(gallery_paths)} images from "
                              f"{len(set(gallery_ids))} locations")
        self.progress_changed.emit(5)

        query_paths, query_ids = [], []
        for loc_dir in sorted(d for d in query_dir.iterdir() if d.is_dir()):
            imgs = sorted(p for p in loc_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            query_paths.extend(imgs)
            query_ids.extend([loc_dir.name] * len(imgs))
        if not query_paths:
            raise FileNotFoundError(f"No query images in {query_dir}")
        self.log_message.emit(f"Queries: {len(query_paths)} images from "
                              f"{len(set(query_ids))} locations")

        # DEBUG: Check location ID overlap
        query_id_set = set(query_ids)
        gallery_id_set = set(gallery_ids)
        overlap = query_id_set & gallery_id_set
        self.log_message.emit(f"DEBUG: Query IDs {min(query_id_set)}→{max(query_id_set)}, "
                              f"Gallery IDs {min(gallery_id_set)}→{max(gallery_id_set)}, "
                              f"Overlap: {len(overlap)}/{len(query_id_set)} locations")

        if backbone == "routed_swin_b":
            # Coarse route → per-cluster experts → within-top-k-cluster ranking.
            self.log_message.emit("Routed cluster-expert retrieval...")
            sim = self._denseuav_routed_sim(
                model, gallery_paths, gallery_ids, query_paths, device, batch_size, transform)
            if sim is None:
                self.eval_finished.emit("stopped"); return
            qry_scales = gal_scales = None
            self.progress_changed.emit(90)
        else:
            # Extract embeddings and scales (with scale head)
            gal_emb, gal_scales = self._embed_cached(
                f"denseuav_gal_{direction}", gallery_paths, ckpt_path, backbone,
                model, device, batch_size, transform, prog_range=(5, 40), with_scale=True)
            if gal_emb is None:
                self.eval_finished.emit("stopped"); return
            gal_norm = (gal_emb / (np.linalg.norm(gal_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

            qry_emb, qry_scales = self._embed_cached(
                f"denseuav_qry_{direction}", query_paths, ckpt_path, backbone,
                model, device, batch_size, transform, prog_range=(40, 85), with_scale=True)
            if qry_emb is None:
                self.eval_finished.emit("stopped"); return
            qry_norm = (qry_emb / (np.linalg.norm(qry_emb, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

            # Extract cluster assignments (top-kc predictions)
            kc = min(5, len(set(gallery_ids)))  # Top-5 clusters or fewer
            gal_cluster_topk = np.argsort(-gal_emb, axis=1)[:, :kc]  # Top-kc indices by embedding magnitude
            qry_cluster_topk = np.argsort(-qry_emb, axis=1)[:, :kc]

            self.log_message.emit(f"Extracted scales: gallery={gal_scales.shape if gal_scales is not None else None}, "
                                f"query={qry_scales.shape if qry_scales is not None else None}")
            self.log_message.emit(f"Extracted cluster top-{kc}: gallery={gal_cluster_topk.shape}, query={qry_cluster_topk.shape}")

            self.log_message.emit("Computing similarity and metrics...")
            if self.denseuav_mahalanobis_input.isChecked():
                shrink = self.denseuav_maha_shrinkage_input.value()
                self.log_message.emit(
                    f"Mahalanobis whitening (gallery covariance, shrinkage={shrink})...")
                qry_white, gal_white = whiten_embeddings(gal_emb, qry_emb, shrinkage=shrink)
                sim = qry_white @ gal_white.T
            else:
                sim = qry_norm @ gal_norm.T
            self.progress_changed.emit(90)

        # Apply soft scale penalty: penalize matches with large scale/altitude mismatch
        width = self.denseuav_scale_penalty_width.value()
        if qry_scales is not None and gal_scales is not None and width > 0:
            # Compute scale difference (log10 scale in meters)
            scale_diff = np.abs(qry_scales[:, None] - gal_scales[None, :])  # (n_query, n_gallery)

            # Soft penalty using tanh: 1 - tanh(diff/width)
            # width controls how sharp the penalty ramp is (smaller = sharper)
            scale_penalty = 1.0 - np.tanh(scale_diff / width)  # maps [0, inf) → [0, 1)

            # Apply penalty to similarity
            sim = sim * scale_penalty
            self.log_message.emit(f"Applied soft scale penalty (width={width}) — {dir_label}")

            # Analyze scale predictions
            scale_diffs = np.abs(qry_scales - gal_scales[np.argsort(-sim, axis=1)[:, 0]])
            self.log_message.emit(f"Scale head analysis: mean_diff={np.mean(scale_diffs):.3f}, "
                                f"std_diff={np.std(scale_diffs):.3f}, max_diff={np.max(scale_diffs):.3f}")
        elif width == 0:
            self.log_message.emit("Scale penalty disabled (width=0)")

        # GPS is used for SDM@1 (official DenseUAV spatial-distance metric) and the
        # distance display in the failed-match preview. The hit condition itself is
        # exact location-ID match (official protocol). Prefer Dense_GPS_ALL.txt: it
        # covers all 3,033 gallery locations incl. distractors, which SDM needs when
        # the top-1 is a distractor tile (Dense_GPS_test.txt covers only 777 queries).
        gps_test_file = test_dir.parent / "Dense_GPS_ALL.txt"
        if not gps_test_file.exists():
            gps_test_file = test_dir.parent / "Dense_GPS_test.txt"
        loc_gps_map = {}  # location_id -> (lat, lon)

        if gps_test_file.exists():
            self.log_message.emit(f"Loading GPS coordinates ({gps_test_file.name})...")
            with open(gps_test_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        path = parts[0]
                        lon_str = parts[1].lstrip('EW')
                        lat_str = parts[2].lstrip('NS')
                        try:
                            lat, lon = float(lat_str), float(lon_str)
                            if parts[2].startswith('S'): lat = -lat
                            if parts[1].startswith('W'): lon = -lon
                            loc_id = Path(path).parent.name
                            if loc_id not in loc_gps_map:
                                loc_gps_map[loc_id] = (lat, lon)
                        except ValueError:
                            pass
            # Map query and gallery IDs to GPS using location_id (display only)
            query_gps_full = {qid: v for qid in set(query_ids)
                              if (v := loc_gps_map.get(qid)) is not None}
            gallery_gps_full = {gid: v for gid in set(gallery_ids)
                                if (v := loc_gps_map.get(gid)) is not None}

        # Hit condition: EXACT location-ID match over the FULL confusion gallery
        # (official DenseUAV protocol — comparable to reference-paper numbers).
        recalls, ap, n_q = _recall_and_ap(sim, query_ids, gallery_ids,
                                          ks=(1, 5, 10), exclude_junk=False)
        self.log_message.emit("Hit condition: exact location ID, full confusion gallery "
                              "(official DenseUAV protocol)")

        # SDM@1 — official DenseUAV spatial-distance metric, replicated faithfully
        # from denseuav_repo/evaluateDistance.py: score = exp(-5000 * d) with d the
        # raw lat/lon DEGREE-space Euclidean distance between query and top-1 GPS
        # (deliberately no cos(latitude) correction — the official code omits it).
        # e-folding scale ≈ 22 m: exact hit = 1.0, 20 m neighbor ≈ 0.41, 100 m ≈ 0.01.
        sdm1 = None
        if loc_gps_map:
            top1 = np.argmax(sim, axis=1)
            scores, n_skipped = [], 0
            for q_i, qid in enumerate(query_ids):
                qg = loc_gps_map.get(qid)
                gg = loc_gps_map.get(gallery_ids[top1[q_i]])
                if qg is None or gg is None:
                    n_skipped += 1
                    continue
                d_deg = math.sqrt((qg[0] - gg[0]) ** 2 + (qg[1] - gg[1]) ** 2)
                scores.append(math.exp(-5000.0 * d_deg))
            if scores:
                sdm1 = float(np.mean(scores))
                self.log_message.emit(
                    f"SDM@1 = {sdm1*100:.2f}%  ({len(scores)} queries"
                    + (f", {n_skipped} skipped without GPS" if n_skipped else "") + ")")

        # Cluster head analysis (similarity distribution)
        top1_sim = np.max(sim, axis=1)
        top5_sim = np.partition(-sim, 4, axis=1)[:, :5]
        self.log_message.emit(f"\n=== Cluster Head Analysis ===")
        self.log_message.emit(f"Top-1 similarity: mean={np.mean(top1_sim):.4f}, std={np.std(top1_sim):.4f}")
        self.log_message.emit(f"Top-5 similarity spread: mean={np.mean(-top5_sim[:, -1]):.4f}")

        n_g = len(gallery_paths)
        self.log_message.emit(
            f"{dir_label}  ({n_q} queries / {n_g} gallery)  "
            f"R@1={recalls[1]*100:.2f}%  R@5={recalls[5]*100:.2f}%  "
            f"R@10={recalls[10]*100:.2f}%  mAP={ap*100:.2f}%"
            + (f"  SDM@1={sdm1*100:.2f}%" if sdm1 is not None else ""))

        final = {direction: {"r1": recalls[1], "r5": recalls[5], "r10": recalls[10],
                             "ap": ap, "n_queries": n_q, "n_gallery": n_g}}
        if sdm1 is not None:
            final[direction]["sdm1"] = sdm1

        if self.denseuav_georank_input.isChecked():
            if not hasattr(model, "encode_spatial_features"):
                self.log_message.emit(
                    "Geo re-rank skipped: backbone doesn't support encode_spatial_features "
                    "(routed/distilled models).")
            elif gal_scales is None or qry_scales is None:
                self.log_message.emit(
                    "Geo re-rank skipped: no per-image scale predictions available "
                    "(routed mode disables the scale head here).")
            else:
                opts = dict(
                    stage=self.denseuav_rerank_stage_input.currentData(),
                    estimate_scale=self.denseuav_georank_estimate_scale_input.isChecked(),
                    top_k=self.denseuav_rerank_topk_input.value(),
                    sim_threshold=self.denseuav_georank_sim_input.value(),
                    ransac_thresh_px=self.denseuav_georank_ransac_input.value(),
                    ransac_iters=self.denseuav_georank_ransac_iters_input.value(),
                    min_inliers=self.denseuav_georank_min_inliers_input.value(),
                    scale_range=(self.denseuav_georank_scale_min_input.value(),
                                self.denseuav_georank_scale_max_input.value()),
                    reject_threshold_m=self.denseuav_georank_reject_input.value(),
                    quantize_k=self.denseuav_rerank_k_input.value(),
                    check_reflection=self.denseuav_georank_check_reflection_input.isChecked(),
                    max_reflection_ratio=self.denseuav_georank_reflection_ratio_input.value(),
                )
                sim_filtered = self._geo_rerank_denseuav(
                    model, device, batch_size, transform,
                    gallery_paths, query_paths, sim, gal_scales, opts,
                    ckpt_path, backbone, query_ids, gallery_ids)
                if sim_filtered is None:
                    self.eval_finished.emit("stopped"); return
                rec_f, ap_f, _ = _recall_and_ap(sim_filtered, query_ids, gallery_ids,
                                                ks=(1, 5, 10), exclude_junk=False)
                self.log_message.emit(
                    f"[Geo re-rank: filtered]  R@1={rec_f[1]*100:.2f}%  "
                    f"R@5={rec_f[5]*100:.2f}%  R@10={rec_f[10]*100:.2f}%  mAP={ap_f*100:.2f}%")
                final[f"{direction}_georank_filtered"] = {
                    "r1": rec_f[1], "r5": rec_f[5], "r10": rec_f[10],
                    "ap": ap_f, "n_queries": n_q, "n_gallery": n_g}

        if self.denseuav_vqrank_input.isChecked():
            if not hasattr(model, "encode_spatial_features"):
                self.log_message.emit(
                    "VQ map re-rank skipped: backbone doesn't support "
                    "encode_spatial_features (routed/distilled models).")
            else:
                vq_opts = dict(
                    stage=self.denseuav_rerank_stage_input.currentData(),
                    quantize_k=self.denseuav_rerank_k_input.value(),
                    top_k=self.denseuav_rerank_topk_input.value(),
                    score=self.denseuav_vqrank_score_input.currentData(),
                    alpha=self.denseuav_vqrank_alpha_input.value(),
                )
                sim_vq = self._vq_map_rerank_denseuav(
                    model, device, batch_size, transform,
                    gallery_paths, query_paths, sim, vq_opts,
                    ckpt_path, backbone, query_ids, gallery_ids)
                if sim_vq is None:
                    self.eval_finished.emit("stopped"); return
                rec_v, ap_v, _ = _recall_and_ap(sim_vq, query_ids, gallery_ids,
                                                ks=(1, 5, 10), exclude_junk=False)
                self.log_message.emit(
                    f"[VQ map re-rank: {vq_opts['score']}, a={vq_opts['alpha']:g}, "
                    f"K={vq_opts['quantize_k']}, s{vq_opts['stage']}]  "
                    f"R@1={rec_v[1]*100:.2f}%  R@5={rec_v[5]*100:.2f}%  "
                    f"R@10={rec_v[10]*100:.2f}%  mAP={ap_v*100:.2f}%")
                final[f"{direction}_vqrank"] = {
                    "r1": rec_v[1], "r5": rec_v[5], "r10": rec_v[10],
                    "ap": ap_v, "n_queries": n_q, "n_gallery": n_g}

        self.results_ready.emit(final, DATASET_DENSEUAV)

        if not self.no_history_input.isChecked() and self.history_input.text().strip():
            hp = Path(self.history_input.text().strip())
            if not hp.is_absolute(): hp = HERE / hp
            self._save_history(final, backbone, ckpt_path, self.label_input.text().strip(),
                               hp, DATASET_DENSEUAV)

        top_idx = np.argsort(-sim, axis=1)
        gal_arr = np.array(gallery_ids)

        # GPS already loaded above, reuse it for display logic
        query_gps = query_gps_full if 'query_gps_full' in locals() else {}
        gallery_gps = gallery_gps_full if 'gallery_gps_full' in locals() else {}

        sample_correct, sample_wrong = [], []
        failed_entries = []
        kml_rows = []

        def gps_distance(lat1, lon1, lat2, lon2):
            dlat_m = abs(lat2 - lat1) * 111_320.0
            dlon_m = abs(lon2 - lon1) * 111_320.0 * math.cos(math.radians(lat1))
            return math.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)

        for q_i in range(len(query_paths)):
            # Full confusion gallery — top-1 may be a distractor location,
            # consistent with the exact-ID protocol used for the metrics.
            top1_gal_idx = top_idx[q_i][0]
            best_gal = gallery_paths[top1_gal_idx]

            # Calculate GPS distance for display
            gps_dist_m = None
            exact_hit = bool(gal_arr[top1_gal_idx] == query_ids[q_i])
            correct = exact_hit
            if query_ids[q_i] in query_gps and gal_arr[top1_gal_idx] in gallery_gps:
                q_lat, q_lon = query_gps[query_ids[q_i]]
                g_lat, g_lon = gallery_gps[gal_arr[top1_gal_idx]]
                gps_dist_m = gps_distance(q_lat, q_lon, g_lat, g_lon)
                if not correct and gps_dist_m < 100.0:  # Accept locations within 100m as same
                    correct = True

            # Collect map rows (exact-ID correctness — matches the official metric)
            kml_rows.append((
                Path(query_paths[q_i]).name, query_ids[q_i],
                str(gal_arr[top1_gal_idx]),
                query_gps.get(query_ids[q_i]),
                gallery_gps.get(gal_arr[top1_gal_idx]),
                exact_hit, gps_dist_m))

            # Add scale and distance predictions
            scale_info = ""
            if qry_scales is not None and gal_scales is not None:
                q_scale = qry_scales[q_i][0] if len(qry_scales[q_i].shape) > 0 else qry_scales[q_i]
                g_scale = gal_scales[top1_gal_idx][0] if len(gal_scales[top1_gal_idx].shape) > 0 else gal_scales[top1_gal_idx]
                scale_info = f" | scale: pred={g_scale:.2f}, query={q_scale:.2f}"

            dist_info = f" | dist: {gps_dist_m:.0f}m" if gps_dist_m is not None else ""
            caption = f"id {query_ids[q_i]}{scale_info}{dist_info}"
            entry = (query_paths[q_i], best_gal, correct, caption)
            (sample_correct if correct else sample_wrong).append(entry)
            if not correct:
                gt_indices = np.where(gal_arr == query_ids[q_i])[0]
                gt_path = gallery_paths[gt_indices[0]] if len(gt_indices) > 0 else None
                pred_id = gal_arr[top1_gal_idx]
                failed_entries.append((query_paths[q_i], best_gal, gt_path, caption, pred_id, query_ids[q_i]))

        # KML map export: GT (blue) vs predicted top-1 (green=exact hit, red=miss)
        # with error lines connecting misses to their ground truth. Open in Google Earth.
        if query_gps and kml_rows:
            try:
                kml_path = HERE / "denseuav_eval_map.kml"
                n_gt, n_ok, n_bad = write_denseuav_kml(kml_path, query_gps, kml_rows)
                self.log_message.emit(
                    f"KML map written → {kml_path}  "
                    f"({n_gt} GT locations, {n_ok} correct / {n_bad} wrong predictions)")
            except Exception as exc:
                self.log_message.emit(f"KML export failed: {exc}")

        random.shuffle(sample_correct); random.shuffle(sample_wrong)
        half = MAX_SAMPLES_DISPLAY // 2
        samples = sample_wrong[:half] + sample_correct[:half]
        random.shuffle(samples)
        self.samples_ready.emit(samples[:MAX_SAMPLES_DISPLAY])
        random.shuffle(failed_entries)
        self.failed_ready.emit(failed_entries)
        self.progress_changed.emit(100)
        self.eval_finished.emit("complete")

    # ── Routed cluster-expert retrieval ────────────────────────────────────

    def _embed_routed_gallery(self, model, paths, device, batch_size, transform,
                              centroids, prog_range):
        """Each gallery tile embedded by its OWN cluster's expert.
        Returns (fine [N,Df] np, cluster_ids [N] np) or (None, None) if stopped."""
        loader = DataLoader(ImageDataset(paths, transform), batch_size=batch_size,
                            shuffle=False, **loader_kwargs(device.type))
        fine_chunks, cid_chunks = [], []
        n = max(1, len(loader)); model.eval()
        with torch.inference_mode():
            for bi, (imgs, _) in enumerate(loader):
                if self.stop_event.is_set():
                    return None, None
                fine, cids = model.embed_gallery(imgs.to(device, non_blocking=True), centroids)
                fine_chunks.append(fine.cpu().float().numpy())
                cid_chunks.append(cids.cpu().numpy())
                self.progress_changed.emit(int(prog_range[0] + (bi + 1) / n *
                                               (prog_range[1] - prog_range[0])))
        return np.concatenate(fine_chunks), np.concatenate(cid_chunks)

    def _embed_routed_query(self, model, paths, device, batch_size, transform, prog_range):
        """Query coarse (for routing) + fine under every expert.
        Returns (coarse [N,Dc] np, fine_all [N,K,Df] np) or (None, None)."""
        loader = DataLoader(ImageDataset(paths, transform), batch_size=batch_size,
                            shuffle=False, **loader_kwargs(device.type))
        coarse_chunks, fine_all_chunks = [], []
        n = max(1, len(loader)); model.eval()
        with torch.inference_mode():
            for bi, (imgs, _) in enumerate(loader):
                if self.stop_event.is_set():
                    return None, None
                coarse, fine_all = model.embed_query(imgs.to(device, non_blocking=True))
                coarse_chunks.append(coarse.cpu().float().numpy())
                fine_all_chunks.append(fine_all.cpu().float().numpy())
                self.progress_changed.emit(int(prog_range[0] + (bi + 1) / n *
                                               (prog_range[1] - prog_range[0])))
        return np.concatenate(coarse_chunks), np.concatenate(fine_all_chunks)

    def _denseuav_routed_sim(self, model, gallery_paths, gallery_ids, query_paths,
                             device, batch_size, transform):
        """Similarity matrix under coarse-routing: a query only competes against
        gallery tiles whose cluster is among the query's top-k coarse clusters, each
        scored by that cluster's expert. Gallery tiles outside the searched clusters
        stay at -inf, so a true match in an unsearched cluster is honestly a miss."""
        centroids = model._eval_centroids.to(device)          # [K, Dc]
        K = int(centroids.shape[0])
        topk = min(self.denseuav_routed_topk.value(), K)

        gal_fine, gal_cluster = self._embed_routed_gallery(
            model, gallery_paths, device, batch_size, transform, centroids, (5, 45))
        if gal_fine is None:
            return None
        qry_coarse, qry_fine_all = self._embed_routed_query(
            model, query_paths, device, batch_size, transform, (45, 88))
        if qry_coarse is None:
            return None

        # Top-k coarse clusters per query (cosine to centroids).
        cen = centroids.cpu().numpy()
        csim = qry_coarse @ cen.T                              # [Nq, K]
        topk_idx = np.argsort(-csim, axis=1)[:, :topk]
        topk_mask = np.zeros((qry_coarse.shape[0], K), dtype=bool)
        np.put_along_axis(topk_mask, topk_idx, True, axis=1)

        Nq, Ng = qry_fine_all.shape[0], gal_fine.shape[0]
        sim = np.full((Nq, Ng), -1e9, dtype=np.float32)
        for c in range(K):
            cols = np.where(gal_cluster == c)[0]
            if cols.size == 0:
                continue
            block = qry_fine_all[:, c, :] @ gal_fine[cols].T   # [Nq, n_c]
            block[~topk_mask[:, c], :] = -1e9                  # query not routed to c
            sim[:, cols] = block

        gal_counts = np.bincount(gal_cluster, minlength=K)
        searched_pct = float((topk_mask.astype(np.float32) @ gal_counts).mean() / Ng * 100)
        self.log_message.emit(
            f"Routed: K={K}, top-{topk} clusters/query, gallery/cluster "
            f"min/med/max={gal_counts.min()}/{int(np.median(gal_counts))}/{gal_counts.max()}, "
            f"avg gallery searched/query={searched_pct:.1f}%")
        return sim

    # ── Model loading ──────────────────────────────────────────────────────

    def _load_routed(self, ckpt: dict, device: torch.device):
        """Build a RoutedClusterModel from a routed checkpoint dict."""
        from routed_model import RoutedClusterModel
        cfg = ckpt.get("config", {})
        centroids = ckpt.get("centroids")
        if centroids is None:
            raise ValueError("Routed checkpoint has no centroids — run step 1 first.")
        model = RoutedClusterModel(
            num_clusters=cfg.get("num_clusters", 16), coarse_dim=cfg.get("coarse_dim", 512),
            fine_dim=cfg.get("fine_dim", 512), img_size=cfg.get("img_size", 384),
            pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model._eval_centroids = torch.as_tensor(np.asarray(centroids), dtype=torch.float32)
        model.to(device).eval()
        img_size = cfg.get("img_size", 384)
        self.log_message.emit(
            f"Routed model ready — K={model.num_clusters}, {img_size}×{img_size}, "
            f"centroids={model._eval_centroids.shape}")
        return model, "routed_swin_b", make_eval_transform(img_size=img_size)

    @staticmethod
    def _is_routed_ckpt(ckpt) -> bool:
        """Routed checkpoints carry per-cluster expert weights ('experts.*')."""
        state = ckpt.get("model_state_dict") if isinstance(ckpt, dict) else None
        return state is not None and any(k.startswith("experts.") for k in state)

    def _load_model(self, ckpt_path: Path, device: torch.device):
        self.log_message.emit(f"Loading checkpoint: {ckpt_path}")
        ui_bb = self.backbone_input.currentData()

        if ui_bb == "routed_swin_b":
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            return self._load_routed(ckpt, device)

        if ui_bb == "distill_swin_t":
            from distill_model import SwinStudent
            raw     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            student = SwinStudent(teacher_dim=1024, pretrained=False, backbone="swin_b")
            state   = raw.get("model_state_dict", raw.get("student", raw))
            student.load_state_dict(state, strict=True)
            student.to(device).eval()
            transform = make_eval_transform(img_size=224)
            self.log_message.emit("Model ready — Distilled Swin-B  224×224  →  1024-D CLS")
            return student, "distill_swin_t", transform

        ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if ui_bb == "auto" and self._is_routed_ckpt(ckpt):
            self.log_message.emit("Auto-detected routed cluster-expert checkpoint.")
            return self._load_routed(ckpt, device)
        state = ckpt.get("model_state_dict", ckpt)
        backbone = (ckpt.get("training_params", {}).get("backbone", "swin_t")
                    if ui_bb == "auto" else ui_bb)
        self.log_message.emit(f"Backbone: {backbone}")
        model = SwinEmbedding(backbone=backbone)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        img_size  = backbone_img_size(backbone)
        transform = make_eval_transform(img_size=img_size)
        self.log_message.emit(f"Model ready — input {img_size}×{img_size}")
        return model, backbone, transform

    # ── History persistence ────────────────────────────────────────────────

    def _save_history(self, results: dict, backbone: str, ckpt_path: Path,
                      label: str, history_path: Path, dataset_type: str):
        entry = {
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            "dataset":    dataset_type,
            "backbone":   backbone,
            "checkpoint": str(ckpt_path),
            "label":      label,
            "results":    {
                str(k): {kk: round(vv, 6) if isinstance(vv, float) else vv
                         for kk, vv in v.items()}
                for k, v in results.items()
            },
        }
        history = []
        if history_path.exists():
            try: history = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception: pass
        history.append(entry)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        self.log_message.emit(f"History saved → {history_path}")
        self._render_html(history_path, dataset_type)

    def _render_html(self, history_path: Path, dataset_type: str):
        try:
            runs = json.loads(history_path.read_text(encoding="utf-8"))
        except Exception:
            return
        runs = [r for r in runs if r.get("dataset", DATASET_SUES200) == dataset_type
                and isinstance(r.get("results"), dict)]
        if not runs:
            return

        if dataset_type == DATASET_G4L:
            pk_list = ["pos", "pos_semipos"]
            has_vq = any("pos_vq" in r.get("results", {}) for r in runs)
            if has_vq:
                pk_list += ["pos_vq", "pos_semipos_vq"]
            col_headers = []
            _lbl = {"pos": "pos", "pos_semipos": "semipos",
                   "pos_vq": "pos+VQ", "pos_semipos_vq": "semipos+VQ"}
            for pk in pk_list:
                lbl = _lbl[pk]
                col_headers += [f"{lbl} R@1", f"{lbl} R@5", f"{lbl} AP",
                                f"{lbl} SDM@3"]
            def _row_cells(r):
                res = r.get("results", {})
                cells = ""
                for pk in pk_list:
                    sub = res.get(pk, {})
                    for k, scale in (("r1", 100), ("r5", 100), ("ap", 100),
                                     ("sdm3", 1)):
                        v = sub.get(k)
                        fmt = (f"{v*scale:.2f}" if scale == 100
                               else f"{v:.4f}") if v is not None else ""
                        cells += f"<td>{fmt}</td>"
                return cells
        elif dataset_type == DATASET_SUES200:
            # vq_overall column block only when at least one run has it, so
            # plain coarse-only history doesn't grow blank columns.
            has_vq = any("vq_overall" in r.get("results", {}) for r in runs)
            result_keys = [str(a) for a in SUES_ALTITUDES] + ["overall"]
            if has_vq:
                result_keys.append("vq_overall")
            col_headers = []
            for ak in result_keys:
                lbl = ("Overall" if ak == "overall" else
                       "VQ rerank" if ak == "vq_overall" else f"{ak}m")
                for k in (1, 5, 10):
                    col_headers.append(f"{lbl} R@{k}")
            def _row_cells(r):
                res = r.get("results", {})
                cells = ""
                for ak in result_keys:
                    sub = res.get(ak, {})
                    for k in (1, 5, 10):
                        v = sub.get(f"r{k}")
                        cells += f"<td>{f'{v*100:.2f}' if v is not None else ''}</td>"
                return cells
        else:
            col_headers = ["Direction", "Queries", "Gallery", "R@1 (%)", "R@5 (%)",
                           "R@10 (%)", "mAP (%)", "SDM@1 (%)"]
            # Re-rank columns (DenseUAV only) — added dynamically, only for
            # variants that actually appear in at least one run, so plain
            # coarse-only history (and University-1652, which never sets these
            # keys) doesn't get blank columns bolted on.
            rerank_variants = []
            for d in ("d2s", "s2d"):
                dl = "D→S" if d == "d2s" else "S→D"
                rerank_variants += [(f"{d}_georank_filtered", f"{dl} georank"),
                                    (f"{d}_vqrank", f"{dl} VQ rerank")]
            present_georank = [key for key, _ in rerank_variants
                               if any(key in r.get("results", {}) for r in runs)]
            rerank_labels = dict(rerank_variants)
            for key in present_georank:
                lbl = rerank_labels[key]
                col_headers += [f"{lbl} R@1", f"{lbl} R@5", f"{lbl} R@10", f"{lbl} mAP"]

            def _row_cells(r):
                res = r.get("results", {})
                cells = ""
                for key in ("d2s", "s2d"):
                    sub = res.get(key)
                    if sub is None: continue
                    dlbl = "Drone→Sat" if key == "d2s" else "Sat→Drone"
                    sdm = sub.get("sdm1")
                    cells += (f"<td>{dlbl}</td><td>{sub.get('n_queries','')}</td>"
                              f"<td>{sub.get('n_gallery','')}</td>"
                              f"<td>{sub['r1']*100:.2f}</td><td>{sub['r5']*100:.2f}</td>"
                              f"<td>{sub['r10']*100:.2f}</td><td>{sub['ap']*100:.2f}</td>"
                              f"<td>{f'{sdm*100:.2f}' if sdm is not None else ''}</td>")
                for gkey in present_georank:
                    sub = res.get(gkey)
                    if sub is None:
                        cells += "<td></td>" * 4
                    else:
                        cells += (f"<td>{sub['r1']*100:.2f}</td><td>{sub['r5']*100:.2f}</td>"
                                  f"<td>{sub['r10']*100:.2f}</td><td>{sub['ap']*100:.2f}</td>")
                return cells

        rows_html = ""
        for i, r in enumerate(runs):
            cells = (f"<td>{i+1}</td><td>{r.get('timestamp','')}</td>"
                     f"<td>{r.get('backbone','')}</td><td>{r.get('label','')}</td>"
                     f"<td title='{r.get('checkpoint','')}' style='max-width:160px;"
                     f"overflow:hidden;text-overflow:ellipsis'>"
                     f"{Path(r.get('checkpoint','')).name}</td>")
            cells += _row_cells(r)
            rows_html += f"<tr>{cells}</tr>\n"

        html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
                f"<title>Eval History — {dataset_type}</title>"
                f"<style>body{{font-family:monospace;font-size:12px;padding:8px}}"
                f"table{{border-collapse:collapse;width:100%}}"
                f"th,td{{border:1px solid #ccc;padding:3px 7px;white-space:nowrap;text-align:center}}"
                f"th{{background:#e8e8e8;position:sticky;top:0}}"
                f"tr:nth-child(even){{background:#f8f8f8}}"
                f"td:nth-child(-n+5){{text-align:left}}</style></head><body>"
                f"<h2>Eval History — {dataset_type}</h2><table><tr>"
                f"<th>#</th><th>Timestamp</th><th>Backbone</th><th>Label</th><th>Checkpoint</th>"
                f"{''.join(f'<th>{h}</th>' for h in col_headers)}"
                f"</tr>{rows_html}</table></body></html>")
        html_path = history_path.with_suffix(".html")
        html_path.write_text(html, encoding="utf-8")
        self.log_message.emit(f"HTML written → {html_path}")

    # ── Settings persistence ───────────────────────────────────────────────

    def _load_settings(self):
        if not SETTINGS_PATH.exists():
            return
        try:
            s = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if "dataset" in s:
                idx = self.dataset_combo.findData(s["dataset"])
                if idx >= 0: self.dataset_combo.setCurrentIndex(idx)
            if "dataset_root" in s: self.root_input.setText(s["dataset_root"])
            if "checkpoint"   in s: self.checkpoint_input.setText(s["checkpoint"])
            if "backbone"     in s:
                idx = self.backbone_input.findData(s["backbone"])
                if idx >= 0: self.backbone_input.setCurrentIndex(idx)
            if "device" in s:
                idx = self.device_input.findData(s["device"])
                if idx >= 0: self.device_input.setCurrentIndex(idx)
            if "batch_size"   in s: self.batch_size_input.setValue(int(s["batch_size"]))
            if "altitude"     in s:
                idx = self.altitude_input.findData(s["altitude"])
                if idx >= 0: self.altitude_input.setCurrentIndex(idx)
            if "u1652_direction" in s:
                idx = self.u1652_direction.findData(s["u1652_direction"])
                if idx >= 0: self.u1652_direction.setCurrentIndex(idx)
            if "sues_gallery_root" in s: self.sues_gallery_root_input.setText(s["sues_gallery_root"])
            if "g4l_test_json" in s: self.g4l_json_input.setText(s["g4l_test_json"])
            if "label"      in s: self.label_input.setText(s["label"])
            if "history"    in s: self.history_input.setText(s["history"])
            if "no_history" in s: self.no_history_input.setChecked(bool(s["no_history"]))
        except Exception as e:
            self.log.append(f"[settings] load failed: {e}")

    def _save_settings(self):
        try:
            s = {
                "dataset":         self.dataset_combo.currentData(),
                "dataset_root":    self.root_input.text(),
                "checkpoint":      self.checkpoint_input.text(),
                "backbone":        self.backbone_input.currentData(),
                "device":          self.device_input.currentData(),
                "batch_size":      self.batch_size_input.value(),
                "altitude":        self.altitude_input.currentData(),
                "sues_gallery_root": self.sues_gallery_root_input.text().strip(),
                "g4l_test_json": self.g4l_json_input.text().strip(),
                "u1652_direction": self.u1652_direction.currentData(),
                "label":           self.label_input.text(),
                "history":         self.history_input.text(),
                "no_history":      self.no_history_input.isChecked(),
            }
            SETTINGS_PATH.write_text(json.dumps(s, indent=2), encoding="utf-8")
        except Exception as e:
            self.log.append(f"[settings] save failed: {e}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _make_hbox(*widgets) -> QWidget:
    w = QWidget(); h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    for ww in widgets: h.addWidget(ww)
    return w


def main():
    app = QApplication(sys.argv)
    win = GeneralEvalApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
