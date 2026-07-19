import io
import math
import os
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm


# ImageNet stats — used by Swin-T pretrained backbone and Game4Loc.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Drop-in normalisation transform (use at the end of every inference pipeline too).
IMAGENET_NORM = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


class _RandomCardinalRotation:
    """Rotate by a uniformly random multiple of 90° — identical to A.RandomRotate90(p=1.0)."""
    def __call__(self, img):
        k = random.randint(0, 3)
        return img if k == 0 else img.rotate(k * 90, expand=False)


class _JpegCompression:
    """Simulate JPEG compression artefacts (p=0.5) — mirrors A.ImageCompression."""
    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            quality = random.randint(90, 100)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            buf.seek(0)
            return Image.open(buf).convert("RGB")
        return img


class _BlurOrSharpen:
    """50/50 Gaussian blur or PIL sharpen with probability p — mirrors A.OneOf([AdvancedBlur, Sharpen], p=0.3)."""
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p:
            if random.random() < 0.5:
                radius = random.uniform(0.5, 2.5)
                return img.filter(ImageFilter.GaussianBlur(radius))
            return img.filter(ImageFilter.SHARPEN)
        return img


class _CoarseDropout:
    """Randomly zero-out 10–25 rectangular patches of size 10–20 % of the image.

    Applied to a float Tensor [C, H, W] after ToTensor, before Normalize.
    Approximates A.OneOf([A.GridDropout, A.CoarseDropout], p=0.3).
    """
    def __init__(self, p: float = 0.3, img_size: int = 224,
                 min_holes: int = 10, max_holes: int = 25,
                 min_frac: float = 0.1, max_frac: float = 0.2):
        self.p = p
        self.min_holes = min_holes
        self.max_holes = max_holes
        self.min_hw = max(1, int(min_frac * img_size))
        self.max_hw = max(1, int(max_frac * img_size))

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.p:
            return tensor
        _, H, W = tensor.shape
        result = tensor.clone()
        for _ in range(random.randint(self.min_holes, self.max_holes)):
            h = random.randint(self.min_hw, self.max_hw)
            w = random.randint(self.min_hw, self.max_hw)
            y = random.randint(0, max(0, H - h))
            x = random.randint(0, max(0, W - w))
            result[:, y:y + h, x:x + w] = 0.0
        return result


DEFAULT_AUG = dict(
    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05,
    jitter_p=0.5,
    dropout_p=1.0, min_holes=2, max_holes=3, min_frac=0.10, max_frac=0.20,
)


# ── Parallel-dataloading defaults ─────────────────────────────────────────────
# Single-process decoding (~15 img/s on the dual-Xeon box) starves the GPU, so
# workers are required. BUT steady-state measurement (2026-07-13, dual Xeon
# E5-2680 v4, 28 physical / 56 logical cores) shows throughput plateaus at
# ~200-290 img/s regardless of worker count from 12 up through 32 — verified
# with CPU at only 3-6% and PhysicalDisk Reads/sec = 0 (fully OS-cached) DURING
# an active run, and reproduced identically with Windows Defender real-time
# protection fully disabled. So the ceiling is neither CPU decode capacity nor
# disk I/O nor antivirus — it's almost certainly the DataLoader's single
# main-process consumer thread (deserializing + pinning each worker's tensor),
# a known Windows multiprocessing-IPC characteristic that doesn't scale with
# producer (worker) count. Raising num_workers past ~16 buys nothing but RAM
# and process overhead. (Earlier in-process back-to-back tests claiming
# 700-900+ img/s were a measurement artifact — too few batches, dominated by
# draining the prefetch_factor queue rather than true steady state; do not
# trust short single-shot benchmarks here, use a warmup + 100+ batch average.)
# This ceiling doesn't bottleneck TRAINING itself (dispatch-bound at ~110-160
# img/s on this CPU, comfortably under 200-290) — only the periodic Quick-Eval
# / full-gallery embedding passes (want up to ~480 img/s GPU-side) run somewhat
# decode-limited. Fixing that for real needs a smaller per-item IPC payload
# (e.g. transfer uint8 arrays and do ToTensor/Normalize on GPU), not more
# workers.
#
# CRITICAL (2026-07-13): each worker process is a separate Windows "spawn"
# process that re-imports torch. On this machine, EVERY such process was
# observed privately committing ~8.75-8.78 GB of virtual memory (Get-Process
# PagedMemorySize64) — almost certainly the CUDA driver's per-process virtual
# address reservation on Windows, even though decode workers are pure-CPU and
# never touch the GPU. With 14-16 workers that's 120-140 GB of system commit
# charge for nothing, and this machine's total commit LIMIT (RAM + pagefile)
# is ~162 GB — we measured 99.7% committed system-wide, which crashes
# DataLoader workers with "Couldn't open shared file mapping ... error 1455"
# (ERROR_COMMITMENT_LIMIT), a SYSTEM-WIDE virtual-memory exhaustion, not a
# per-process or GPU-memory issue. Since >8-12 workers buys zero decode
# throughput anyway (see above), the default is capped low specifically to
# bound this per-worker memory tax, not for CPU/throughput reasons.
#
# FIX (2026-07-13): _hide_gpu_from_worker (below) as worker_init_fn confirmed
# via Get-Process sampling to cut per-worker commit from ~8.78 GB to ~1.5-1.75
# GB. With the fix active, 8 workers measured 171 img/s steady-state (safe
# margin over training's ~110-160 img/s dispatch-bound consumption) at only
# ~12-14 GB total worker memory — the balance point used below. System commit
# also observed at a healthy 23% with this configuration (was 99.7% before the
# fix, at 14-16 unfixed workers). Override with the UAV_NUM_WORKERS env var
# (0 = old single-process behaviour).
def default_num_workers() -> int:
    """REVERTED 2026-07-16: briefly raised to min(16, cores*2//3) (=16 on the
    i9-12900) based on a decode-throughput benchmark showing gains through
    nw=16 — but that only checked the CPU-speed/throughput constraint, not
    the SEPARATE, RAM-capacity-bound constraint that originally justified
    capping at 8 (see the near-system-wide-crash note in hardware_rtx5090:
    each worker process privately commits ~1.5-1.75GB of virtual memory even
    with _hide_gpu_from_worker applied). RAM is still 64GB after the CPU/RAM
    upgrade (only DDR4->DDR5 generation changed, not capacity) — raising to
    16 immediately reproduced the exact same crash (error 1455,
    ERROR_COMMITMENT_LIMIT, system commit measured at 98.7%). These are two
    independent ceilings: throughput scales with CPU speed (improved), but
    the safe worker COUNT is capped by total system RAM (unchanged). Back to
    the proven-safe min(8, cores//4).
    """
    env = os.environ.get("UAV_NUM_WORKERS")
    if env is not None:
        try:
            return max(0, int(env))
        except ValueError:
            pass
    return min(8, max(0, (os.cpu_count() or 1) // 4))


def _hide_gpu_from_worker(_worker_id: int) -> None:
    """DataLoader worker_init_fn: decode workers are pure-CPU and never touch
    the GPU. Hiding CUDA devices inside the worker process is a best-effort
    attempt to stop torch/the NVIDIA driver from reserving its large
    per-process virtual-memory footprint on Windows (see default_num_workers
    docstring) — may be too late if torch already initialized CUDA during the
    worker's own import bootstrap, but costs nothing to set."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def loader_kwargs(device_type: str = "cuda", num_workers: int | None = None,
                  persistent: bool = False, pin_memory: bool | None = None) -> dict:
    """Keyword arguments for torch DataLoader with parallel decoding.

    persistent=False (default) respawns workers per epoch — REQUIRED when the
    dataset is mutated between epochs (hard-mining masks, .shuffle() reordering):
    Windows spawn workers pickle the dataset at iterator creation, so persistent
    workers would keep serving a stale copy and silently ignore the mutation.
    Use persistent=True only for one-shot or immutable-dataset loaders.

    pin_memory=False is REQUIRED for any loader created ad hoc from a background
    thread while the main training loop's own pinned-memory DataLoader may still
    be active in the same process (e.g. a periodic mid-training eval callback) —
    two concurrent pin-memory threads racing on CUDA's pinned-memory allocator
    triggers `cudaErrorAlreadyMapped` ("resource already mapped"). Default
    (None) follows device_type as before for ordinary single-loader use.
    """
    n = default_num_workers() if num_workers is None else num_workers
    pin = (device_type == "cuda") if pin_memory is None else pin_memory
    kw = dict(num_workers=n, pin_memory=pin)
    if n > 0:
        kw["persistent_workers"] = persistent
        # Keep the pinned-memory queue small: page-locked batches count against
        # Windows' shared-GPU-memory cap, and measured decode throughput
        # comfortably exceeds what training/eval actually consumes anyway.
        kw["prefetch_factor"] = 2
        # Best-effort: stop worker processes from reserving CUDA's per-process
        # virtual memory footprint on Windows (see default_num_workers).
        kw["worker_init_fn"] = _hide_gpu_from_worker
    return kw


def make_train_transform(img_size: int = 224, aug: dict | None = None,
                         augment: bool = True) -> T.Compose:
    """Game4Loc-equivalent training transform.

    Pass aug=dict(...) to override any subset of DEFAULT_AUG parameters.
    augment=False disables ALL augmentation (JPEG, jitter, blur/sharpen,
    cardinal rotation, dropout) and returns a clean Resize→ToTensor→Normalize
    pipeline identical to the eval transform.
    """
    if not augment:
        return make_eval_transform(img_size=img_size)
    a = {**DEFAULT_AUG, **(aug or {})}
    return T.Compose([
        _JpegCompression(),
        T.Resize((img_size, img_size)),
        T.RandomApply([T.ColorJitter(
            brightness=a["brightness"], contrast=a["contrast"],
            saturation=a["saturation"], hue=a["hue"],
        )], p=a["jitter_p"]),
        _BlurOrSharpen(p=0.3),
        _RandomCardinalRotation(),
        T.ToTensor(),
        _CoarseDropout(p=a["dropout_p"], img_size=img_size,
                       min_holes=a["min_holes"], max_holes=a["max_holes"],
                       min_frac=a["min_frac"], max_frac=a["max_frac"]),
        IMAGENET_NORM,
    ])


def make_eval_transform(img_size: int = 224) -> T.Compose:
    """Inference-only transform: resize shorter side → center crop to square → normalise.

    Using shorter-side resize + center crop instead of a hard squish ensures that
    non-square images (e.g. 3:2 drone photos) are presented to the model as a true
    square geographic region, consistent with the square scale-crop training data.
    Square images (satellite tiles, scale crops) are unaffected.
    """
    return T.Compose([
        T.Resize(img_size),       # shorter side → img_size, preserves aspect ratio
        T.CenterCrop(img_size),   # crop to img_size × img_size square
        T.ToTensor(),
        IMAGENET_NORM,
    ])


class SatCropDataset(Dataset):
    """Satellite crop dataset returning (q, p, idx) triples.

    group_size=1 (default): standard (q, p, idx) — one augmented view per pair.
    group_size=N>1: returns (views_q, views_p, idx) where views_q/views_p are
        [N, C, H, W] tensors with N independently augmented views of the same image.
        Used with GroupInfoNCE whole_slice training.
    """

    def __init__(self, dataset_root, group_size=1, img_size: int = 224,
                 augment: bool = True):
        self.root = Path(dataset_root)
        self.anchor_dir = self.root / "anchor"
        self.positive_dir = self.root / "positive"
        self.group_size = max(1, int(group_size))
        self.pairs = self._load_pairs()
        self.samples = list(self.pairs)   # active list (may be reordered by shuffle)

        self.transform = make_train_transform(img_size=img_size, augment=augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor_path, positive_path, original_idx = self.samples[idx]
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")

        if self.group_size == 1:
            q = self.transform(anchor)
            p = self.transform(positive)
            return q, p, original_idx

        # Group mode: N independent augmentations per image
        views_q = torch.stack([self.transform(anchor) for _ in range(self.group_size)])
        views_p = torch.stack([self.transform(positive) for _ in range(self.group_size)])
        return views_q, views_p, original_idx

    def shuffle(self, batch_size=64):
        """Game4Loc-style shuffle: ensures each batch window has unique anchor AND positive filenames.

        Prevents false negatives: if anchor A is in a batch, no other pair using A (or its
        positive) can appear in the same batch window.  Pairs left over from the last
        incomplete batch are discarded (they form noisy negatives otherwise).
        """
        pair_pool = list(self.pairs)
        random.shuffle(pair_pool)

        anchor_batch = set()
        positive_batch = set()
        seen_in_epoch = set()

        batches = []
        current_batch = []
        break_counter = 0
        MAX_BREAK = 16384

        pbar = tqdm(desc="Shuffle", unit="pair", leave=False)
        while pair_pool:
            pbar.update()
            entry = pair_pool.pop(0)
            a_name = entry[0].name
            p_name = entry[1].name
            key = (a_name, p_name)

            if a_name not in anchor_batch and p_name not in positive_batch and key not in seen_in_epoch:
                current_batch.append(entry)
                seen_in_epoch.add(key)
                anchor_batch.add(a_name)
                positive_batch.add(p_name)
                break_counter = 0
            else:
                if key not in seen_in_epoch:
                    pair_pool.append(entry)
                break_counter += 1
                if break_counter >= MAX_BREAK:
                    break

            if len(current_batch) >= batch_size:
                batches.extend(current_batch)
                current_batch = []
                anchor_batch = set()
                positive_batch = set()

        pbar.close()
        self.samples = batches
        dropped = len(self.pairs) - len(self.samples)
        if dropped > 0:
            print(f"Shuffle: {len(self.samples)}/{len(self.pairs)} pairs kept "
                  f"({dropped} dropped from last incomplete batch).")

    def _load_pairs(self):
        if not self.anchor_dir.is_dir():
            raise FileNotFoundError(f"Missing anchor directory: {self.anchor_dir}")
        if not self.positive_dir.is_dir():
            raise FileNotFoundError(f"Missing positive directory: {self.positive_dir}")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        anchors = {
            path.name: path
            for path in self.anchor_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts
        }
        positives = {
            path.name: path
            for path in self.positive_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts
        }

        names = sorted(anchors.keys() & positives.keys())
        if not names:
            raise ValueError(
                f"No matching image pairs found in {self.anchor_dir} and {self.positive_dir}"
            )

        return [(anchors[name], positives[name], i) for i, name in enumerate(names)]


# ---------------------------------------------------------------------------
# Game4Loc / UAV-VisLoc JSON-format dataset
# ---------------------------------------------------------------------------

class GtaUavDataset(Dataset):
    """Training dataset for GTA-UAV and UAV-VisLoc (Game4Loc JSON format).

    Returns 4-tuples: (q_drone, p_sat, iou_weight, original_idx).
    q_drone: drone image tensor (query/anchor).
    p_sat:   satellite tile tensor (positive).
    iou_weight: IoU overlap scalar in [0,1] — used by WeightedInfoNCE.
    original_idx: dataset-level index for cluster assignment lookup.

    mode='pos'          — use only strong positives (pair_pos_sate_img_list).
    mode='pos_semipos'  — include semi-positives (pair_pos_semipos_sate_img_list)
                          with their IoU weights.

    group_size=N>1: returns ([N,C,H,W], [N,C,H,W], weight, idx) for GroupInfoNCE.
    """

    def __init__(self, data_root, pairs_meta_file,
                 mode="pos_semipos", group_size=1, img_size: int = 224,
                 augment_positives: bool = False, augment: bool = True):
        import json
        self.data_root = Path(data_root)
        self.group_size = max(1, int(group_size))
        # Instance-level transforms so compute_embedding_cluster_stats can
        # temporarily swap in the eval transform on both sides.
        self.drone_transform = make_train_transform(img_size=img_size, augment=augment)
        self.transform = make_train_transform(img_size=img_size, augment=augment)  # satellite / positive side

        meta_path = self.data_root / pairs_meta_file
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.pairs = []   # (drone_path, sat_path, iou_weight, orig_idx)
        pos_key = f"pair_{mode}_sate_img_list"
        wt_key  = f"pair_{mode}_sate_weight_list"
        _gps_list = []   # parallel to self.pairs before orig_idx attachment

        for entry in meta:
            drone_path = self.data_root / entry["drone_img_dir"] / entry["drone_img_name"]
            sat_dir    = self.data_root / entry["sate_img_dir"]
            sat_imgs   = entry.get(pos_key, [])
            sat_wts    = entry.get(wt_key,  [1.0] * len(sat_imgs))
            # GPS: support both our custom format and VisLoc format
            lat = entry.get("drone_lat")
            lon = entry.get("drone_lon")
            if lat is None:
                loc = entry.get("drone_loc_lat_lon")
                if loc and len(loc) == 2:
                    lat, lon = loc
            gps = (float(lat), float(lon)) if lat is not None else None
            for sat_name, weight in zip(sat_imgs, sat_wts):
                self.pairs.append((drone_path, sat_dir / sat_name, float(weight)))
                _gps_list.append(gps)

        if not self.pairs:
            raise ValueError(
                f"No pairs found in {meta_path} with mode='{mode}'. "
                "Check that the JSON keys match the mode string."
            )

        # Attach original index for cluster lookup
        self.pairs = [(d, s, w, i) for i, (d, s, w) in enumerate(self.pairs)]
        self.samples = list(self.pairs)
        # GPS map: orig_idx → (lat, lon); used for proximity-based negative exclusion
        self.pair_gps = {i: gps for i, gps in enumerate(_gps_list) if gps is not None}

        if augment_positives:
            self._augment_with_tile_recall(meta)

        # Scale labels: {orig_idx → (drone_log_scale, sat_log_scale)} in log(metres).
        # Populated for VisLoc entries where altitude CSV is available; empty otherwise.
        self.scale_labels = self._compute_scale_labels(meta)

    # ------------------------------------------------------------------

    def _compute_scale_labels(self, meta: list) -> dict:
        """Build {orig_idx: (drone_log, sat_log)} in log(metres).

        drone_log = log(drone footprint width in metres) — from altitude CSV.
        sat_log   = log(tile geographic width in metres) — from tile name bounds.

        Returns an empty dict for non-VisLoc datasets or when CSV is unavailable.
        Missing altitude entries have drone_log = nan; tile bounds always computable.
        """
        import math
        try:
            from visloc_eval import _load_drone_csv_meta, _FOV_H, _tile2bounds, _R_EARTH
        except Exception:
            return {}

        visloc_entries = [e for e in meta if "drone_loc_lat_lon" in e]
        if not visloc_entries:
            return {}

        seqs = sorted(set(e["drone_img_name"].split("_")[0] for e in visloc_entries))
        try:
            drone_csv = _load_drone_csv_meta(self.data_root, seqs)
        except Exception:
            return {}
        if not drone_csv:
            return {}

        drone_alt = {name: m["height"] for name, m in drone_csv.items()}
        tan_half_fov = math.tan(math.radians(_FOV_H) / 2.0)

        def _sat_log(tile_name):
            try:
                lat_min, lon_min, lat_max, lon_max = _tile2bounds(tile_name)
                lat_c = (lat_min + lat_max) / 2.0
                w_m = (lon_max - lon_min) * math.cos(math.radians(lat_c)) * _R_EARTH * math.pi / 180.0
                return math.log(w_m) if w_m > 0 else float("nan")
            except Exception:
                return float("nan")

        labels: dict = {}
        for drone_path, sat_path, _, orig_idx in self.pairs:
            alt = drone_alt.get(drone_path.name)
            drone_log = math.log(2.0 * alt * tan_half_fov) if (alt is not None and alt > 0) else float("nan")
            sat_log = _sat_log(sat_path.name)
            labels[orig_idx] = (drone_log, sat_log)

        valid_d = sum(1 for d, _ in labels.values() if math.isfinite(d))
        valid_s = sum(1 for _, s in labels.values() if math.isfinite(s))
        print(f"  Scale labels: {valid_d}/{len(labels)} drone GT, {valid_s}/{len(labels)} tile GT.", flush=True)
        return labels

    # ------------------------------------------------------------------
    # Lower threshold than eval (0.80): training benefits from inclusive positives
    # to avoid false negatives in contrastive learning.  Tiles with ≥40% of their
    # area inside the drone footprint are clearly the same geographic area.
    _TILE_RECALL_TRAIN_THRESHOLD = 0.40

    def _augment_with_tile_recall(self, meta: list):
        """Add pairs whose tile_recall >= _TILE_RECALL_TRAIN_THRESHOLD (40%) but not
        already in the loaded JSON pairs.  Catches adjacent and partially-overlapping
        tiles that IoU >= 0.39 misses.  VisLoc-only (requires drone CSV metadata).
        """
        import math
        try:
            from visloc_eval import (
                _iou_footprint_tile,
                _load_drone_csv_meta, _tile2bounds,
                _FOV_H, _FOV_V,
            )
        except ImportError:
            print("  augment_positives: shapely/visloc_eval unavailable, skipping.", flush=True)
            return

        visloc_entries = [e for e in meta if "drone_loc_lat_lon" in e]
        if not visloc_entries:
            return

        seqs = sorted(set(e["drone_img_name"].split("_")[0] for e in visloc_entries))
        drone_csv = _load_drone_csv_meta(self.data_root, seqs)
        if not drone_csv:
            print("  augment_positives: no CSV metadata found, skipping.", flush=True)
            return

        sate_img_dir = next(
            (e.get("sate_img_dir") for e in visloc_entries if e.get("sate_img_dir")), None
        )
        if sate_img_dir is None:
            return
        sat_dir = self.data_root / sate_img_dir
        if not sat_dir.is_dir():
            print(f"  augment_positives: satellite dir not found: {sat_dir}", flush=True)
            return

        # All satellite tiles grouped by sequence
        seq_set = set(seqs)
        seq_tile_info: dict = {}
        for p in sat_dir.iterdir():
            if p.suffix.lower() != ".png":
                continue
            seq = p.name.split("_")[0]
            if seq not in seq_set:
                continue
            try:
                bounds = _tile2bounds(p.name)
            except Exception:
                continue
            seq_tile_info.setdefault(seq, []).append((p.name, bounds))

        # Existing (drone_name, sat_name) pairs to avoid duplicates
        existing = {(d.name, s.name) for d, s, *_ in self.pairs}

        new_pairs = []
        for entry in visloc_entries:
            dname = entry["drone_img_name"]
            m = drone_csv.get(dname)
            if m is None:
                continue
            dlat, dlon = entry["drone_loc_lat_lon"]
            alt = m["height"]
            hdg = m["heading"]
            qseq = dname.split("_")[0]
            sat_dir_e = self.data_root / entry["sate_img_dir"]
            drone_path = self.data_root / entry["drone_img_dir"] / dname

            max_r = math.sqrt(
                (alt * math.tan(math.radians(_FOV_H / 2))) ** 2 +
                (alt * math.tan(math.radians(_FOV_V / 2))) ** 2
            ) * 1.5

            for t_name, bounds in seq_tile_info.get(qseq, []):
                if (dname, t_name) in existing:
                    continue
                glat = (bounds[0] + bounds[2]) / 2.0
                glon = (bounds[1] + bounds[3]) / 2.0
                if abs(glat - dlat) * 111_320.0 > max_r:
                    continue
                if abs(glon - dlon) * 111_320.0 * max(math.cos(math.radians(dlat)), 1e-9) > max_r:
                    continue
                _, tile_recall = _iou_footprint_tile(t_name, dlat, dlon, alt, hdg)
                if tile_recall >= self._TILE_RECALL_TRAIN_THRESHOLD:
                    new_pairs.append((drone_path, sat_dir_e / t_name, float(tile_recall)))
                    existing.add((dname, t_name))

        if new_pairs:
            base = len(self.pairs)
            self.pairs += [(d, s, w, base + i) for i, (d, s, w) in enumerate(new_pairs)]
            self.samples = list(self.pairs)
            print(f"  augment_positives: +{len(new_pairs)} tile-recall pairs "
                  f"(recall ≥ {self._TILE_RECALL_TRAIN_THRESHOLD:.0%}). "
                  f"Total: {len(self.pairs)} pairs.", flush=True)
        else:
            print("  augment_positives: no new tile-recall pairs found.", flush=True)

    # ------------------------------------------------------------------
    def apply_hard_mining_mask(self, easiness, mask_step=0.10, mask_reset=0.70,
                               log_fn=print):
        """Curriculum hard-mining (called once per epoch).

        easiness: {orig_idx -> margin (pos_sim - hardest_neg_sim)}; higher = easier.
        Masks the easiest `mask_step` fraction of currently-active pairs out of the
        next epoch. Once the cumulative masked fraction exceeds `mask_reset`,
        unmasks everything (so the model revisits easy cases and doesn't forget).
        orig_idx == the pair's position in the full list, so it stays stable across
        masking; self.samples is just subset (true indices preserved).
        """
        if not hasattr(self, "_hm_full"):
            self._hm_full = list(self.pairs)     # immutable full set
            self._hm_masked = set()
        N = len(self._hm_full)
        active = [(p[-1], easiness.get(p[-1], float("-inf")))
                  for p in self._hm_full if p[-1] not in self._hm_masked]
        active.sort(key=lambda x: -x[1])         # easiest (highest margin) first
        k = max(1, int(N * mask_step))
        newly = [oi for oi, _ in active[:k]]
        self._hm_masked |= set(newly)
        frac = len(self._hm_masked) / max(1, N)
        if frac > mask_reset:
            self._hm_masked = set()
            self.samples = list(self._hm_full)
            # Tracked so the trainer can tag the NEXT epoch's clustering-stat log
            # as a "reset checkpoint": same-cluster rate measured on the full
            # (100%) population, comparable across mask cycles to see whether
            # the hard-mining curriculum is genuinely improving over time.
            self._hm_last_frac = 0.0
            self._hm_just_reset = True
            log_fn(f"Hard-mining: masked {frac:.0%} > {mask_reset:.0%} → "
                   f"RESET to 100% active ({N} pairs).")
            return
        self.samples = [p for p in self._hm_full if p[-1] not in self._hm_masked]
        self._hm_last_frac = frac
        self._hm_just_reset = False
        log_fn(f"Hard-mining: +{len(newly)} easiest masked → "
               f"{len(self.samples)}/{N} active ({frac:.0%} masked).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        drone_path, sat_path, iou_weight, original_idx = self.samples[idx]
        drone = Image.open(drone_path).convert("RGB")
        sat   = Image.open(sat_path).convert("RGB")

        weight_tensor = torch.tensor(iou_weight, dtype=torch.float32)

        if self.group_size == 1:
            q = self.drone_transform(drone)
            p = self.transform(sat)
            return q, p, weight_tensor, original_idx

        views_q = torch.stack([self.drone_transform(drone) for _ in range(self.group_size)])
        views_p = torch.stack([self.transform(sat)         for _ in range(self.group_size)])
        return views_q, views_p, weight_tensor, original_idx

    # ------------------------------------------------------------------
    def shuffle(self, batch_size=64):
        """Game4Loc-style shuffle: unique drone AND satellite per batch window."""
        pair_pool = list(self.pairs)
        random.shuffle(pair_pool)

        drone_batch = set()
        sat_batch   = set()
        seen        = set()
        batches     = []
        current     = []
        breaks      = 0
        MAX_BREAK   = 16384

        pbar = tqdm(desc="GTA shuffle", unit="pair", leave=False)
        while pair_pool:
            pbar.update()
            entry = pair_pool.pop(0)
            d_name = entry[0].name
            s_name = entry[1].name
            key = (d_name, s_name)

            if d_name not in drone_batch and s_name not in sat_batch and key not in seen:
                current.append(entry)
                seen.add(key)
                drone_batch.add(d_name)
                sat_batch.add(s_name)
                breaks = 0
            else:
                if key not in seen:
                    pair_pool.append(entry)
                breaks += 1
                if breaks >= MAX_BREAK:
                    break

            if len(current) >= batch_size:
                batches.extend(current)
                current = []
                drone_batch = set()
                sat_batch   = set()

        pbar.close()
        self.samples = batches
        dropped = len(self.pairs) - len(self.samples)
        if dropped:
            print(f"GTA shuffle: {len(self.samples)}/{len(self.pairs)} pairs "
                  f"kept ({dropped} dropped from incomplete last batch).")


# ---------------------------------------------------------------------------
# University-1652  drone → satellite  training dataset
# ---------------------------------------------------------------------------

class University1652Dataset(Dataset):
    """Training dataset for University-1652 (drone → satellite, D→S only).

    Train directory layout:
      <train_root>/
        drone/<location_id>/image-01.jpeg ...
        satellite/<location_id>/<location_id>.jpg

    Returns 4-tuples: (q_drone, p_sat, iou_weight=1.0, original_idx).
    Interface is identical to GtaUavDataset so it drops straight into trainer.py.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

    def __init__(self, train_root, group_size: int = 1,
                 img_size: int = 224, augment: bool = False,
                 queries_per_location: int = 10):
        self.train_root           = Path(train_root)
        self.group_size           = max(1, int(group_size))
        self.queries_per_location = queries_per_location
        self.drone_transform = make_train_transform(img_size=img_size, augment=augment)
        self.transform       = make_train_transform(img_size=img_size, augment=augment)

        drone_root = self.train_root / "drone"
        sat_root   = self.train_root / "satellite"
        if not drone_root.is_dir():
            raise FileNotFoundError(f"University-1652 drone directory not found: {drone_root}")
        if not sat_root.is_dir():
            raise FileNotFoundError(f"University-1652 satellite directory not found: {sat_root}")

        pairs = []
        for loc_dir in sorted(d for d in drone_root.iterdir() if d.is_dir()):
            loc_id = loc_dir.name
            sat_dir = sat_root / loc_id
            if not sat_dir.is_dir():
                continue
            sat_imgs = sorted(p for p in sat_dir.iterdir()
                              if p.suffix.lower() in self.IMAGE_EXTS)
            if not sat_imgs:
                continue
            sat_path = sat_imgs[0]
            drone_imgs = sorted(p for p in loc_dir.iterdir()
                                if p.suffix.lower() in self.IMAGE_EXTS)
            if queries_per_location and len(drone_imgs) > queries_per_location:
                drone_imgs = random.sample(drone_imgs, queries_per_location)
            for dp in drone_imgs:
                pairs.append((dp, sat_path))

        if not pairs:
            raise ValueError(f"No University-1652 pairs found under {train_root}")

        self.pairs   = [(d, s, 1.0, i) for i, (d, s) in enumerate(pairs)]
        self.samples = list(self.pairs)
        self.scale_labels: dict = {}
        self.pair_gps:     dict = {}

        n_locs = len({p[0].parent.name for p in self.pairs})
        print(f"University-1652 D→S: {len(self.pairs)} pairs from {n_locs} locations "
              f"({queries_per_location} queries/loc, augment={augment}).", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        drone_path, sat_path, iou_weight, original_idx = self.samples[idx]
        drone = Image.open(drone_path).convert("RGB")
        sat   = Image.open(sat_path).convert("RGB")
        weight_tensor = torch.tensor(iou_weight, dtype=torch.float32)
        if self.group_size == 1:
            q = self.drone_transform(drone)
            p = self.transform(sat)
            return q, p, weight_tensor, original_idx
        views_q = torch.stack([self.drone_transform(drone) for _ in range(self.group_size)])
        views_p = torch.stack([self.transform(sat)         for _ in range(self.group_size)])
        return views_q, views_p, weight_tensor, original_idx

    def shuffle(self, batch_size: int = 64):
        """Simple shuffle — no same-location-per-batch guarantee needed for U1652."""
        lst = list(self.pairs)
        random.shuffle(lst)
        self.samples = lst

    def apply_hard_mining_mask(self, easiness, mask_step=0.10, mask_reset=0.70,
                                log_fn=print):
        if not hasattr(self, "_hm_full"):
            self._hm_full   = list(self.pairs)
            self._hm_masked = set()
        N      = len(self._hm_full)
        active = [(p[-1], easiness.get(p[-1], float("-inf")))
                  for p in self._hm_full if p[-1] not in self._hm_masked]
        active.sort(key=lambda x: -x[1])
        k      = max(1, int(N * mask_step))
        newly  = [oi for oi, _ in active[:k]]
        self._hm_masked |= set(newly)
        frac   = len(self._hm_masked) / max(1, N)
        if frac > mask_reset:
            self._hm_masked = set()
            self.samples    = list(self._hm_full)
            self._hm_last_frac = 0.0
            self._hm_just_reset = True
            log_fn(f"Hard-mining: masked {frac:.0%} > {mask_reset:.0%} → "
                   f"RESET to 100% active ({N} pairs).")
            return
        self.samples = [p for p in self._hm_full if p[-1] not in self._hm_masked]
        self._hm_last_frac = frac
        self._hm_just_reset = False
        log_fn(f"Hard-mining: +{len(newly)} easiest masked → "
               f"{len(self.samples)}/{N} active ({frac:.0%} masked).")


class Sues200Dataset(University1652Dataset):
    """Training dataset for SUES-200 (drone → satellite, D→S only).

    Train directory layout (one extra altitude level vs University-1652):
      <train_root>/
        drone_view_512/<location_id>/<altitude>/0.jpg ... 49.jpg
        satellite-view/<location_id>/0.png

    IMPORTANT: point train_root at a TRAIN SPLIT (e.g. SUES-200-split/train,
    120 locations), never the raw unsplit 200-location dataset — evaluating on
    locations the model trained against makes the eval meaningless.

    Interface identical to University1652Dataset (4-tuples, shuffle,
    hard-mining) — only the pair scan differs, so everything else is
    inherited. queries_per_location subsamples PER ALTITUDE (0 = all 50).
    """

    ALTITUDES = ["150", "200", "250", "300"]

    def __init__(self, train_root, group_size: int = 1,
                 img_size: int = 224, augment: bool = False,
                 queries_per_location: int = 0):
        self.train_root           = Path(train_root)
        self.group_size           = max(1, int(group_size))
        self.queries_per_location = queries_per_location
        self.drone_transform = make_train_transform(img_size=img_size, augment=augment)
        self.transform       = make_train_transform(img_size=img_size, augment=augment)

        drone_root = self.train_root / "drone_view_512"
        sat_root   = self.train_root / "satellite-view"
        if not drone_root.is_dir():
            raise FileNotFoundError(f"SUES-200 drone directory not found: {drone_root}")
        if not sat_root.is_dir():
            raise FileNotFoundError(f"SUES-200 satellite directory not found: {sat_root}")

        pairs = []
        for loc_dir in sorted(d for d in drone_root.iterdir() if d.is_dir()):
            sat_dir = sat_root / loc_dir.name
            if not sat_dir.is_dir():
                continue
            sat_imgs = sorted(p for p in sat_dir.iterdir()
                              if p.suffix.lower() in self.IMAGE_EXTS)
            if not sat_imgs:
                continue
            sat_path = sat_imgs[0]
            for alt in self.ALTITUDES:
                alt_dir = loc_dir / alt
                if not alt_dir.is_dir():
                    continue
                drone_imgs = sorted(p for p in alt_dir.iterdir()
                                    if p.suffix.lower() in self.IMAGE_EXTS)
                if queries_per_location and len(drone_imgs) > queries_per_location:
                    drone_imgs = random.sample(drone_imgs, queries_per_location)
                for dp in drone_imgs:
                    pairs.append((dp, sat_path))

        if not pairs:
            raise ValueError(f"No SUES-200 pairs found under {train_root}")

        self.pairs   = [(d, s, 1.0, i) for i, (d, s) in enumerate(pairs)]
        self.samples = list(self.pairs)
        self.scale_labels: dict = {}
        self.pair_gps:     dict = {}

        n_locs = len({p[0].parent.parent.name for p in self.pairs})
        print(f"SUES-200 D→S: {len(self.pairs)} pairs from {n_locs} locations "
              f"(altitudes={self.ALTITUDES}, "
              f"{queries_per_location or 'all'} queries/loc/alt, "
              f"augment={augment}).", flush=True)


class DenseUAVDataset(Dataset):
    """Training dataset for DenseUAV (drone → satellite, D→S, multi-scale).

    Train directory layout:
      <train_root>/
        drone/<location_id>/H{80,90,100}.JPG
        satellite/<location_id>/H{80,90,100}.tif

    Returns 4-tuples: (q_drone, p_sat, pos_weight, original_idx).

    Multi-positive weighting: when cross_altitude=True, each drone image is paired
    with the satellite tile at *every* altitude of the same location (all are the
    same geographic place, hence all valid positives).

    altitude_weight_tau controls how strongly cross-altitude pairs are softened:
      - a float (e.g. 20.0): w = exp(-|Δalt_m| / tau) in (0, 1]. Same altitude →
        w=1.0 (hard positive); larger scale gap → smaller w (softer positive via
        WeightedInfoNCE label smoothing). With H80/H90/H100 and tau=20:
        Δ0→1.00, Δ10→0.61, Δ20→0.37.
      - None: DISABLE weighting — every combo gets w=1.0 (full-strength positive),
        matching the official DenseUAV baseline (Dataloader_University.py), which
        treats every drone/satellite altitude combination as the same class with
        no distance-based attenuation (its CE/triplet/KL losses never see a
        softened target). Use this to test whether our altitude-based softening
        helps or hurts vs. the baseline's uniform full-strength approach.

    Set cross_altitude=False for same-altitude-only (no multi-positive at all).
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif"}

    def __init__(self, train_root, group_size: int = 1,
                 img_size: int = 224, augment: bool = False,
                 altitudes: list = None, cross_altitude: bool = True,
                 altitude_weight_tau: float | None = 20.0):
        self.train_root = Path(train_root)
        self.group_size = max(1, int(group_size))
        self.altitudes = altitudes or ["H80", "H90", "H100"]
        self.cross_altitude = cross_altitude
        self.altitude_weight_tau = (None if altitude_weight_tau is None
                                    else float(altitude_weight_tau))
        self.drone_transform = make_train_transform(img_size=img_size, augment=augment)
        self.transform = make_train_transform(img_size=img_size, augment=augment)

        drone_root = self.train_root / "drone"
        sat_root = self.train_root / "satellite"
        if not drone_root.is_dir():
            raise FileNotFoundError(f"DenseUAV drone directory not found: {drone_root}")
        if not sat_root.is_dir():
            raise FileNotFoundError(f"DenseUAV satellite directory not found: {sat_root}")

        def _alt_m(a: str) -> float:
            """Parse altitude string like 'H80' → 80.0 metres."""
            return float(a.lstrip("Hh"))

        pairs = []   # (drone_img, sat_img, weight)
        for loc_dir in sorted(d for d in drone_root.iterdir() if d.is_dir()):
            loc_id = loc_dir.name
            sat_dir = sat_root / loc_id
            if not sat_dir.is_dir():
                continue

            for d_alt in self.altitudes:
                drone_img = loc_dir / f"{d_alt}.JPG"
                if not drone_img.is_file():
                    continue
                # Multi-positive: same-location satellite tiles at all altitudes are
                # valid positives; weight each by altitude match. cross_altitude=False
                # keeps only the same-altitude satellite (original behaviour).
                sat_alts = self.altitudes if self.cross_altitude else [d_alt]
                for s_alt in sat_alts:
                    sat_img = sat_dir / f"{s_alt}.tif"
                    if not sat_img.is_file():
                        continue
                    if self.altitude_weight_tau is None:
                        weight = 1.0   # full-strength for every combo (baseline-style)
                    else:
                        weight = math.exp(
                            -abs(_alt_m(d_alt) - _alt_m(s_alt)) / self.altitude_weight_tau)
                    pairs.append((drone_img, sat_img, weight))

        if not pairs:
            raise ValueError(f"No DenseUAV pairs found under {train_root}")

        self.pairs = [(d, s, w, i) for i, (d, s, w) in enumerate(pairs)]
        self.samples = list(self.pairs)
        self.scale_labels: dict = {}
        self.pair_gps: dict = {}

        # Load GPS coordinates from Dense_GPS_train.txt for hard negative exclusion
        gps_file = self.train_root.parent / "Dense_GPS_train.txt"
        print(f"DEBUG: Trying GPS file at: {gps_file} (exists: {gps_file.exists()})", flush=True)

        if gps_file:
            print(f"DEBUG: Loading GPS from {gps_file}...", flush=True)
            try:
                loc_gps = {}
                with open(gps_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            path = parts[0]
                            # Format: path lon(E/W) lat(N/S) height
                            lon_str = parts[1].lstrip('EW')
                            lat_str = parts[2].lstrip('NS')
                            try:
                                lat = float(lat_str)
                                lon = float(lon_str)
                                if parts[2].startswith('S'):
                                    lat = -lat
                                if parts[1].startswith('W'):
                                    lon = -lon

                                loc_id = Path(path).parent.name
                                if loc_id not in loc_gps:
                                    loc_gps[loc_id] = (lat, lon)
                            except ValueError:
                                pass

                # Map each pair index to its location's GPS
                for i, (drone_p, sat_p, _, _) in enumerate(self.pairs):
                    loc_id = Path(drone_p).parent.name
                    if loc_id in loc_gps:
                        self.pair_gps[i] = loc_gps[loc_id]
                print(f"DenseUAV GPS: Loaded {len(self.pair_gps)} pairs from {len(loc_gps)} locations", flush=True)
            except Exception as e:
                print(f"Warning: GPS loading failed: {e}", flush=True)
                import traceback
                traceback.print_exc()
        if not gps_file:
            print(f"Warning: GPS file not found in any of the expected locations", flush=True)

        n_locs = len({Path(p[0]).parent.name for p in self.pairs})
        if self.cross_altitude and self.altitude_weight_tau is None:
            print(f"DenseUAV D→S: {len(self.pairs)} pairs from {n_locs} locations "
                  f"(cross-altitude, altitude weighting DISABLED — all combos "
                  f"w=1.0 full-strength, baseline-style; altitudes={self.altitudes}, "
                  f"augment={augment}).", flush=True)
        elif self.cross_altitude:
            n_full = sum(1 for p in self.pairs if p[2] >= 0.999)
            print(f"DenseUAV D→S: {len(self.pairs)} weighted pairs from {n_locs} locations "
                  f"({n_full} same-altitude w=1.0, {len(self.pairs)-n_full} cross-altitude "
                  f"w<1; tau={self.altitude_weight_tau:g}, altitudes={self.altitudes}, "
                  f"augment={augment}).", flush=True)
        else:
            print(f"DenseUAV D→S: {len(self.pairs)} pairs from {n_locs} locations "
                  f"(same-altitude only, altitudes={self.altitudes}, augment={augment}).",
                  flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        drone_path, sat_path, iou_weight, original_idx = self.samples[idx]
        drone = Image.open(drone_path).convert("RGB")
        sat = Image.open(sat_path).convert("RGB")
        weight_tensor = torch.tensor(iou_weight, dtype=torch.float32)
        if self.group_size == 1:
            q = self.drone_transform(drone)
            p = self.transform(sat)
            return q, p, weight_tensor, original_idx
        views_q = torch.stack([self.drone_transform(drone) for _ in range(self.group_size)])
        views_p = torch.stack([self.transform(sat) for _ in range(self.group_size)])
        return views_q, views_p, weight_tensor, original_idx

    def shuffle(self, batch_size: int = 64):
        """Simple shuffle."""
        lst = list(self.pairs)
        random.shuffle(lst)
        self.samples = lst

    def apply_hard_mining_mask(self, easiness, mask_step=0.10, mask_reset=0.70,
                                log_fn=print):
        if not hasattr(self, "_hm_full"):
            self._hm_full = list(self.pairs)
            self._hm_masked = set()
        N = len(self._hm_full)
        active = [(p[-1], easiness.get(p[-1], float("-inf")))
                  for p in self._hm_full if p[-1] not in self._hm_masked]
        active.sort(key=lambda x: -x[1])
        k = max(1, int(N * mask_step))
        newly = [oi for oi, _ in active[:k]]
        self._hm_masked |= set(newly)
        frac = len(self._hm_masked) / max(1, N)
        if frac > mask_reset:
            self._hm_masked = set()
            self.samples = list(self._hm_full)
            # Tracked so the trainer can tag the NEXT epoch's clustering-stat log
            # as a "reset checkpoint": same-cluster rate measured on the full
            # (100%) population, comparable across mask cycles to see whether
            # the hard-mining curriculum is genuinely improving over time.
            self._hm_last_frac = 0.0
            self._hm_just_reset = True
            log_fn(f"Hard-mining: masked {frac:.0%} > {mask_reset:.0%} → "
                   f"RESET to 100% active ({N} pairs).")
            return
        self.samples = [p for p in self._hm_full if p[-1] not in self._hm_masked]
        self._hm_last_frac = frac
        self._hm_just_reset = False
        log_fn(f"Hard-mining: +{len(newly)} easiest masked → "
               f"{len(self.samples)}/{N} active ({frac:.0%} masked).")
