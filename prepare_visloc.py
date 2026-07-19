#!/usr/bin/env python3
"""
prepare_visloc.py — Convert raw UAV-VisLoc dataset to Game4Loc JSON format.

Output layout inside data_root:
  satellite/          flat dir of 256×256 PNG tiles (two zoom levels per seq)
  drone/images/       flat dir of all drone images
  same-area-drone2sate-train.json
  same-area-drone2sate-test.json
  cross-area-drone2sate-train.json
  cross-area-drone2sate-test.json

Run once before training; re-running is safe — existing tiles and drone
images are skipped, JSON files are always regenerated.

Usage:
  python prepare_visloc.py --data-root D:/UAV_DATASET/VisLoc
  python prepare_visloc.py --data-root D:/UAV_DATASET/VisLoc --split same-area
  python prepare_visloc.py --data-root D:/UAV_DATASET/VisLoc --no-tiling
  python prepare_visloc.py --data-root D:/UAV_DATASET/VisLoc --seqs 3 4 8 11

Default splits (Game4Loc paper):
  same-area : sequences 03 + 04 only, 80 pct train / 20 pct test
  cross-area: train = seqs 01 + 03, test = seqs 02 + 04
              (full set: --train-seqs 1 2 3 4 5 8 11 --test-seqs 1 2 3 4 5 8 11)

Sequence notes:
  07 is only 170 px tall (a sliver) — skip it.
  09 is not in Game4Loc's coordinate table — included here but may give poor results.
  06, 10, 11 are optional extras Game4Loc sometimes uses.
"""

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

TILE_SIZE = 256
FOV_H     = 36    # horizontal camera FOV (degrees)
FOV_V     = 52    # vertical camera FOV (degrees)
IOU_POS   = 0.39  # IoU threshold for positive pair
IOU_SEMI  = 0.14  # IoU threshold for semi-positive pair

SATE_LATLON = {
    "01": (29.774065, 115.970635, 29.702283, 115.996851),
    "02": (29.817376, 116.033769, 29.725402, 116.064566),
    "03": (32.355491, 119.805926, 32.29029,  119.900052),
    "04": (32.254036, 119.90598,  32.151018, 119.954509),
    "05": (24.666899, 102.340055, 24.650422, 102.365252),
    "06": (32.373177, 109.63516,  32.346944, 109.656837),
    "07": (40.340058, 115.791182, 40.339604, 115.79923 ),
    "08": (30.947227, 120.136489, 30.903521, 120.252951),
    "09": (30.999512, 120.030076, 30.910733, 120.149648),
    "10": (40.355093, 115.776356, 40.341475, 115.794041),
    "11": (38.852301, 101.013109, 38.807825, 101.092483),
}


# ---------------------------------------------------------------------------
# Coordinate helpers (all self-contained — no global state needed in workers)
# ---------------------------------------------------------------------------

def _geo_to_px(lat, lon, lt_lat, lt_lon, rb_lat, rb_lon, H, W):
    """Matches Game4Loc geo_to_image_coords exactly (y_range is negative)."""
    R = 6_378_137.0
    x_range = R * (rb_lon - lt_lon) * math.cos(math.radians((lt_lat + rb_lat) / 2))
    y_range = R * (rb_lat - lt_lat)   # negative: rb_lat < lt_lat
    x = R * (lon - lt_lon) * math.cos(math.radians((lat + lt_lat) / 2)) / x_range * W
    y = R * (lat - lt_lat) / y_range * H   # both num & denom negative -> positive y
    return int(x), int(y)


def _offset_to_latlon(lat, lon, dx, dy):
    R = 6_378_137.0
    dlat = dy / R * 180 / math.pi
    dlon = dx / (R * math.cos(math.radians(lat))) * 180 / math.pi
    return lat + dlat, lon + dlon


def _coverage_corners(lat, lon, height, heading_deg):
    fh  = math.radians(FOV_H)
    fv  = math.radians(FOV_V)
    hh  = height * math.tan(fh / 2)
    hv  = height * math.tan(fv / 2)
    ah  = math.radians((90 - heading_deg) % 360)
    ca, sa = math.cos(ah), math.sin(ah)
    offsets = [
        (-hh * ca - hv * sa, -hh * sa + hv * ca),
        ( hh * ca - hv * sa,  hh * sa + hv * ca),
        (-hh * ca + hv * sa, -hh * sa - hv * ca),
        ( hh * ca + hv * sa,  hh * sa - hv * ca),
    ]
    return [_offset_to_latlon(lat, lon, dx, dy) for dx, dy in offsets]


def _tile_center_latlon(seq, zoom, tx, ty, sate_h, sate_w, max_zoom):
    """Compute tile center lat/lon — self-contained, safe for subprocess."""
    lt_lat, lt_lon, rb_lat, rb_lon = SATE_LATLON[seq]
    scale = 2 ** (max_zoom - zoom)
    sw    = math.ceil(sate_w / scale)
    sh    = math.ceil(sate_h / scale)
    coe_lon = (tx + 0.5) * TILE_SIZE / sw
    coe_lat = (ty + 0.5) * TILE_SIZE / sh
    lat = lt_lat - coe_lat * (lt_lat - rb_lat)
    lon = lt_lon + coe_lon * (rb_lon - lt_lon)
    return lat, lon


def _ordered_polygon(pts):
    hull = ConvexHull(pts)
    return Polygon([pts[i] for i in hull.vertices])


def _find_overlapping_tiles(seq, zoom, px_corners, sate_h, sate_w, max_zoom):
    scale   = 2 ** (max_zoom - zoom)
    sw_zoom = math.ceil(sate_w / scale)
    sh_zoom = math.ceil(sate_h / scale)
    scaled  = [(math.ceil(x / scale), math.ceil(y / scale)) for x, y in px_corners]

    poly_q = _ordered_polygon(scaled)
    area_q = poly_q.area

    cx_t = int(sum(p[0] for p in scaled) / 4 // TILE_SIZE)
    cy_t = int(sum(p[1] for p in scaled) / 4 // TILE_SIZE)
    tx_max = sw_zoom // TILE_SIZE
    ty_max = sh_zoom // TILE_SIZE

    pos, semi = [], []
    for tx in range(max(0, cx_t - 5), min(cx_t + 6, tx_max + 1)):
        for ty in range(max(0, cy_t - 5), min(cy_t + 6, ty_max + 1)):
            tile_pts = [
                (tx * TILE_SIZE,       ty * TILE_SIZE),
                ((tx + 1) * TILE_SIZE, ty * TILE_SIZE),
                (tx * TILE_SIZE,       (ty + 1) * TILE_SIZE),
                ((tx + 1) * TILE_SIZE, (ty + 1) * TILE_SIZE),
            ]
            poly_t = _ordered_polygon(tile_pts)
            inter  = poly_q.intersection(poly_t).area
            union  = area_q + poly_t.area - inter
            iou    = inter / union if union > 0 else 0.0
            if iou >= IOU_SEMI:
                name   = f"{seq}_{zoom}_{tx:03}_{ty:03}.png"
                latlon = _tile_center_latlon(seq, zoom, tx, ty, sate_h, sate_w, max_zoom)
                entry  = (name, float(iou), list(latlon))
                if iou >= IOU_POS:
                    pos.append(entry)
                semi.append(entry)
    return pos, semi


# ---------------------------------------------------------------------------
# Per-image worker — all data passed via args, no global state
# ---------------------------------------------------------------------------

def _process_image(args):
    seq, img_name, lat, lon, height, heading, zoom_list, sate_h, sate_w, max_zoom = args

    lt_lat, lt_lon, rb_lat, rb_lon = SATE_LATLON[seq]
    corners = _coverage_corners(lat, lon, height, heading)
    px_corners = [
        _geo_to_px(la, lo, lt_lat, lt_lon, rb_lat, rb_lon, sate_h, sate_w)
        for la, lo in corners
    ]

    pos_imgs, pos_wts, pos_locs    = [], [], []
    semi_imgs, semi_wts, semi_locs = [], [], []

    for zoom in zoom_list:
        p_list, s_list = _find_overlapping_tiles(seq, zoom, px_corners, sate_h, sate_w, max_zoom)
        for name, iou, loc in p_list:
            pos_imgs.append(name); pos_wts.append(iou); pos_locs.append(loc)
        for name, iou, loc in s_list:
            semi_imgs.append(name); semi_wts.append(iou); semi_locs.append(loc)

    if not semi_imgs:
        return None

    return {
        "seq": seq, "drone_img_name": img_name,
        "lat": lat, "lon": lon,
        "pos_imgs": pos_imgs, "pos_wts": pos_wts, "pos_locs": pos_locs,
        "semi_imgs": semi_imgs, "semi_wts": semi_wts, "semi_locs": semi_locs,
    }


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _get_sate_size(data_root: Path, seq: str):
    tif = data_root / seq / f"satellite{seq}.tif"
    img = Image.open(tif)
    W, H = img.size
    img.close()
    return H, W


def _tile_seq(data_root: Path, seq: str, force: bool = False):
    tif_path = data_root / seq / f"satellite{seq}.tif"
    tile_dir  = data_root / seq / "tile"
    if not tif_path.exists():
        print(f"  [{seq}] TIF not found, skipping.")
        return

    # Get dimensions without loading pixels
    probe = Image.open(tif_path)
    W, H  = probe.size
    probe.close()
    max_zoom = math.ceil(math.log2(max(H, W) / TILE_SIZE))

    # Only generate the two zoom levels that will be copied to satellite/
    # (zoom_list[-3:-1] = the two second-to-finest levels, same as Game4Loc)
    zoom_levels = list(range(max_zoom + 1))[-3:-1]
    print(f"  [{seq}] {H}×{W} px, max_zoom={max_zoom}, "
          f"generating zoom levels {zoom_levels} only", flush=True)

    # Load the full TIF once — reused for both zoom levels
    img = Image.open(tif_path).convert("RGB")

    for zoom in zoom_levels:
        zoom_dir = tile_dir / str(zoom)
        zoom_dir.mkdir(parents=True, exist_ok=True)
        scale  = 2 ** (max_zoom - zoom)
        sw, sh = math.ceil(W / scale), math.ceil(H / scale)
        n_x, n_y = math.ceil(sw / TILE_SIZE), math.ceil(sh / TILE_SIZE)

        if not force and len(list(zoom_dir.glob("*.png"))) >= n_x * n_y:
            print(f"  [{seq}] zoom {zoom}: already complete ({n_x}×{n_y}), skipping")
            continue

        print(f"  [{seq}] zoom {zoom}: resizing to {sw}×{sh} ...", flush=True)
        scaled = img.resize((sw, sh), Image.Resampling.LANCZOS)
        print(f"  [{seq}] zoom {zoom}: writing {n_x}×{n_y}={n_x*n_y} tiles ...", flush=True)
        for xi in range(n_x):
            for yi in range(n_y):
                out = zoom_dir / f"{seq}_{zoom}_{xi:03}_{yi:03}.png"
                if not force and out.exists():
                    continue
                box  = (xi * TILE_SIZE, yi * TILE_SIZE,
                        min((xi + 1) * TILE_SIZE, sw),
                        min((yi + 1) * TILE_SIZE, sh))
                tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE))
                tile.paste(scaled.crop(box))
                tile.save(out)

    img.close()
    print(f"  [{seq}] tiling done.", flush=True)


def _copy_tiles(data_root: Path, seq: str, sate_out: Path):
    tile_dir = data_root / seq / "tile"
    # Collect whatever zoom dirs were generated (we now only create 2)
    zooms    = sorted(int(d.name) for d in tile_dir.iterdir() if d.is_dir())
    copied   = 0
    for zoom in zooms:
        for src in (tile_dir / str(zoom)).iterdir():
            if src.suffix == ".png" and not (sate_out / src.name).exists():
                shutil.copy2(src, sate_out / src.name)
                copied += 1
    return zooms, copied


def _copy_drone(data_root: Path, seq: str, drone_out: Path):
    src_dir = data_root / seq / "drone"
    copied  = 0
    for src in src_dir.iterdir():
        if src.suffix.upper() in {".JPG", ".JPEG", ".PNG"}:
            dst = drone_out / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    return copied


# ---------------------------------------------------------------------------
# Sequence processing
# ---------------------------------------------------------------------------

def _read_csv(data_root: Path, seq: str):
    rows = []
    with open(data_root / seq / f"{seq}.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append((row[1], float(row[3]), float(row[4]),
                         float(row[5]), float(row[-2])))
    return rows   # (img_name, lat, lon, height, heading)


def _get_zoom_list(data_root: Path, seq: str):
    """Return (selected_zoom_levels, max_zoom).

    Works whether all zoom dirs exist or only the 2 we generated.
    max_zoom is inferred from the TIF if tile/ has fewer than 3 dirs.
    """
    tile_dir = data_root / seq / "tile"
    zooms    = sorted(int(d.name) for d in tile_dir.iterdir() if d.is_dir())
    if len(zooms) >= 3:
        # All zoom levels present — pick same slice as Game4Loc
        return zooms[-3:-1], zooms[-1]
    else:
        # Only generated the 2 we need; max_zoom is one above the highest present
        tif = data_root / seq / f"satellite{seq}.tif"
        probe = Image.open(tif); W, H = probe.size; probe.close()
        max_zoom = math.ceil(math.log2(max(H, W) / TILE_SIZE))
        return zooms, max_zoom


def _process_seq(data_root: Path, seq: str, workers: int):
    rows             = _read_csv(data_root, seq)
    H, W             = _get_sate_size(data_root, seq)
    zoom_sel, max_zoom = _get_zoom_list(data_root, seq)

    task_args = [
        (seq, img, lat, lon, height, heading, zoom_sel, H, W, max_zoom)
        for img, lat, lon, height, heading in rows
    ]

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_image, a): a for a in task_args}
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc=f"  [{seq}]", leave=False):
            r = fut.result()
            if r is not None:
                results.append(r)
    return results


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def _to_entry(r):
    return {
        "drone_img_dir":   "drone/images",
        "drone_img_name":  r["drone_img_name"],
        "drone_loc_lat_lon": [r["lat"], r["lon"]],
        "sate_img_dir":    "satellite",
        "pair_pos_sate_img_list":          r["pos_imgs"],
        "pair_pos_sate_weight_list":       r["pos_wts"],
        "pair_pos_sate_loc_lat_lon_list":  r["pos_locs"],
        "pair_pos_semipos_sate_img_list":          r["semi_imgs"],
        "pair_pos_semipos_sate_weight_list":       r["semi_wts"],
        "pair_pos_semipos_sate_loc_lat_lon_list":  r["semi_locs"],
        "drone_metadata": {
            "height": None, "drone_roll": None, "drone_pitch": None,
            "drone_yaw": None, "cam_roll": None, "cam_pitch": None, "cam_yaw": None,
        },
    }


def _write_json(data_root: Path, entries: list, split_type: str, split_name: str):
    path = data_root / f"{split_type}-drone2sate-{split_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_to_entry(r) for r in entries], f, indent=2, ensure_ascii=False)
    print(f"  {len(entries):5d} entries -> {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Prepare UAV-VisLoc dataset for Game4Loc-compatible training",
    )
    ap.add_argument("--data-root",    required=True, type=Path)
    ap.add_argument("--split",        choices=["same-area", "cross-area", "both"],
                    default="both")
    ap.add_argument("--seqs",         nargs="+", type=int, default=[3, 4],
                    help="Sequences for same-area split (default: 3 4)")
    ap.add_argument("--train-seqs",   nargs="+", type=int, default=[1, 3],
                    help="Train sequences for cross-area (default: 1 3, Game4Loc paper)")
    ap.add_argument("--test-seqs",    nargs="+", type=int, default=[2, 4],
                    help="Test sequences for cross-area (default: 2 4, Game4Loc paper)")
    ap.add_argument("--no-tiling",    action="store_true",
                    help="Skip tiling (use when tiles already exist)")
    ap.add_argument("--force-tiling", action="store_true",
                    help="Re-generate tiles even if they exist")
    ap.add_argument("--workers",      type=int, default=4,
                    help="Worker processes for IoU matching (default: 4)")
    ap.add_argument("--seed",         type=int, default=42)
    args = ap.parse_args()

    data_root  = args.data_root
    same_seqs  = [f"{s:02}" for s in sorted(set(args.seqs))]
    train_seqs = [f"{s:02}" for s in sorted(set(args.train_seqs))]
    test_seqs  = [f"{s:02}" for s in sorted(set(args.test_seqs))]
    all_seqs   = sorted(set(same_seqs + train_seqs + test_seqs))

    sate_out  = data_root / "satellite"
    drone_out = data_root / "drone" / "images"
    sate_out.mkdir(parents=True, exist_ok=True)
    drone_out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: tile ──────────────────────────────────────────────────────
    if not args.no_tiling:
        print("=== Step 1/4: Tile satellite images ===")
        for seq in all_seqs:
            print(f"\n[{seq}] tiling satellite{seq}.tif ...")
            _tile_seq(data_root, seq, force=args.force_tiling)

        print("\n=== Step 2/4: Copy tiles -> satellite/ ===")
        for seq in all_seqs:
            sel, n = _copy_tiles(data_root, seq, sate_out)
            print(f"  [{seq}] zoom levels {sel}, {n} tiles copied")

        print("\n=== Step 3/4: Copy drone images -> drone/images/ ===")
        for seq in all_seqs:
            n = _copy_drone(data_root, seq, drone_out)
            print(f"  [{seq}] {n} drone images copied")
    else:
        print("Skipping tiling (--no-tiling).")
        print("\n=== Step 2/4: Copy tiles -> satellite/ ===")
        for seq in all_seqs:
            sel, n = _copy_tiles(data_root, seq, sate_out)
            print(f"  [{seq}] zoom levels {sel}, {n} tiles copied")
        print("\n=== Step 3/4: Copy drone images -> drone/images/ ===")
        for seq in all_seqs:
            n = _copy_drone(data_root, seq, drone_out)
            print(f"  [{seq}] {n} drone images copied")

    # ── Step 2: IoU matching ───────────────────────────────────────────────
    print("\n=== Step 4/4: Drone <-> tile IoU matching ===")
    seq_results = {}
    for seq in all_seqs:
        rows = _read_csv(data_root, seq)
        print(f"\n[{seq}] {len(rows)} drone images ...")
        seq_results[seq] = _process_seq(data_root, seq, args.workers)
        n_matched = len(seq_results[seq])
        n_pos = sum(len(r["pos_imgs"]) > 0 for r in seq_results[seq])
        print(f"  [{seq}] {n_matched} images matched, {n_pos} with strong positives")

    # ── Step 3: Write JSON ─────────────────────────────────────────────────
    print("\n=== Writing JSON files ===")

    if args.split in ("same-area", "both"):
        combined = []
        for seq in same_seqs:
            combined.extend(seq_results.get(seq, []))
        random.seed(args.seed)
        random.shuffle(combined)
        cut = len(combined) * 4 // 5
        _write_json(data_root, combined[:cut], "same-area", "train")
        _write_json(data_root, combined[cut:], "same-area", "test")

    if args.split in ("cross-area", "both"):
        _write_json(data_root,
                    [r for s in train_seqs for r in seq_results.get(s, [])],
                    "cross-area", "train")
        _write_json(data_root,
                    [r for s in test_seqs  for r in seq_results.get(s, [])],
                    "cross-area", "test")

    total_tiles = len(list(sate_out.glob("*.png")))
    print(f"\nDone.  satellite/ has {total_tiles} tiles.")
    print(f"Set training data root to: {data_root}")


if __name__ == "__main__":
    main()
