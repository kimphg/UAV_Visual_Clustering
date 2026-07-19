"""Fine geometric verification for cross-view matching.

Coarse retrieval (global descriptor cosine similarity) finds visually similar
candidates but has no notion of geographic alignment: two locations can look
alike yet be tens/hundreds of metres apart. This module takes the per-location
SPATIAL descriptors (model.encode_spatial_features's stage-2 grid, [H,W,C]) of
a query and a candidate gallery tile, matches individual grid locations between
them, robustly fits a 2D transform (rotation + translation, scale optional),
and converts the fitted translation into a real-world metre shift. A match is
rejected if that shift exceeds a threshold, regardless of how high its global
descriptor similarity was.

Pipeline per (query, gallery-candidate) pair:
  1. Flatten each [H,W,C] grid to [H*W, C], L2-normalize each location's vector.
  2. Mutual nearest-neighbour matching (cosine) between query and gallery grid
     locations — keeps only reciprocal, above-threshold matches (cheap, robust
     pre-filter before RANSAC).
  3. RANSAC-fit a similarity transform (rotation + uniform scale + translation)
     between matched grid COORDINATES via a closed-form 2-point Procrustes
     solve wrapped in a manual RANSAC loop (_fit_similarity_ransac), which
     also rejects any hypothesis whose scale falls outside scale_range before
     it can compete on inlier count — guards against a couple of spurious
     token matches locking onto a degenerate near-zero/huge-scale "fit". The
     fitted translation comes out already expressed in the GALLERY's own grid
     frame (query -> gallery), so no query-side metadata is needed to read
     off the shift.
  4. Convert the grid-unit translation to metres using the gallery tile's
     known ground footprint width (metres_per_token = footprint_m / grid_size).
  5. Accept/reject by comparing the metre shift to reject_threshold_m.

estimate_scale controls what happens with the fitted scale factor:
  - True  (variable-altitude datasets, e.g. DenseUAV H80/H90/H100, UAV-VisLoc):
           trust the general similarity fit; scale absorbs any query/gallery
           altitude mismatch so the translation stays a clean grid-frame shift.
           scale_range (default 0.7-1.5x) bounds how much altitude mismatch is
           considered plausible — tune per dataset if its altitude bands imply
           a wider/narrower real footprint ratio.
  - False (fixed-altitude datasets, e.g. University-1652, SUES-200 per band):
           fit a RIGID transform (rotation + translation only, scale locked to
           1) via a closed-form Procrustes/Kabsch solve wrapped in a small
           manual RANSAC loop — avoids a spurious scale absorbing noise that
           should be attributed to rotation/translation when scale is known
           to be ~1.
"""
import math
import random

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans


def _flatten_normalize(spatial: torch.Tensor) -> np.ndarray:
    """[H,W,C] (or already [N,C]) -> L2-normalized [N,C] numpy float32."""
    if spatial.dim() == 3:
        h, w, c = spatial.shape
        spatial = spatial.reshape(h * w, c)
    x = spatial.detach().cpu().float().numpy()
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return x


def _grid_coords(h: int, w: int) -> np.ndarray:
    """[H*W, 2] (row, col) coordinates matching a row-major [H,W,C].reshape(H*W,C)."""
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.float32)


def mutual_nn_matches(query_spatial: torch.Tensor, gallery_spatial: torch.Tensor,
                      sim_threshold: float = 0.5):
    """Mutual (reciprocal) nearest-neighbour token matches between two spatial grids.

    Returns (query_coords [M,2], gallery_coords [M,2], similarities [M]) — grid
    (row, col) coordinates of the M matched location pairs, or empty arrays if
    query_spatial/gallery_spatial don't share a grid size (H,W must match — same
    backbone/img_size for both), or no match clears sim_threshold.
    """
    qh, qw = query_spatial.shape[0], query_spatial.shape[1]
    gh, gw = gallery_spatial.shape[0], gallery_spatial.shape[1]
    q = _flatten_normalize(query_spatial)          # [Nq, C]
    g = _flatten_normalize(gallery_spatial)         # [Ng, C]
    sim = q @ g.T                                    # [Nq, Ng]

    q_best = np.argmax(sim, axis=1)                  # each query's best gallery match
    g_best = np.argmax(sim, axis=0)                  # each gallery's best query match
    q_idx = np.arange(sim.shape[0])
    mutual = g_best[q_best] == q_idx                 # reciprocal only
    q_sel = q_idx[mutual]
    g_sel = q_best[mutual]
    sims = sim[q_sel, g_sel]

    keep = sims >= sim_threshold
    q_sel, g_sel, sims = q_sel[keep], g_sel[keep], sims[keep]
    if q_sel.size == 0:
        return (np.zeros((0, 2), np.float32), np.zeros((0, 2), np.float32),
                np.zeros((0,), np.float32))

    q_coords = _grid_coords(qh, qw)[q_sel]
    g_coords = _grid_coords(gh, gw)[g_sel]
    return q_coords, g_coords, sims


def _reflection_inlier_count(src, dst, s_c, d_c, U, Vt, d_sign, thresh_px,
                             S=None, var_s=None, fixed_scale=None):
    """Given the SVD factors (U, Vt, d_sign) of the ACCEPTED proper-rotation
    fit, construct the opposite-handedness (mirror-reflected) transform and
    count how many of the FULL correspondence set (src/dst) it explains
    within thresh_px. Comparing this against the proper fit's own
    inlier_count is how verify_and_reject detects mirror-ambiguous matches
    (see its max_reflection_ratio parameter): genuine, correctly-oriented
    content is essentially never also well-explained by its own mirror
    image, so a HIGH reflection inlier count relative to the proper fit's is
    a red flag — e.g. a bilaterally-symmetric building complex matched
    against its own left-right flipped duplicate, which looks like a
    confident geometric fit if you only ever check proper rotations.

    fixed_scale bypasses scale estimation for the rigid (scale=1) case; pass
    S (singular values) + var_s instead to fit the reflected transform's own
    optimal scale via the same Umeyama formula used for the proper fit.
    """
    D_reflected = np.diag([1.0, -d_sign])
    R_reflected = Vt.T @ D_reflected @ U.T
    if fixed_scale is not None:
        scale_reflected = fixed_scale
    else:
        scale_reflected = (float((S * np.array([1.0, -d_sign])).sum() / var_s)
                           if var_s is not None and var_s > 1e-8 else 1.0)
    t_reflected = d_c - scale_reflected * (R_reflected @ s_c)
    pred = (scale_reflected * (R_reflected @ src.T)).T + t_reflected
    err = np.linalg.norm(pred - dst, axis=1)
    return int((err < thresh_px).sum())


def _fit_rigid_ransac(src: np.ndarray, dst: np.ndarray, thresh_px: float = 2.0,
                      iters: int = 64, seed: int = 0):
    """RANSAC rotation+translation (scale locked to 1) via closed-form 2-point
    Procrustes per trial. src/dst: [N,2]. Returns (R [2,2], t [2], inlier_mask)
    or None if fewer than 2 correspondences.
    """
    n = src.shape[0]
    if n < 2:
        return None
    rng = random.Random(seed)
    best_inliers = None
    best_count = -1
    idx_all = list(range(n))
    for _ in range(iters):
        i, j = rng.sample(idx_all, 2)
        s2 = src[[i, j]]
        d2 = dst[[i, j]]
        s_c = s2.mean(axis=0)
        d_c = d2.mean(axis=0)
        s0 = s2 - s_c
        d0 = d2 - d_c
        # Closed-form 2D rigid (Kabsch, no scale) from 2 point pairs.
        h = s0.T @ d0
        U, _, Vt = np.linalg.svd(h)
        d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, d_sign])
        R = Vt.T @ D @ U.T
        t = d_c - R @ s_c
        pred = (R @ src.T).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = err < thresh_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 2:
        return None
    # Refit on all inliers for the final estimate.
    s_in, d_in = src[best_inliers], dst[best_inliers]
    s_c, d_c = s_in.mean(axis=0), d_in.mean(axis=0)
    s0, d0 = s_in - s_c, d_in - d_c
    h = s0.T @ d0
    U, _, Vt = np.linalg.svd(h)
    d_sign = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d_sign])
    R = Vt.T @ D @ U.T
    t = d_c - R @ s_c
    reflection_inlier_count = _reflection_inlier_count(
        src, dst, s_c, d_c, U, Vt, d_sign, thresh_px, fixed_scale=1.0)
    return R, t, best_inliers, reflection_inlier_count


def _fit_similarity_ransac(src: np.ndarray, dst: np.ndarray, thresh_px: float = 2.0,
                           iters: int = 64, seed: int = 0,
                           scale_range: tuple = (0.7, 1.5)):
    """RANSAC rotation+uniform-scale+translation via closed-form 2-point fit,
    discarding any hypothesis whose 2-point scale falls outside scale_range.

    Without this, a plain best-inlier-count RANSAC (e.g. cv2's) can lock onto
    a degenerate near-zero or huge scale driven by just a couple of spurious
    token matches, since that hypothesis can still "explain" a few points to
    within thresh_px purely by accident. Rejecting out-of-range hypotheses
    outright — before they ever get to compete on inlier count — prevents
    that; only the final least-squares refit is (lightly) clamped as a safety
    net. src/dst: [N,2]. Returns (R [2,2], scale, t [2], inlier_mask) or None.

    iters=64 (down from an earlier, unjustified 300): for a 2-point minimal
    sample, standard RANSAC theory (k = ln(1-p) / ln(1-w^2), p=0.99 confidence)
    needs only ~50 trials even at a pessimistic 30% inlier ratio, ~16 at 50%.
    This matters at DenseUAV eval scale: with auto_escalate_ransac retrying up
    to 4x on top of this, a full run does n_queries * top_k * iters * (up to 5
    attempts) RANSAC trials — at 300 iters this was the dominant eval cost.
    """
    n = src.shape[0]
    if n < 2:
        return None
    rng = random.Random(seed)
    best_inliers = None
    best_count = -1
    idx_all = list(range(n))
    for _ in range(iters):
        i, j = rng.sample(idx_all, 2)
        v_s = src[j] - src[i]
        v_d = dst[j] - dst[i]
        norm_s = np.linalg.norm(v_s)
        if norm_s < 1e-6:
            continue
        scale = np.linalg.norm(v_d) / norm_s
        if not (scale_range[0] <= scale <= scale_range[1]):
            continue
        theta = math.atan2(v_d[1], v_d[0]) - math.atan2(v_s[1], v_s[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        t = dst[i] - scale * (R @ src[i])
        pred = (scale * (R @ src.T)).T + t
        err = np.linalg.norm(pred - dst, axis=1)
        inliers = err < thresh_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
    if best_inliers is None or best_count < 2:
        return None
    # Refit on all inliers via closed-form Umeyama (with scaling).
    s_in, d_in = src[best_inliers], dst[best_inliers]
    s_c, d_c = s_in.mean(axis=0), d_in.mean(axis=0)
    s0, d0 = s_in - s_c, d_in - d_c
    n_in = s_in.shape[0]
    h = (s0.T @ d0) / n_in            # Umeyama's Sigma_xy has a 1/n factor —
    U, S, Vt = np.linalg.svd(h)       # dropping it left scale off by n (R is
    d_sign = np.sign(np.linalg.det(Vt.T @ U.T))  # unaffected: scaling h by a
    D = np.diag([1.0, d_sign])                   # positive constant doesn't
    R = Vt.T @ D @ U.T                           # change its singular vectors).
    var_s = (s0 ** 2).sum() / n_in
    scale = float((S * np.array([1.0, d_sign])).sum() / var_s) if var_s > 1e-8 else 1.0
    scale = float(np.clip(scale, scale_range[0], scale_range[1]))
    t = d_c - scale * (R @ s_c)
    reflection_inlier_count = _reflection_inlier_count(
        src, dst, s_c, d_c, U, Vt, d_sign, thresh_px, S=S, var_s=var_s)
    return R, scale, t, best_inliers, reflection_inlier_count


def fit_transform(query_coords: np.ndarray, gallery_coords: np.ndarray,
                  estimate_scale: bool, ransac_thresh_px: float = 2.0,
                  scale_range: tuple = (0.7, 1.5), ransac_iters: int = 64):
    """Fit query -> gallery 2D transform. Returns dict with keys:
    translation [2] (row,col shift in GALLERY grid units), scale (float),
    rotation_deg (float), inlier_count (int), n_matches (int),
    reflection_inlier_count (int, see verify_and_reject's max_reflection_ratio);
    or None if too few correspondences to fit (need >=2).

    scale_range bounds the accepted scale to a plausible band (default
    0.7-1.5x) when estimate_scale=True — see _fit_similarity_ransac.
    """
    n = query_coords.shape[0]
    if estimate_scale:
        fit = _fit_similarity_ransac(query_coords, gallery_coords,
                                     thresh_px=ransac_thresh_px,
                                     scale_range=scale_range, iters=ransac_iters)
        if fit is None:
            return None
        R, scale, t, inlier_mask, reflection_inlier_count = fit
        rotation_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        translation = t.astype(np.float32)               # (row, col)
        inlier_count = int(inlier_mask.sum())
    else:
        fit = _fit_rigid_ransac(query_coords, gallery_coords, thresh_px=ransac_thresh_px,
                                iters=ransac_iters)
        if fit is None:
            return None
        R, t, inlier_mask, reflection_inlier_count = fit
        scale = 1.0
        rotation_deg = math.degrees(math.atan2(R[1, 0], R[0, 0]))
        translation = t.astype(np.float32)              # (row, col)
        inlier_count = int(inlier_mask.sum())
    return {
        "translation": translation, "scale": float(scale),
        "rotation_deg": float(rotation_deg),
        "inlier_count": inlier_count, "n_matches": n,
        "reflection_inlier_count": reflection_inlier_count,
    }


def verify_and_reject(query_spatial: torch.Tensor, gallery_spatial: torch.Tensor,
                      gallery_footprint_m: float, *, estimate_scale: bool = True,
                      reject_threshold_m: float = 20.0, sim_threshold: float = 0.5,
                      ransac_thresh_px: float = 2.0, min_inliers: int = 4,
                      scale_range: tuple = (0.7, 1.5), auto_escalate_ransac: bool = True,
                      ransac_escalate_factor: float = 2.0, max_ransac_escalations: int = 4,
                      ransac_iters: int = 64, check_reflection: bool = True,
                      max_reflection_ratio: float = 0.7):
    """End-to-end: match tokens, fit transform, convert to metres, accept/reject.

    gallery_footprint_m: real-world width (metres) the gallery tile's full grid
        spans — used to convert the fitted grid-unit translation to metres.
        (Grid is square: metres_per_token = gallery_footprint_m / grid_size.)

    auto_escalate_ransac: if the fit at ransac_thresh_px doesn't clear
        min_inliers, retry with a looser threshold (x ransac_escalate_factor
        each time, up to max_ransac_escalations attempts, capped at half the
        grid size) instead of rejecting outright as "too few inliers". This is
        safe because the actual accept/reject decision downstream is the
        real-world metre-shift check — a looser RANSAC tolerance can only let
        a BAD fit's large recovered shift get caught by reject_threshold_m
        anyway, so there's no benefit to failing early on inlier count alone
        when a looser fit might reveal a genuinely good, well-supported match.

    check_reflection: reject matches where a mirror-reflected transform
        explains as many of the SAME correspondence points as the accepted
        proper-rotation fit did (ratio >= max_reflection_ratio) — a sign the
        fit's handedness is data-ambiguous (e.g. a symmetric building complex
        matched to its own flipped duplicate), not a confidently correct
        orientation. See _reflection_inlier_count.

    Returns dict: accept (bool), shift_m (float or None), reason (str),
    plus the raw fit_transform() output under "fit" (or None if unfit-able).
    fit["ransac_thresh_used"] records which threshold produced the returned
    fit (may be larger than ransac_thresh_px if escalation kicked in).
    A pair with too few matches / too few RANSAC inliers even at the loosest
    attempted threshold is REJECTED — an unverifiable match is not a safe match.
    """
    q_coords, g_coords, sims = mutual_nn_matches(query_spatial, gallery_spatial,
                                                  sim_threshold=sim_threshold)
    if q_coords.shape[0] < 3:
        return {"accept": False, "shift_m": None, "reason": "too few token matches",
               "fit": None}

    grid_size = gallery_spatial.shape[0]
    max_thresh = grid_size / 2.0
    thresh = ransac_thresh_px
    fit = None
    for attempt in range(max_ransac_escalations + 1):
        fit = fit_transform(q_coords, g_coords, estimate_scale=estimate_scale,
                            ransac_thresh_px=thresh, scale_range=scale_range,
                            ransac_iters=ransac_iters)
        if fit is not None and fit["inlier_count"] >= min_inliers:
            fit["ransac_thresh_used"] = thresh
            break
        if not auto_escalate_ransac or attempt == max_ransac_escalations:
            if fit is not None:
                fit["ransac_thresh_used"] = thresh
            break
        thresh = min(thresh * ransac_escalate_factor, max_thresh)

    if fit is None or fit["inlier_count"] < min_inliers:
        return {"accept": False, "shift_m": None,
               "reason": f"transform fit failed or too few inliers even at "
                         f"RANSAC threshold={thresh:.1f} "
                         f"({0 if fit is None else fit['inlier_count']}<{min_inliers})",
               "fit": fit}

    metres_per_token = gallery_footprint_m / grid_size
    # fit['translation'] is defined around the grid's (0,0) CORNER, so its raw
    # norm conflates genuine positional drift with "rotation swing": a pure
    # in-place rotation about the tile CENTER still produces a large nonzero
    # corner-referenced translation (e.g. ~20 tokens at 80 degrees), because
    # rotating about a point far from the pivot used in the fit requires a
    # large compensating offset even with zero real displacement. What we
    # actually want is how far the query tile's own CENTER ends up relative to
    # the gallery tile's center once mapped through the fit.
    center = np.array([(grid_size - 1) / 2.0, (grid_size - 1) / 2.0], dtype=np.float32)
    theta = math.radians(fit["rotation_deg"])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    predicted_center = fit["scale"] * (rot @ center) + fit["translation"]
    center_shift = predicted_center - center
    shift_m = float(np.linalg.norm(center_shift) * metres_per_token)

    if shift_m > reject_threshold_m:
        return {"accept": False, "shift_m": shift_m,
               "reason": f"required shift {shift_m:.1f}m > {reject_threshold_m:.1f}m",
               "fit": fit}

    # Mirror/reflection check: a genuine, correctly-oriented match's token
    # positions are essentially never ALSO well-explained by their own mirror
    # image. If a reflected transform explains a comparable fraction of the
    # SAME correspondence set, the fit's handedness isn't actually determined
    # by the data — a red flag for e.g. a bilaterally-symmetric building
    # complex matched against its own left-right flipped duplicate, which
    # would otherwise look like a confident, small-shift, well-supported fit.
    if check_reflection:
        reflection_ratio = fit["reflection_inlier_count"] / max(fit["inlier_count"], 1)
        if reflection_ratio >= max_reflection_ratio:
            return {"accept": False, "shift_m": shift_m,
                   "reason": f"mirror-ambiguous: reflected fit explains "
                             f"{fit['reflection_inlier_count']}/{fit['inlier_count']} "
                             f"as many points as the accepted fit "
                             f"(ratio={reflection_ratio:.2f} >= {max_reflection_ratio:.2f})",
                   "fit": fit}

    return {"accept": True, "shift_m": shift_m, "reason": "ok", "fit": fit}


def rerank_candidates(query_spatial: torch.Tensor, candidate_spatials: list,
                      candidate_footprints_m: list, **verify_kwargs):
    """Geometrically verify a query against a LIST of candidate spatial
    descriptors (e.g. a query's top-K coarse-retrieval gallery candidates) and
    return one result dict per candidate, in the SAME order as candidate_spatials:

      {"shift_m": float or None, "accept": bool, "match_confidence": float,
       "fit": dict or None, "reason": str}

    reason is verify_and_reject's own diagnostic string ("ok", "too few token
    matches", "transform fit failed or too few inliers ...", "required shift
    ...m > ...m") — useful for aggregating WHY candidates were rejected across
    many queries, not just how many.

    match_confidence = RANSAC inlier_count / n_matches (0.0 if no transform
    could be fit at all) — a [0,1] measure of how well-supported the fitted
    transform is, independent of the actual shift distance. Use this for a
    "best matching probability" ranking; use shift_m (ascending, None treated
    as worst) for a "closest predicted location" ranking. Two different, valid
    ways to re-rank the same candidate set — kept separate since they can
    disagree (e.g. a confidently-fit transform that implies a large shift).

    verify_kwargs are forwarded to verify_and_reject (estimate_scale,
    reject_threshold_m, sim_threshold, ransac_thresh_px, min_inliers,
    scale_range, auto_escalate_ransac, etc).
    """
    results = []
    for cand_spatial, footprint_m in zip(candidate_spatials, candidate_footprints_m):
        res = verify_and_reject(query_spatial, cand_spatial, footprint_m, **verify_kwargs)
        fit = res["fit"]
        match_confidence = (fit["inlier_count"] / fit["n_matches"]
                            if fit is not None and fit["n_matches"] > 0 else 0.0)
        results.append({
            "shift_m": res["shift_m"], "accept": res["accept"],
            "match_confidence": match_confidence, "fit": fit,
            "reason": res["reason"],
        })
    return results


# ── Spatial descriptor compression (vector quantization / visual codebook) ──
#
# [24,24,512] float32 costs ~1.1 MB/image (~20 GB for an 18k-image gallery).
# Quantizing each grid location's 512-D descriptor to its nearest of K shared
# ("global", fit once over a sample of tokens from many images) centroids lets
# us store only a per-location cluster ID: [24,24] uint8 (K<=256) — 576 bytes/
# image, ~10 MB for the same 18k-image gallery (~2000x smaller), plus one
# shared codebook (K x 512 floats, tens of KB to a couple MB, stored once).
#
# Quantized descriptors plug straight back into mutual_nn_matches/fit_transform
# UNCHANGED: reconstruct_from_ids looks up each location's assigned centroid as
# its approximate descriptor, then the existing cosine-based matcher runs as-is.
# The real trade-off is not storage (tiny either way) but MATCHING PRECISION:
# collapsing a continuous 512-D descriptor to 1-of-K categories means every
# token sharing a cluster ID becomes indistinguishable (identical reconstructed
# vector, cosine similarity exactly 1.0 to every other same-cluster token) —
# increasing false/ambiguous token correspondences before RANSAC has to sort
# them out. K=256 (a standard Bag-of-Visual-Words size, fits in uint8) is the
# default; validate empirically (see test harness) before trusting it blindly.

def fit_codebook(token_descriptors: np.ndarray, n_clusters: int = 256, seed: int = 42):
    """K-means codebook over sampled stage-2 tokens (rows of [N,512], any mix
    of images/locations — e.g. sample ~50-200 tokens each from a few hundred
    representative images). Returns L2-normalized centroids [K,512] float32.
    """
    x = token_descriptors.astype(np.float32)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(x)
    centroids = km.cluster_centers_.astype(np.float32)
    centroids /= (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    return centroids


def quantize_spatial(spatial: torch.Tensor, codebook: np.ndarray) -> np.ndarray:
    """[H,W,C] spatial descriptor + codebook [K,C] -> [H,W] cluster-id array
    (uint8 if K<=256, else uint16) — nearest centroid per grid location."""
    h, w, c = spatial.shape
    x = _flatten_normalize(spatial)                  # [H*W, C], L2-normalized
    sims = x @ codebook.T                             # [H*W, K]
    ids = np.argmax(sims, axis=1)
    dtype = np.uint8 if codebook.shape[0] <= 256 else np.uint16
    return ids.astype(dtype).reshape(h, w)


def reconstruct_from_ids(cluster_ids: np.ndarray, codebook: np.ndarray) -> torch.Tensor:
    """Inverse of quantize_spatial: [H,W] cluster ids + codebook [K,C] -> the
    approximate [H,W,C] descriptor (each location = its assigned centroid),
    ready to feed straight into mutual_nn_matches/verify_and_reject unchanged."""
    h, w = cluster_ids.shape
    approx = codebook[cluster_ids.reshape(-1)]         # [H*W, C]
    return torch.from_numpy(approx.reshape(h, w, -1).astype(np.float32))


def warp_to_gallery(query_image: np.ndarray, fit: dict, grid_size: int,
                    gallery_hw: tuple) -> np.ndarray:
    """Warp a query image into the gallery's pixel frame using a fitted
    query->gallery transform (fit_transform's output), for visual before/after
    confirmation that the recovered rotation/scale/translation is sane.

    query_image: [H,W,3] uint8 array, already resized to the SAME square size
        fed to the model (so its pixel grid lines up with the token grid).
    gallery_hw: (H,W) pixel size of the gallery image/frame to warp into
        (usually equal to query_image's own size).

    fit_transform's translation/rotation/scale operate in GRID-token units
    around the grid's (0,0) corner (row,col order, forward query->gallery:
    gallery = scale * R(rotation_deg) @ query + translation). This rebuilds
    that as a fresh pixel-space cv2 affine matrix (scaling translation by
    pixels-per-token) rather than reusing fit_transform's internal cv2 M,
    since that M's (x,y) slot assignment is deliberately non-standard (row in
    the x-slot) and reusing it directly against real image pixel arrays would
    silently swap axes.
    """
    gh, gw = gallery_hw
    ppt_row = gh / grid_size
    ppt_col = gw / grid_size
    t_row, t_col = fit["translation"]
    theta = math.radians(fit["rotation_deg"])
    s = fit["scale"]
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # Forward (query->gallery) in pixel (x=col, y=row) convention, built from
    # the same row_g = s*cos*row_q - s*sin*col_q + t_row (etc.) relation that
    # fit_transform's rotation_deg/scale/translation were extracted from.
    M = np.array([
        [ s * cos_t,  s * sin_t, t_col * ppt_col],
        [-s * sin_t,  s * cos_t, t_row * ppt_row],
    ], dtype=np.float32)
    return cv2.warpAffine(query_image, M, dsize=(gw, gh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def compression_stats(grid_size: int, n_clusters: int, n_gallery: int, channels: int = 512):
    """Report the storage trade-off for a given (grid_size, K, gallery size)."""
    id_bytes = 1 if n_clusters <= 256 else 2
    per_image_bytes = grid_size * grid_size * id_bytes
    codebook_bytes = n_clusters * channels * 4
    original_bytes = grid_size * grid_size * channels * 4
    return {
        "per_image_quantized_bytes": per_image_bytes,
        "per_image_original_bytes": original_bytes,
        "codebook_bytes": codebook_bytes,
        "full_gallery_quantized_mb": (per_image_bytes * n_gallery + codebook_bytes) / 1024**2,
        "full_gallery_original_gb": (original_bytes * n_gallery) / 1024**3,
        "compression_ratio": original_bytes / per_image_bytes,
    }
