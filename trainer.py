import hashlib
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.utils.data import DataLoader
from clustering import ClusterIndex, assign_fixed_centroids
from loss import weighted_info_nce, group_whole_slice_info_nce
from model import NUM_CLUSTERS

# Cache of tile file path -> md5 content hash, so repeated cluster-data rebuilds
# (once per epoch) don't re-hash unchanged files. Catches tiles that are
# byte-identical under DIFFERENT paths (e.g. University-1652 building IDs 0761
# and 0768 share one duplicated satellite image under the official release) --
# a case pair_to_pos_tile's path-keyed identity cannot see, since the paths
# genuinely differ even though the pixels don't.
_TILE_CONTENT_HASH_CACHE = {}


def _tile_content_hash(path_str):
    cached = _TILE_CONTENT_HASH_CACHE.get(path_str)
    if cached is not None:
        return cached
    try:
        with open(path_str, "rb") as f:
            h = hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None
    _TILE_CONTENT_HASH_CACHE[path_str] = h
    return h


def even_batch_size(n_samples: int, target_batch_size: int) -> int:
    """Return a batch size so all batches (including the last) differ by at most 1 sample."""
    if n_samples <= target_batch_size:
        return n_samples
    n_batches = math.ceil(n_samples / target_batch_size)
    return math.ceil(n_samples / n_batches)


NEGATIVES_PER_QUERY = 9  # 6 from same cluster + 3 from other clusters
STAGE1_PRIMARY_NEGATIVE_RATIO = 6/9  # 6 from same cluster (hardest)
STAGE1_GLOBAL_NEGATIVE_RATIO = 3/9   # 3 from nearest/other clusters
PRIMARY_NEGATIVE_TAU = 0.07
GLOBAL_NEGATIVE_TAU = 0.10
STAGE1_ALIGNMENT_WEIGHT = 1.00
STAGE1_ALIGNMENT_TAU = 0.07
STAGE1_VARIANCE_WEIGHT = 1.00
STAGE1_NEGATIVE_WEIGHT = 10.00
STAGE1_NEGATIVE_MARGIN = 0.25
# Ramp negative weight from 0 → full over this many batches at the start of each
# epoch (after reclustering).  Prevents the loss spike caused by sudden new hard
# negatives that the model hasn't adapted to yet.
NEGATIVE_WARMUP_STEPS = 10
# Adaptive reclustering: while the positive-anchor same-cluster probability is
# below this, queries and their positives are NOT co-clustering well (embeddings
# still drifting), so recluster every epoch to keep assignments meaningful. Once
# it reaches the threshold the clustering is stable and cluster_every applies.
RECLUSTER_POS_SAME_PROB = 0.80
STAGE1_CLUSTER_CONSISTENCY_WEIGHT = 2.00
STAGE1_CLUSTER_CONSISTENCY_TAU = 0.10
# Game4Loc-style label smoothing for WeightedInfoNCE / GroupInfoNCE.
# 0.0 = hard InfoNCE (original behaviour); ~0.05-0.10 = soft.
STAGE1_LABEL_SMOOTHING = 0.05
# Group size for GroupInfoNCE whole_slice. 1 = standard (WeightedInfoNCE).
# 2 = each pair produces 2 independent augmentations; requires dataset group_size=2.
STAGE1_GROUP_SIZE = 1
TRAINING_MODE_GLOBAL = "cluster_head_512"
POSITIVE_CLUSTER_TARGET_PROBABILITY = 0.98
AUTO_K_CANDIDATES = [4, 8, 12, 16, 24, 32, 48, 64]
AUTO_K_MAX_SAMPLES = 2000


def find_optimal_k_silhouette(embeddings, k_candidates=None, max_samples=AUTO_K_MAX_SAMPLES, status_callback=None):
    """Sweep K-means over k_candidates and return (best_k, scores_dict) via silhouette score.

    Expects L2-normalized float32 embeddings (numpy array). Uses cosine distance.
    Subsamples to max_samples for speed. Returns (None, {}) if sklearn unavailable.
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        if status_callback:
            status_callback("Auto-K: sklearn not available, falling back to default K.")
        return None, {}

    import numpy as np

    if k_candidates is None:
        k_candidates = AUTO_K_CANDIDATES

    n = len(embeddings)
    if n > max_samples:
        idx = np.random.default_rng(42).choice(n, max_samples, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    k_candidates = [k for k in k_candidates if 2 <= k < len(sample)]
    if not k_candidates:
        return None, {}

    best_k, best_score = k_candidates[0], -2.0
    scores = {}

    for k in k_candidates:
        if status_callback:
            status_callback(f"Auto-K: testing K={k}...")
        try:
            ci = ClusterIndex(n_clusters=k, dim=sample.shape[1], nredo=3)
            ci.fit(sample)
            labels = ci.assign(sample)
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            score = float(silhouette_score(sample, labels, metric="cosine"))
            scores[k] = score
            if score > best_score:
                best_score, best_k = score, k
        except Exception as e:
            if status_callback:
                status_callback(f"Auto-K: K={k} skipped ({e}).")

    if status_callback:
        parts = ", ".join(f"K{k}={v:.3f}" for k, v in sorted(scores.items()))
        status_callback(f"Auto-K silhouette: [{parts}] → best K={best_k} (score={best_score:.3f})")

    return best_k, scores
from dataset import make_eval_transform

def train(
    model,
    loader,
    optimizer,
    device,
    epochs=1,
    use_amp=False,
    cluster_count=NUM_CLUSTERS,
    cluster_every=1,
    stop_event=None,
    loss_callback=None,
    cluster_callback=None,
    centroids_callback=None,
    model_lock=None,
    training_mode=TRAINING_MODE_GLOBAL,
    status_callback=None,
    train_microbatch_size=8,
    initial_centroids=None,
    dead_clusters_callback=None,
    dead_cluster_preview_callback=None,
    cluster_epoch_stats_callback=None,
    cluster_sampling_callback=None,
    cluster_consistency_weight=STAGE1_CLUSTER_CONSISTENCY_WEIGHT,
    negative_weight=STAGE1_NEGATIVE_WEIGHT,
    auto_k=False,
    auto_k_callback=None,
    epoch_end_callback=None,
    cluster_model=None,
    hard_mining=False,
    hard_mining_step=0.10,
    hard_mining_reset=0.70,
    enable_clustering=True,
    pause_event=None,
):
    model.train()
    lock_context = model_lock if model_lock is not None else nullcontext()
    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    global_step = 0

    cluster_assignments = None
    cluster_sampling = None
    cluster_stats = None
    prev_centroids = initial_centroids
    cluster_every = max(1, int(cluster_every))
    keep_frozen_backbone_eval(model)

    # External cheap cluster model — DISABLED for training.
    # The structured-negative loss compares the main model's q_global against the
    # negative bank (cluster_sampling["embedding_lookup"]); those negatives MUST be
    # in the main model's embedding space for the gradient to be meaningful. A CNN
    # cluster model lives in a different space (and often a different dim → the
    # 512-vs-1024 crash), so it cannot feed that bank, and the main model has to
    # embed every pair for the bank regardless — leaving no speedup to gain.
    # Fall back to backbone clustering and warn instead of silently training a
    # wrong (cross-space) negative loss.
    cluster_model_cache = None
    if cluster_model is not None:
        msg = ("Cluster model is incompatible with the structured-negative "
               "training loss (negative bank must be in the main model's "
               "embedding space). Falling back to backbone clustering for "
               "this run. The cluster model remains usable for eval-side "
               "gallery clustering.")
        if status_callback is not None:
            status_callback("WARNING: " + msg)
        else:
            print("WARNING: " + msg, flush=True)
        cluster_model = None

    for epoch in range(1, epochs + 1):
        # Clustering disabled: pure pairwise training (InfoNCE + variance only).
        # No K-means pass, no cluster-consistency loss, no structured negatives —
        # also skips the per-epoch embedding extraction, so epochs run faster.
        if not enable_clustering:
            need_cluster = False
            if epoch == 1 and status_callback is not None:
                status_callback(
                    "Clustering DISABLED: training with InfoNCE + variance only "
                    "(no K-means, no consistency loss, no structured hard negatives).")
        # With a frozen external cluster model, embeddings never change, so cluster
        # ONCE on the first epoch and reuse forever (cluster_every is ignored).
        elif cluster_model is not None:
            need_cluster = cluster_assignments is None
        elif cluster_assignments is None:
            need_cluster = True
        else:
            # Adaptive: if the last clustering's positive-anchor same-cluster
            # probability is below threshold, the embeddings are still drifting —
            # recluster every epoch. Otherwise fall back to the cluster_every cadence.
            last_pos_same = (cluster_sampling or {}).get(
                "positive_same_cluster_probability", 1.0)
            if last_pos_same < RECLUSTER_POS_SAME_PROB:
                need_cluster = True
                if status_callback is not None:
                    status_callback(
                        f"Epoch {epoch}: pos-same-cluster {last_pos_same:.3f} < "
                        f"{RECLUSTER_POS_SAME_PROB} → reclustering this epoch.")
            else:
                need_cluster = (epoch - 1) % cluster_every == 0
        if need_cluster:
            if status_callback is not None:
                status_callback(f"Epoch {epoch}: extracting embeddings and building clusters...")
            cluster_assignments, cluster_stats, cluster_sampling = compute_embedding_cluster_stats(
                model, loader, device, cluster_count, amp_enabled, stop_event, model_lock,
                return_assignments=True, prev_centroids=prev_centroids,
                auto_k=auto_k, auto_k_callback=auto_k_callback, status_callback=status_callback,
                cluster_model=cluster_model, cluster_model_cache=cluster_model_cache,
            )
            if cluster_stats is not None and cluster_stats.get("auto_k_resolved") is not None:
                cluster_count = cluster_stats["auto_k_resolved"]
        elif enable_clustering:
            if status_callback is not None:
                status_callback(f"Epoch {epoch}: reusing clusters from previous epoch.")

        if cluster_sampling is not None:
            prev_centroids = cluster_sampling.get("centroids")
        if cluster_callback is not None and cluster_stats is not None:
            cluster_callback(cluster_stats, epoch)
        if centroids_callback is not None and prev_centroids is not None:
            centroids_callback(prev_centroids)
        epoch_dead_info_for_callback = None
        if dead_cluster_preview_callback is not None and cluster_sampling is not None:
            frozen_state = cluster_sampling.get("_frozen_cluster_state")
            if frozen_state is not None:
                epoch_dead_info_for_callback = detect_dead_clusters(
                    frozen_state["assignments"],
                    frozen_state["embedding_array"],
                    frozen_state["sample_indices"],
                    cluster_count,
                    positive_assignments=frozen_state.get("positive_assignments"),
                )
        if enable_clustering and cluster_assignments is None:
            return False
        if cluster_sampling_callback is not None and cluster_sampling is not None:
            cluster_sampling_callback(cluster_sampling)

        positive_same_cluster_probability = (cluster_sampling or {}).get(
            "positive_same_cluster_probability", 0.0)
        _ptopk = (cluster_sampling or {}).get("positive_topk_probability", {})
        _ptopk_str = "".join(f" P@{k}={v:.3f}" for k, v in sorted(_ptopk.items()))
        # Hard-mining mask context: the same-cluster rate above is measured on
        # whichever subset is CURRENTLY active (dataset.samples) — at partial
        # masking that's a smaller, curated (historically-hardest) population,
        # not the full dataset, so it is not directly comparable epoch-to-epoch
        # unless you also know the mask fraction. Tag it explicitly.
        _hm_frac = getattr(loader.dataset, "_hm_last_frac", None)
        _hm_just_reset = getattr(loader.dataset, "_hm_just_reset", False)
        _hm_str = f", mask={_hm_frac:.0%}" if _hm_frac is not None else ""
        if status_callback is not None:
            status_callback(
                f"Epoch {epoch}: {'clusters ready; ' if enable_clustering else ''}"
                f"training with mode={training_mode}, "
                f"negatives_per_query={NEGATIVES_PER_QUERY if enable_clustering else 0}, "
                f"anchor-positive same-cluster={positive_same_cluster_probability:.4f}"
                f"{_ptopk_str}{_hm_str}, "
                f"alignment_weight={STAGE1_ALIGNMENT_WEIGHT:.1f}, "
                f"variance_weight={STAGE1_VARIANCE_WEIGHT:.1f}, "
                f"negative_weight={negative_weight:.1f}, "
                f"cluster_consistency_weight={cluster_consistency_weight:.1f}, "
                f"negative_mix={negative_sampling_mix() if enable_clustering else 'disabled'}."
            )
            if _hm_just_reset:
                # Distinctly tagged so these lines can be grepped out to see the
                # trend across mask cycles: rising = the hard-mining curriculum
                # is genuinely improving alignment; flat/falling = it may be
                # overfitting to whatever's currently "hard" at the cost of
                # overall calibration. Only these full-population (mask=0%)
                # measurements are directly comparable to each other.
                status_callback(
                    f"[Hard-mining RESET checkpoint] Epoch {epoch}: "
                    f"anchor-positive same-cluster={positive_same_cluster_probability:.4f} "
                    f"on FULL population (100% active, just reset from prior mask cycle)."
                )
        model.train()
        keep_frozen_backbone_eval(model)
        per_cluster_pos_prob_sum = {}
        per_cluster_pos_prob_count = {}
        per_cluster_loss_sum = {}
        per_cluster_loss_count = {}
        epoch_easiness = {}   # orig_idx → positive similarity (higher = easier)

        epoch_start_time = time.perf_counter()
        for batch_step, batch in enumerate(loader, start=1):
            if stop_event is not None and stop_event.is_set():
                return False
            if pause_event is not None and pause_event.is_set():
                if status_callback is not None:
                    status_callback(f"Paused at epoch {epoch} batch {batch_step} "
                                    f"(GPU idle; press Pause again to resume).")
                while pause_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        return False
                    time.sleep(0.5)
                if status_callback is not None:
                    status_callback("Resumed.")
            q, p, idx, batch_weights = unpack_batch(batch, batch_step, loader.batch_size)

            non_blocking = device.type == "cuda"
            loop_centroids = None
            if cluster_sampling is not None:
                _c = cluster_sampling.get("centroids")
                if _c is not None:
                    loop_centroids = _c.to(device, non_blocking=non_blocking)
            if enable_clustering and cluster_sampling is not None:
                negative_indices, _ = sample_structured_negative_indices(
                    idx.long().cpu(),
                    cluster_assignments,
                    cluster_sampling,
                    negatives_per_query=NEGATIVES_PER_QUERY,
                )
            else:
                # Clustering disabled: no structured negative bank.
                negative_indices = torch.zeros((idx.shape[0], 0), dtype=torch.long)

            with lock_context:
                optimizer.zero_grad()
                batch_size = q.shape[0]
                microbatch_size = max(1, min(int(train_microbatch_size), batch_size))
                loss_value = 0.0
                metrics_accumulator = {}
                for start in range(0, batch_size, microbatch_size):
                    end = min(start + microbatch_size, batch_size)
                    weight = (end - start) / batch_size
                    q_chunk = q[start:end].to(device, non_blocking=non_blocking)
                    p_chunk = p[start:end].to(device, non_blocking=non_blocking)
                    idx_chunk = idx[start:end].long()
                    # IoU weights from GtaUavDataset (None for SatCropDataset)
                    weights_chunk = (
                        batch_weights[start:end].to(device, non_blocking=non_blocking)
                        if batch_weights is not None else None
                    )
                    cluster_ids = (
                        cluster_assignments[idx_chunk].to(device, non_blocking=non_blocking)
                        if cluster_assignments is not None
                        else torch.zeros(idx_chunk.shape[0], dtype=torch.long, device=device))
                    negative_indices_chunk = negative_indices[start:end]
                    if cluster_sampling is not None and negative_indices_chunk.numel():
                        flat_neg_global_bank = lookup_negative_global_embeddings(
                            cluster_sampling["embedding_lookup"],
                            negative_indices_chunk.reshape(-1),
                        ).to(device, non_blocking=non_blocking)
                    else:
                        flat_neg_global_bank = torch.zeros(0, device=device)
                    q_neg_global = neg_global = None

                    with torch.amp.autocast("cuda", enabled=amp_enabled):
                        q_feats = model.encode_features(q_chunk)
                        q_global = model.encode_cluster_from_features(q_feats)
                        p_feats = model.encode_features(p_chunk)
                        p_global = model.encode_cluster_from_features(p_feats)
                        variance_loss, descriptor_std = variance_regularization(q_global, p_global)
                        negative_loss = q_global.new_tensor(0.0)
                        negative_similarity = q_global.new_tensor(0.0)
                        per_sample_neg = None   # per-sample hardest-negative sim (for hard-mining)
                        # Alignment loss: GroupInfoNCE whole_slice when group_size>1,
                        # WeightedInfoNCE with optional IoU soft labels otherwise.
                        _chunk_group = STAGE1_GROUP_SIZE if q_chunk.shape[0] % STAGE1_GROUP_SIZE == 0 else 1
                        if _chunk_group > 1:
                            alignment_loss = group_whole_slice_info_nce(
                                q_global, p_global,
                                group_len=_chunk_group,
                                label_smoothing=STAGE1_LABEL_SMOOTHING,
                                tau=STAGE1_ALIGNMENT_TAU,
                            )
                        else:
                            alignment_loss = weighted_info_nce(
                                q_global, p_global,
                                tau=STAGE1_ALIGNMENT_TAU,
                                label_smoothing=STAGE1_LABEL_SMOOTHING,
                                positive_weights=weights_chunk,
                            )
                        if flat_neg_global_bank.numel():
                            neg_global = flat_neg_global_bank.reshape(
                                q_global.shape[0],
                                NEGATIVES_PER_QUERY,
                                -1,
                            )
                            q_negative_similarities = torch.sum(q_global.unsqueeze(1) * neg_global, dim=-1)
                            p_negative_similarities = torch.sum(p_global.unsqueeze(1) * neg_global, dim=-1)
                            negative_similarities = 0.5 * (
                                q_negative_similarities + p_negative_similarities
                            )
                            per_sample_neg = negative_similarities.max(dim=1).values.detach()
                            negative_similarity = negative_similarities.max(dim=1).values.mean()
                            negative_loss = torch.relu(
                                negative_similarities - STAGE1_NEGATIVE_MARGIN
                            ).pow(2).mean()
                        cluster_consistency_loss = q_global.new_tensor(0.0)
                        if loop_centroids is not None:
                            q_logits = torch.matmul(q_global, loop_centroids.T) / STAGE1_CLUSTER_CONSISTENCY_TAU
                            p_logits = torch.matmul(p_global, loop_centroids.T) / STAGE1_CLUSTER_CONSISTENCY_TAU
                            q_log_soft = F.log_softmax(q_logits, dim=-1)
                            p_log_soft = F.log_softmax(p_logits, dim=-1)
                            cluster_consistency_loss = 0.5 * (
                                F.kl_div(q_log_soft, p_log_soft.exp().detach(), reduction="batchmean")
                                + F.kl_div(p_log_soft, q_log_soft.exp().detach(), reduction="batchmean")
                            )
                        neg_weight = (
                            min(1.0, batch_step / NEGATIVE_WARMUP_STEPS)
                            * negative_weight
                        )
                        loss = (
                            STAGE1_ALIGNMENT_WEIGHT * alignment_loss
                            + STAGE1_VARIANCE_WEIGHT * variance_loss
                            + neg_weight * negative_loss
                            + cluster_consistency_weight * cluster_consistency_loss
                        )

                        with torch.no_grad():
                            positive_similarity = torch.sum(q_global.detach() * p_global.detach(), dim=-1)
                            positive_probability = torch.sigmoid(
                                20.0 * (positive_similarity - negative_similarity.detach())
                            )
                            for _cid, _prob in zip(
                                cluster_ids.cpu().tolist(),
                                positive_probability.cpu().tolist(),
                            ):
                                per_cluster_pos_prob_sum[_cid] = per_cluster_pos_prob_sum.get(_cid, 0.0) + _prob
                                per_cluster_pos_prob_count[_cid] = per_cluster_pos_prob_count.get(_cid, 0) + 1
                            if hard_mining:
                                # Easiness = margin: positive sim minus the per-sample
                                # hardest-negative sim. High margin = easy (well matched
                                # AND well separated from negatives).
                                if per_sample_neg is not None:
                                    _easy = (positive_similarity - per_sample_neg)
                                else:
                                    _easy = positive_similarity
                                for _oi, _e in zip(idx_chunk.tolist(),
                                                   _easy.cpu().tolist()):
                                    epoch_easiness[int(_oi)] = float(_e)
                        metrics = {
                            "total_loss": loss.detach().item(),
                            "positive_similarity": positive_similarity.mean().item(),
                            "negative_similarity": negative_similarity.detach().item(),
                            "positive_score": positive_similarity.mean().item(),
                            "negative_score": negative_similarity.detach().item(),
                            "positive_probability": positive_probability.mean().item(),
                            "alignment_loss": alignment_loss.detach().item(),
                            "alignment_weight": STAGE1_ALIGNMENT_WEIGHT,
                            "variance_loss": variance_loss.detach().item(),
                            "variance_weight": STAGE1_VARIANCE_WEIGHT,
                            "negative_loss": negative_loss.detach().item(),
                            "negative_weight": neg_weight,
                            "negative_margin": STAGE1_NEGATIVE_MARGIN,
                            "cluster_consistency_loss": cluster_consistency_loss.detach().item(),
                            "cluster_consistency_weight": cluster_consistency_weight,
                            "descriptor_std": descriptor_std.detach().item(),
                            "descriptor_dim": float(q_global.shape[-1]),
                            "anchor_positive_same_cluster": positive_same_cluster_probability,
                        }

                    scaler.scale(loss * weight).backward()
                    loss_value += float(loss.detach().item()) * weight
                    _loss_val = float(loss.detach().item())
                    for _cid in cluster_ids.cpu().tolist():
                        per_cluster_loss_sum[_cid] = per_cluster_loss_sum.get(_cid, 0.0) + _loss_val
                        per_cluster_loss_count[_cid] = per_cluster_loss_count.get(_cid, 0) + 1
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + float(value) * weight
                    del q_chunk, p_chunk, q_feats, p_feats, q_global, p_global, q_neg_global, neg_global, flat_neg_global_bank
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


                scaler.step(optimizer)
                scaler.update()
                metrics = metrics_accumulator
                metrics["training_mode"] = training_mode
                metrics["microbatch_size"] = microbatch_size

            if device.type == "cuda":
                print(f"[MEMDEBUG] step {batch_step} allocated={torch.cuda.memory_allocated()/1e9:.3f}GB "
                      f"reserved={torch.cuda.memory_reserved()/1e9:.3f}GB "
                      f"max_allocated={torch.cuda.max_memory_allocated()/1e9:.3f}GB", flush=True)

            global_step += 1
            metrics["total_loss"] = loss_value
            if loss_callback is not None:
                loss_callback(metrics, global_step, epoch, batch_step)
            print(f"Epoch {epoch} Step {batch_step} Loss:", loss_value)

        epoch_elapsed = time.perf_counter() - epoch_start_time
        avg_batch_time = epoch_elapsed / batch_step if batch_step else 0.0
        if status_callback is not None:
            status_callback(
                f"Epoch {epoch}: {epoch_elapsed:.1f}s total, "
                f"avg batch time={avg_batch_time:.2f}s over {batch_step} batches."
            )

        cluster_training_loss = {
            cid: per_cluster_loss_sum[cid] / per_cluster_loss_count[cid]
            for cid in per_cluster_loss_sum
            if per_cluster_loss_count.get(cid, 0) > 0
        }
        if dead_cluster_preview_callback is not None and epoch_dead_info_for_callback is not None:
            cluster_training_pos_prob = {
                cid: per_cluster_pos_prob_sum[cid] / per_cluster_pos_prob_count[cid]
                for cid in per_cluster_pos_prob_sum
                if per_cluster_pos_prob_count.get(cid, 0) > 0
            }
            epoch_dead_info_for_callback["cluster_training_pos_prob"] = cluster_training_pos_prob
            epoch_dead_info_for_callback["cluster_training_loss"] = cluster_training_loss
            dead_cluster_preview_callback(epoch_dead_info_for_callback, epoch)
        if cluster_epoch_stats_callback is not None:
            cluster_sample_indices = {}
            if cluster_sampling is not None:
                for cid, members in cluster_sampling.get("cluster_members", {}).items():
                    step = max(1, len(members) // 16)
                    cluster_sample_indices[int(cid)] = members[::step][:16]
            cluster_epoch_stats_callback(
                {
                    "cluster_loss": cluster_training_loss,
                    "stage": 1,
                    "cluster_sample_indices": cluster_sample_indices,
                },
                epoch,
            )

        # Curriculum hard-mining: after each epoch, mask the easiest positives so
        # the model focuses on hard cases. When too much is masked, reset to 100%
        # so easy cases are not forgotten.
        if hard_mining and epoch_easiness and hasattr(loader.dataset, "apply_hard_mining_mask"):
            loader.dataset.apply_hard_mining_mask(
                epoch_easiness, mask_step=hard_mining_step,
                mask_reset=hard_mining_reset,
                log_fn=(status_callback or print))
            # Active set changed → assignments stale; force recluster next epoch.
            cluster_assignments = None

        if epoch_end_callback is not None:
            epoch_end_callback(epoch, model)

    if not enable_clustering:
        # Clustering disabled: nothing to refresh.
        final_stats = None
        final_sampling = None
        if status_callback is not None:
            status_callback("Training finished (clustering disabled — no centroid refresh).")
    elif stop_event is None or not stop_event.is_set():
        if status_callback is not None:
            status_callback(
                "Training finished: refreshing K-means centroids with final cluster-head weights "
                "before checkpoint save."
            )
        _, final_stats, final_sampling = compute_embedding_cluster_stats(
            model,
            loader,
            device,
            cluster_count,
            amp_enabled,
            stop_event,
            model_lock,
            return_assignments=True,
            prev_centroids=prev_centroids,
        )
    else:
        # User stopped training; skip expensive final clustering
        final_stats = None
        final_sampling = None
        if final_sampling is not None:
            prev_centroids = final_sampling.get("centroids")
            frozen_state = final_sampling.get("_frozen_cluster_state")
            if frozen_state is not None:
                dead_info = detect_dead_clusters(
                    frozen_state["assignments"],
                    frozen_state["embedding_array"],
                    frozen_state["sample_indices"],
                    cluster_count,
                    positive_assignments=frozen_state.get("positive_assignments"),
                )
                if final_stats is not None:
                    final_stats["dead_cluster_ids"] = dead_info["dead_cluster_ids"]
                    final_stats["dead_sample_indices"] = dead_info["dead_sample_indices"]
                    final_stats["dead_cluster_count"] = len(dead_info["dead_cluster_ids"])
                if dead_clusters_callback is not None:
                    dead_clusters_callback(dead_info)
                if status_callback is not None:
                    n_dead = len(dead_info["dead_cluster_ids"])
                    n_excl = len(dead_info["dead_sample_indices"])
                    status_callback(
                        f"Dead clusters: {n_dead}/{cluster_count} clusters flagged "
                        f"(per-dim variance < 10% of cluster average); "
                        f"{n_excl} samples in dead clusters."
                    )
            if centroids_callback is not None and prev_centroids is not None:
                centroids_callback(prev_centroids)
            if cluster_callback is not None and final_stats is not None:
                cluster_callback(final_stats, epochs)
            if status_callback is not None and final_stats is not None:
                status_callback(
                    "Final centroid refresh complete; "
                    f"anchor-positive same-cluster={final_stats['positive_same_cluster_probability']:.4f}."
                )

    return True


def keep_frozen_backbone_eval(model):
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        return
    if not any(param.requires_grad for param in backbone.parameters()):
        backbone.eval()
        return
    # Partial unfreeze: set each immediate child of backbone to eval unless any
    # of its params are trainable (those stay in train mode for correct BN stats).
    for child in backbone.children():
        if not any(p.requires_grad for p in child.parameters()):
            child.eval()


def variance_regularization(q_global, p_global):
    descriptors = torch.cat([q_global, p_global], dim=0).float()
    target_std = descriptors.new_tensor(1.0 / (descriptors.shape[1] ** 0.5))
    per_dim_std = torch.sqrt(descriptors.var(dim=0, unbiased=False) + 1e-4)
    variance_loss = torch.relu(target_std - per_dim_std).mean()
    return variance_loss, per_dim_std.mean()




def unpack_batch(batch, batch_step, batch_size):
    """Return (q, p, idx, weights_or_None).

    Handles 4-tuples (q, p, weight, idx) from GtaUavDataset,
    3-tuples (q, p, idx) from SatCropDataset, and 2-tuples (q, p).
    When group_size > 1 the dataset returns q/p with shape [B, G, C, H, W];
    these are flattened to [B*G, C, H, W].  idx and weights are repeated G times.
    """
    weights = None
    if len(batch) == 4:
        q, p, weights, idx = batch
    elif len(batch) == 3:
        q, p, idx = batch
    else:
        q, p = batch
        start = (batch_step - 1) * batch_size
        idx = torch.arange(start, start + q.shape[0], dtype=torch.long)

    if q.dim() == 5:
        B, G, C, H, W = q.shape
        q = q.view(B * G, C, H, W)
        p = p.view(B * G, C, H, W)
        idx = idx.repeat_interleave(G)
        if weights is not None:
            weights = weights.repeat_interleave(G)

    return q, p, idx, weights


def compute_embedding_cluster_stats(
    model,
    loader,
    device,
    cluster_count,
    amp_enabled,
    stop_event,
    model_lock=None,
    return_assignments=False,
    prev_centroids=None,
    frozen_cluster_state=None,
    use_fixed_centroids=False,
    auto_k=False,
    auto_k_callback=None,
    status_callback=None,
    cluster_model=None,
    cluster_model_cache=None,
):
    """Extract embeddings, run K-means, and build sampling data.

    cluster_model: optional frozen SimpleClusterCNN. When provided, ALL clustering
        / negative-sampling embeddings come from this cheap model instead of the
        heavy main backbone. Because it is frozen and clustering uses the
        deterministic eval transform, embeddings are cached in cluster_model_cache
        (idx → (q_emb, p_emb)) and reused across epochs — after epoch 1 the
        clustering pass does zero forward passes.
    """
    lock_context = model_lock if model_lock is not None else nullcontext()
    non_blocking = device.type == "cuda"
    was_training = model.training
    from dataset import loader_kwargs
    # persistent=False: workers pickle the dataset at __iter__, which must happen
    # AFTER the eval-transform override below (and re-capture it every pass).
    #
    # num_workers capped at the MAIN loader's own count (not left to silently
    # default via loader_kwargs(num_workers=None) -> default_num_workers()):
    # this cluster-rebuild pass runs at the epoch boundary, right as the main
    # loader's just-finished-epoch workers are (asynchronously) tearing down —
    # spinning up a SEPARATE, independently-sized batch of new workers here
    # transiently doubles worker-process count exactly when it's most likely to
    # overlap with not-yet-reaped old ones. Confirmed via a live remote OOM:
    # host RAM + swap both crashed to zero within ~2 minutes of an epoch-3
    # cluster rebuild, process silently SIGKILL'd (no traceback possible),
    # instant full memory recovery the moment it died — classic OOM-killer
    # signature. Capping here bounds the worst case instead of letting this
    # loader's worker count float independently (main=8 could combine with a
    # SEPARATE default_num_workers()=7 here, e.g. on a 28-thread box, for 15
    # peak workers from a single mistaken assumption).
    cluster_num_workers = min(getattr(loader, "num_workers", 0), 4)
    cluster_loader = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        **loader_kwargs(device.type, num_workers=cluster_num_workers, persistent=False),
    )

    model.eval()
    original_transform = getattr(loader.dataset, "transform", None)
    original_drone_transform = getattr(loader.dataset, "drone_transform", None)
    transform_was_overridden = hasattr(loader.dataset, "transform")
    if transform_was_overridden:
        if cluster_model is not None:
            # Use the cluster CNN's own input resolution for its eval transform.
            stat_img_size = getattr(cluster_model, "in_size", 224)
        else:
            from model import backbone_img_size
            stat_img_size = backbone_img_size(getattr(model, "backbone_name", "swin_t"))
        cluster_stat_transform = make_eval_transform(img_size=stat_img_size)
        loader.dataset.transform = cluster_stat_transform
        if original_drone_transform is not None:
            loader.dataset.drone_transform = cluster_stat_transform

    try:
        if frozen_cluster_state is not None:
            embedding_array = frozen_cluster_state["embedding_array"]
            positive_embedding_array = frozen_cluster_state["positive_embedding_array"]
            sample_indices = frozen_cluster_state["sample_indices"]
            assignments = frozen_cluster_state["assignments"]
            positive_assignments = frozen_cluster_state["positive_assignments"]
            centroids_norm = frozen_cluster_state["centroids_norm"]
            actual_clusters = frozen_cluster_state["actual_clusters"]
            _si = sample_indices
            _si_list = _si.tolist() if isinstance(_si, torch.Tensor) else list(_si)
            dataset_idx_to_cluster = {int(si): int(assignments[i]) for i, si in enumerate(_si_list)}

            retrieval_embeddings = []
            with torch.inference_mode():
                for batch_step, batch in enumerate(cluster_loader, start=1):
                    if stop_event is not None and stop_event.is_set():
                        if was_training:
                            model.train()
                        return (None, None, None) if return_assignments else None
                    q, p, idx, _ = unpack_batch(batch, batch_step, cluster_loader.batch_size)
                    q = q.to(device, non_blocking=non_blocking)
                    batch_cluster_ids = torch.tensor(
                        [dataset_idx_to_cluster.get(int(i.item()), 0) for i in idx],
                        dtype=torch.long, device=device,
                    )
                    with lock_context:
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            q_features = model.encode_features(q)
                            batch_retrieval = model.encode_retrieval_from_features(q_features, batch_cluster_ids)
                    retrieval_embeddings.append(batch_retrieval.float().cpu())

            if not retrieval_embeddings:
                return (None, None, None) if return_assignments else None
            retrieval_embedding_tensor = torch.cat(retrieval_embeddings, dim=0)
            positive_embeddings_for_sampling = torch.from_numpy(
                frozen_cluster_state["positive_embedding_array"]
            )

        elif cluster_model is not None:
            # ── Cheap external cluster model: embed q & p with the frozen CNN. ──
            # Frozen + deterministic eval transform → cache embeddings across epochs
            # (cluster_model_cache: idx → (q_emb, p_emb)). After epoch 1 this loop
            # does zero forward passes and just reads the cache.
            cache = cluster_model_cache if cluster_model_cache is not None else {}
            embeddings = []
            positive_embeddings_list = []
            indices = []
            cluster_model.eval()
            with torch.inference_mode():
                for batch_step, batch in enumerate(cluster_loader, start=1):
                    if stop_event is not None and stop_event.is_set():
                        if was_training:
                            model.train()
                        return (None, None, None) if return_assignments else None
                    q, p, idx, _ = unpack_batch(batch, batch_step, cluster_loader.batch_size)
                    idx_list = idx.long().tolist()
                    need = [j for j, ii in enumerate(idx_list) if ii not in cache]
                    if need:
                        qd = q.to(device, non_blocking=non_blocking)
                        pd = p.to(device, non_blocking=non_blocking)
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            qe = cluster_model(qd).float().cpu()
                            pe = cluster_model(pd).float().cpu()
                        for j in need:
                            cache[idx_list[j]] = (qe[j].clone(), pe[j].clone())
                    embeddings.append(torch.stack([cache[ii][0] for ii in idx_list]))
                    positive_embeddings_list.append(torch.stack([cache[ii][1] for ii in idx_list]))
                    indices.append(idx.long().cpu())

            if not embeddings:
                return (None, None, None) if return_assignments else None
            embedding_array = torch.cat(embeddings, dim=0).numpy().astype("float32")
            positive_embedding_array = torch.cat(positive_embeddings_list, dim=0).numpy().astype("float32")
            # Retrieval (hard-negative ranking) embeddings = the same cheap cluster embeddings.
            retrieval_embedding_tensor = torch.from_numpy(embedding_array)
            positive_embeddings_for_sampling = torch.from_numpy(positive_embedding_array)
            sample_indices = torch.cat(indices, dim=0)
            sample_count, dim = embedding_array.shape
            actual_clusters = min(cluster_count, sample_count)

            if use_fixed_centroids and prev_centroids is not None:
                centroids_norm = prev_centroids.cpu()
                centroids_np = centroids_norm.numpy().astype("float32")
                actual_clusters = centroids_np.shape[0]
                assignments = assign_fixed_centroids(embedding_array, centroids_np)
                positive_assignments = assign_fixed_centroids(positive_embedding_array, centroids_np)
            elif prev_centroids is not None:
                init_centroids = prev_centroids.cpu().numpy().astype("float32")
                cluster_index = ClusterIndex(n_clusters=actual_clusters, dim=dim, nredo=1, niter=5)
                cluster_index.fit(embedding_array, init_centroids=init_centroids)
                assignments = cluster_index.assign(embedding_array)
                positive_assignments = cluster_index.assign(positive_embedding_array)
                centroids_raw = torch.from_numpy(cluster_index.kmeans.centroids).float()
                centroids_norm = centroids_raw / centroids_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            else:
                cluster_index = ClusterIndex(n_clusters=actual_clusters, dim=dim, nredo=5)
                cluster_index.fit(embedding_array)
                assignments = cluster_index.assign(embedding_array)
                positive_assignments = cluster_index.assign(positive_embedding_array)
                centroids_raw = torch.from_numpy(cluster_index.kmeans.centroids).float()
                centroids_norm = centroids_raw / centroids_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            frozen_cluster_state = {
                "embedding_array": embedding_array,
                "positive_embedding_array": positive_embedding_array,
                "sample_indices": sample_indices,
                "assignments": assignments,
                "positive_assignments": positive_assignments,
                "centroids_norm": centroids_norm,
                "actual_clusters": actual_clusters,
            }

        else:
            # Full extraction: cluster head (q + p), shared backbone for retrieval (q only).
            embeddings = []
            positive_embeddings_list = []
            retrieval_embeddings = []
            indices = []
            prev_centroids_device = prev_centroids.to(device) if prev_centroids is not None else None

            with torch.inference_mode():
                for batch_step, batch in enumerate(cluster_loader, start=1):
                    if stop_event is not None and stop_event.is_set():
                        if was_training:
                            model.train()
                        return (None, None, None) if return_assignments else None
                    q, p, idx, _ = unpack_batch(batch, batch_step, cluster_loader.batch_size)
                    q = q.to(device, non_blocking=non_blocking)
                    p = p.to(device, non_blocking=non_blocking)
                    with lock_context:
                        with torch.amp.autocast("cuda", enabled=amp_enabled):
                            # Run backbone once for q; share features for cluster and retrieval.
                            q_features = model.encode_features(q)
                            batch_embeddings = model.encode_cluster_from_features(q_features)
                            if prev_centroids_device is not None:
                                hard_cluster_ids = torch.matmul(batch_embeddings, prev_centroids_device.T).argmax(dim=-1)
                            else:
                                hard_cluster_ids = torch.zeros(q_features.shape[0], dtype=torch.long, device=q_features.device)
                            batch_retrieval = model.encode_retrieval_from_features(q_features, hard_cluster_ids)
                            # Run backbone once for p; share features for cluster and retrieval.
                            p_features = model.encode_features(p)
                            positive_embeddings = model.encode_cluster_from_features(p_features)
                    embeddings.append(batch_embeddings.float().cpu())
                    positive_embeddings_list.append(positive_embeddings.float().cpu())
                    retrieval_embeddings.append(batch_retrieval.float().cpu())
                    indices.append(idx.long().cpu())

            if not embeddings:
                return (None, None, None) if return_assignments else None

            embedding_array = torch.cat(embeddings, dim=0).numpy().astype("float32")
            positive_embedding_array = torch.cat(positive_embeddings_list, dim=0).numpy().astype("float32")
            retrieval_embedding_tensor = torch.cat(retrieval_embeddings, dim=0)
            positive_embeddings_for_sampling = torch.from_numpy(positive_embedding_array)
            sample_indices = torch.cat(indices, dim=0)
            sample_count, dim = embedding_array.shape
            actual_clusters = min(cluster_count, sample_count)

            if auto_k and not use_fixed_centroids and prev_centroids is None:
                best_k, _ = find_optimal_k_silhouette(
                    embedding_array, status_callback=status_callback
                )
                if best_k is not None:
                    cluster_count = best_k
                    actual_clusters = min(cluster_count, sample_count)
                    if auto_k_callback is not None:
                        auto_k_callback(best_k)

            if use_fixed_centroids and prev_centroids is not None:
                centroids_norm = prev_centroids.cpu()
                centroids_np = centroids_norm.numpy().astype("float32")
                actual_clusters = centroids_np.shape[0]
                assignments = assign_fixed_centroids(embedding_array, centroids_np)
                positive_assignments = assign_fixed_centroids(positive_embedding_array, centroids_np)
            elif prev_centroids is not None:
                # Warm-start K-means from previous centroids (subsequent epochs).
                # niter=5: few iterations from the previous solution keeps cluster
                # assignments stable (avoids sudden reassignment spikes in the loss)
                # while still tracking embedding drift as training progresses.
                init_centroids = prev_centroids.cpu().numpy().astype("float32")
                cluster_index = ClusterIndex(n_clusters=actual_clusters, dim=dim, nredo=1, niter=5)
                cluster_index.fit(embedding_array, init_centroids=init_centroids)
                assignments = cluster_index.assign(embedding_array)
                positive_assignments = cluster_index.assign(positive_embedding_array)
                centroids_raw = torch.from_numpy(cluster_index.kmeans.centroids).float()
                centroids_norm = centroids_raw / centroids_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            else:
                cluster_index = ClusterIndex(n_clusters=actual_clusters, dim=dim, nredo=5)
                cluster_index.fit(embedding_array)
                assignments = cluster_index.assign(embedding_array)
                positive_assignments = cluster_index.assign(positive_embedding_array)
                centroids_raw = torch.from_numpy(cluster_index.kmeans.centroids).float()
                centroids_norm = centroids_raw / centroids_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

            frozen_cluster_state = {
                "embedding_array": embedding_array,
                "positive_embedding_array": positive_embedding_array,
                "sample_indices": sample_indices,
                "assignments": assignments,
                "positive_assignments": positive_assignments,
                "centroids_norm": centroids_norm,
                "actual_clusters": actual_clusters,
            }

    finally:
        if transform_was_overridden:
            loader.dataset.transform = original_transform
            if original_drone_transform is not None:
                loader.dataset.drone_transform = original_drone_transform

    if was_training:
        model.train()

    sample_count, dim = embedding_array.shape
    positive_same_cluster_probability = float((positive_assignments == assignments).mean())
    # Top-k containment: P(positive's cluster ∈ anchor's k nearest centroids).
    # This — not top-1 — is the recall ceiling for top-Kc routed evaluation.
    positive_topk_probability = {}
    if centroids_norm is not None:
        _emb = torch.from_numpy(embedding_array).float()
        _emb = _emb / _emb.norm(dim=1, keepdim=True).clamp(min=1e-6)
        _sims = _emb @ centroids_norm.T                       # [N, K]
        _pos = torch.from_numpy(positive_assignments).long().unsqueeze(1)
        for _k in (2, 3):
            kk = min(_k, _sims.shape[1])
            _top = _sims.topk(kk, dim=1).indices
            positive_topk_probability[_k] = float((_top == _pos).any(dim=1).float().mean())
    non_empty_clusters = len(set(assignments.tolist()))
    # Size by max orig_idx (not len(dataset)): under hard-mining the active set is
    # a sparse subset, so orig_idx can exceed len(loader.dataset).
    _n_slots = len(loader.dataset)
    if len(sample_indices):
        _n_slots = max(_n_slots, int(sample_indices.max().item()) + 1)
    dataset_assignments = torch.zeros(_n_slots, dtype=torch.long)
    dataset_assignments[sample_indices] = torch.from_numpy(assignments).long()
    cluster_sampling = build_cluster_sampling_data(
        torch.from_numpy(embedding_array),
        sample_indices,
        torch.from_numpy(assignments).long(),
        actual_clusters,
        retrieval_embedding_tensor,
        centroids_norm,
        positive_embeddings_for_sampling,
    )
    cluster_sampling["positive_same_cluster_probability"] = positive_same_cluster_probability
    cluster_sampling["positive_topk_probability"] = positive_topk_probability
    cluster_sampling["_frozen_cluster_state"] = frozen_cluster_state

    # MES: build identity maps for false-negative exclusion in negative sampling.
    # pair_to_pos_tile:    orig_idx  → satellite tile filename (str)
    # pos_tile_to_indices: tile filename → frozenset of orig_idx that share that tile
    # pair_to_drone:       orig_idx  → drone image filename (str)
    # drone_to_indices:    drone filename → frozenset of orig_idx sharing that anchor
    #   The drone map ensures that when batch item is (anchor_A, strong_pos), all other
    #   pairs with anchor_A (its semi-positives, augmented positives, etc.) are also
    #   excluded from the hard-negative pool — preventing false negatives.
    pair_to_pos_tile = {}
    pos_tile_to_indices = {}
    pair_to_drone = {}
    drone_to_indices = {}
    dataset_pairs = getattr(loader.dataset, "pairs", None)
    if dataset_pairs is not None:
        for orig_idx in sample_indices.tolist():
            i = int(orig_idx)
            if i < len(dataset_pairs):
                # Key by the FULL path, NOT parent_dir/name: DenseUAV reuses the
                # same filenames (H80.tif, H80.JPG) in every location folder, so
                # bare-name keys aliased ALL locations into one exclusion bucket
                # (exclude_set = entire dataset → no negatives, self-as-negative
                # fallback, and minute-long batches) — parent_dir/name was an
                # earlier fix for that, but it silently assumed exactly one
                # directory level between dataset root and image file. SUES-200
                # has an EXTRA altitude level (drone_view_512/<loc>/<alt>/N.jpg),
                # so parent.name resolved to the ALTITUDE, not the location —
                # colliding every location's image #N at a given altitude into
                # one bucket, then cascading through the co-positive-tile
                # expansion below to sweep in the entire dataset as "excluded"
                # (confirmed: real run showed excluded=24000/24000 samples).
                # The full path has no such depth assumption and is unique
                # regardless of how many directory levels a dataset uses.
                tile_path = Path(dataset_pairs[i][1])
                tile_name = str(tile_path)
                pair_to_pos_tile[i] = tile_name
                pos_tile_to_indices.setdefault(tile_name, set()).add(i)
                drone_path_ = Path(dataset_pairs[i][0])
                drone_name = str(drone_path_)
                pair_to_drone[i] = drone_name
                drone_to_indices.setdefault(drone_name, set()).add(i)
    # Convert to frozensets for safe sharing
    pos_tile_to_indices = {k: frozenset(v) for k, v in pos_tile_to_indices.items()}
    drone_to_indices    = {k: frozenset(v) for k, v in drone_to_indices.items()}
    # Content-hash map: catches tiles that are byte-identical under DIFFERENT
    # paths (pos_tile_to_indices above only catches same-PATH tiles). Confirmed
    # real-world case: University-1652's official release duplicates one
    # satellite image across two different building-ID folders (e.g. 0761 and
    # 0768) -- without this, those two IDs could be sampled as "hard negatives"
    # of each other despite being pixel-identical, a false negative that no
    # amount of training can resolve. Hashing is cached by path (see
    # _tile_content_hash), so this costs one-time I/O per unique tile, not
    # per-epoch, after the first cluster rebuild.
    pair_to_content_hash = {}
    content_hash_to_indices = {}
    for idx, tile_name in pair_to_pos_tile.items():
        h = _tile_content_hash(tile_name)
        if h is not None:
            pair_to_content_hash[idx] = h
            content_hash_to_indices.setdefault(h, set()).add(idx)
    content_hash_to_indices = {k: frozenset(v) for k, v in content_hash_to_indices.items()
                               if len(v) > 1}
    # Spatial coordinate map: (seq, zoom, col, row) → frozenset of pair indices.
    # Tile names follow {seq}_{zoom}_{col}_{row}.png (e.g. 04_7_009_020.png).
    # Used in the exclusion loop to block adjacent tiles (Chebyshev ≤ 2) from
    # becoming hard negatives even when they are not explicitly in the positive set.
    tile_coord_to_indices = {}
    for idx, tile_name in pair_to_pos_tile.items():
        parts = Path(tile_name).stem.split("_")
        if len(parts) >= 4:
            try:
                r = int(parts[-1]); c = int(parts[-2]); z = int(parts[-3])
                s = "_".join(parts[:-3])
                # Normalize twin-pass pair IDs: dataset_gen.py produces consecutive
                # even/odd pairs from the same TIF run (tif1→gallery and tif2→gallery).
                # They tile the same geographic grid, so (col, row) addresses match.
                # Map both to the even ID so their tiles share one exclusion bucket.
                if len(s) == 4 and s.isdigit():
                    s = str((int(s) // 2) * 2).zfill(4)
                tile_coord_to_indices.setdefault((s, z, c, r), set()).add(idx)
            except ValueError:
                pass
    tile_coord_to_indices = {k: frozenset(v) for k, v in tile_coord_to_indices.items()}
    cluster_sampling["pair_to_pos_tile"]      = pair_to_pos_tile
    cluster_sampling["pos_tile_to_indices"]   = pos_tile_to_indices
    cluster_sampling["pair_to_drone"]         = pair_to_drone
    cluster_sampling["drone_to_indices"]      = drone_to_indices
    cluster_sampling["tile_coord_to_indices"] = tile_coord_to_indices
    cluster_sampling["content_hash_to_indices"] = content_hash_to_indices
    cluster_sampling["pair_to_content_hash"]    = pair_to_content_hash
    raw_gps = getattr(loader.dataset, "pair_gps", {})
    cluster_sampling["pair_to_gps"] = {i: raw_gps[i] for i in sample_indices.tolist()
                                        if i in raw_gps}

    stats = {
        "target_clusters": cluster_count,
        "actual_clusters": actual_clusters,
        "non_empty_clusters": non_empty_clusters,
        "sample_count": sample_count,
        "embedding_dim": dim,
        "positive_same_cluster_probability": positive_same_cluster_probability,
        "kmeans_centroids": centroids_norm,
        "auto_k_resolved": cluster_count if auto_k else None,
    }
    if return_assignments:
        return dataset_assignments, stats, cluster_sampling
    return stats


def detect_dead_clusters(assignments, embeddings, sample_indices, cluster_count, positive_assignments=None):
    """Dead cluster: per-dimension descriptor variance < 0.1 × mean variance across all clusters."""
    if not isinstance(assignments, torch.Tensor):
        assignments = torch.from_numpy(assignments).long()
    else:
        assignments = assignments.long()
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.from_numpy(embeddings).float()
    else:
        embeddings = embeddings.float()
    if positive_assignments is not None:
        if not isinstance(positive_assignments, torch.Tensor):
            positive_assignments = torch.from_numpy(positive_assignments).long()
        else:
            positive_assignments = positive_assignments.long()
    sample_indices_list = (
        sample_indices.long().tolist()
        if isinstance(sample_indices, torch.Tensor)
        else [int(i) for i in sample_indices]
    )
    cluster_variances = {}
    for cid in range(cluster_count):
        mask = assignments == cid
        if not mask.any():
            continue
        members = embeddings[mask]
        mean_var = float(members.var(dim=0, unbiased=False).mean().item()) if members.shape[0] > 1 else 0.0
        cluster_variances[cid] = mean_var
    if not cluster_variances:
        return {
            "dead_cluster_ids": set(),
            "dead_sample_indices": [],
            "cluster_variances": {},
            "variance_threshold": 0.0,
            "global_mean_var": 0.0,
            "cluster_positive_probability": {},
        }
    global_mean_var = sum(cluster_variances.values()) / len(cluster_variances)
    variance_threshold = 0.1 * global_mean_var
    dead_cluster_ids = set()
    dead_sample_indices = []
    cluster_sample_indices = {}
    cluster_positive_probability = {}
    for cid, mean_var in cluster_variances.items():
        mask = assignments == cid
        rows = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        cluster_sample_indices[cid] = [sample_indices_list[r] for r in rows]
        if positive_assignments is not None:
            pos_assigns = positive_assignments[mask]
            cluster_positive_probability[cid] = float((pos_assigns == cid).float().mean().item())
        if mean_var < variance_threshold:
            dead_cluster_ids.add(cid)
            dead_sample_indices.extend(cluster_sample_indices[cid])
    return {
        "dead_cluster_ids": dead_cluster_ids,
        "dead_sample_indices": dead_sample_indices,
        "cluster_variances": cluster_variances,
        "variance_threshold": variance_threshold,
        "global_mean_var": global_mean_var,
        "cluster_sample_indices": cluster_sample_indices,
        "cluster_positive_probability": cluster_positive_probability,
    }


def build_cluster_sampling_data(embeddings, sample_indices, assignments, cluster_count, retrieval_embeddings=None, kmeans_centroids=None, positive_embeddings=None):
    if retrieval_embeddings is None:
        retrieval_embeddings = embeddings
    cluster_members = {cluster_id: [] for cluster_id in range(cluster_count)}
    centroids = torch.zeros(cluster_count, embeddings.shape[1], dtype=embeddings.dtype)
    for cluster_id in range(cluster_count):
        mask = assignments == cluster_id
        if mask.any():
            cluster_members[cluster_id] = sample_indices[mask].long().tolist()
            centroids[cluster_id] = embeddings[mask].mean(dim=0)
    centroids = torch.nn.functional.normalize(centroids, dim=1)
    if kmeans_centroids is not None:
        centroids = kmeans_centroids
    centroid_similarities = torch.matmul(centroids, centroids.T)
    nearest_clusters = {}
    for cluster_id in range(cluster_count):
        order = torch.argsort(centroid_similarities[cluster_id], descending=True).tolist()
        nearest_clusters[cluster_id] = [other for other in order if other != cluster_id][:2]
    positive_embedding_lookup = {}
    if positive_embeddings is not None:
        positive_embedding_lookup = {
            int(idx): positive_embeddings[row].clone()
            for row, idx in enumerate(sample_indices.tolist())
        }
    retrieval_centroids = torch.zeros(cluster_count, retrieval_embeddings.shape[1], dtype=torch.float32)
    for cluster_id in range(cluster_count):
        mask = assignments == cluster_id
        if mask.any():
            retrieval_centroids[cluster_id] = retrieval_embeddings[mask].float().mean(dim=0)
    retrieval_centroids = torch.nn.functional.normalize(retrieval_centroids, dim=1)
    return {
        "cluster_members": cluster_members,
        "nearest_clusters": nearest_clusters,
        "all_indices": sample_indices.long().tolist(),
        "embedding_lookup": {int(idx): embeddings[row].clone() for row, idx in enumerate(sample_indices.tolist())},
        "retrieval_embedding_lookup": {
            int(idx): retrieval_embeddings[row].clone()
            for row, idx in enumerate(sample_indices.tolist())
        },
        "centroids": centroids,
        "positive_embedding_lookup": positive_embedding_lookup,
        "retrieval_centroids": retrieval_centroids,
    }


MES_PROX_EXCL_M = 30.0


def compute_mes_exclude_set(query_idx, cluster_sampling, cluster_pool=None):
    """Minimum-Effective-Similarity exclusion set for a query's hard-negative pool.

    Returns the set of pair indices that must NOT be used as hard negatives for
    `query_idx` because they are really (semi-)positives or share the same scene:
      • self
      • same positive satellite tile (by path, or by content hash -- catches
        tiles that are byte-identical under different paths, e.g. duplicate
        satellite images across two building IDs in University-1652's
        official release)
      • spatially adjacent / twin-pass tiles (Chebyshev ≤ 2, pair-ID normalized so
        even/odd twin passes share one grid bucket — catches the exact-position
        cross-sensor twin tile)
      • same anchor drone image (its other semi/augmented positives)
      • any candidate whose anchor GPS is within MES_PROX_EXCL_M metres

    Used by BOTH training (sample_structured_negative_indices) and the GUI
    hard-negative preview, so the two never diverge. cluster_pool limits the GPS
    scan; if None it is taken from cluster_sampling for the query's cluster.
    """
    pair_to_tile       = cluster_sampling.get("pair_to_pos_tile", {})
    tile_to_pairs      = cluster_sampling.get("pos_tile_to_indices", {})
    tile_coord_map     = cluster_sampling.get("tile_coord_to_indices", {})
    pair_to_drone_map  = cluster_sampling.get("pair_to_drone", {})
    drone_to_pairs_map = cluster_sampling.get("drone_to_indices", {})
    pair_to_gps        = cluster_sampling.get("pair_to_gps", {})
    pair_to_content_hash    = cluster_sampling.get("pair_to_content_hash", {})
    content_hash_to_pairs   = cluster_sampling.get("content_hash_to_indices", {})

    qi = int(query_idx)
    exclude_set = {qi}
    my_tile = pair_to_tile.get(qi)
    if my_tile:
        exclude_set |= tile_to_pairs.get(my_tile, set())
        my_hash = pair_to_content_hash.get(qi)
        if my_hash:
            exclude_set |= content_hash_to_pairs.get(my_hash, frozenset())
        if tile_coord_map:
            parts = Path(my_tile).stem.split("_")
            if len(parts) >= 4:
                try:
                    r = int(parts[-1]); c = int(parts[-2]); z = int(parts[-3])
                    s = "_".join(parts[:-3])
                    if len(s) == 4 and s.isdigit():
                        s = str((int(s) // 2) * 2).zfill(4)
                    for dc in range(-2, 3):
                        for dr in range(-2, 3):
                            exclude_set |= tile_coord_map.get((s, z, c + dc, r + dr), frozenset())
                except ValueError:
                    pass
    my_drone = pair_to_drone_map.get(qi)
    if my_drone:
        sibling_pairs = drone_to_pairs_map.get(my_drone, frozenset())
        exclude_set |= sibling_pairs
        # Co-positive exclusion: if the same anchor has multiple positive tiles
        # (e.g. multi-zoom GTA-UAV-LR), another anchor sharing any of those tiles
        # is geographically adjacent and must not be used as a hard negative.
        for sib_idx in sibling_pairs:
            sib_tile = pair_to_tile.get(int(sib_idx))
            if sib_tile:
                exclude_set |= tile_to_pairs.get(sib_tile, frozenset())

    anchor_gps = pair_to_gps.get(qi)
    if anchor_gps and pair_to_gps:
        if cluster_pool is None:
            ca = cluster_sampling.get("dataset_assignments")
            cm = cluster_sampling.get("cluster_members", {})
            cluster_pool = cm.get(int(ca[qi]), []) if ca is not None else []
        alat, alon = anchor_gps
        cos_lat = math.cos(math.radians(alat))
        # Vectorized GPS proximity (numpy) — the old per-candidate Python loop was a
        # training bottleneck on large epoch-1 clusters (py-spy: 80 s/batch stalls).
        lat_arr, lon_arr = _gps_arrays(cluster_sampling)
        if lat_arr is not None and len(cluster_pool):
            pool_arr = np.asarray(cluster_pool, dtype=np.int64)
            pool_arr = pool_arr[pool_arr < lat_arr.shape[0]]
            dlat = np.abs(lat_arr[pool_arr] - alat) * 111_320.0
            dlon = np.abs(lon_arr[pool_arr] - alon) * 111_320.0 * cos_lat
            near = (dlat < MES_PROX_EXCL_M) & (dlon < MES_PROX_EXCL_M)
            if near.any():
                d = np.hypot(dlat[near], dlon[near])
                for x in pool_arr[near][d < MES_PROX_EXCL_M]:
                    exclude_set.add(int(x))
    return exclude_set


def _np_cache(cluster_sampling):
    """Per-clustering-pass cache (lives inside the cluster_sampling dict, so it
    invalidates automatically whenever clusters are rebuilt)."""
    c = cluster_sampling.get("_np_cache")
    if c is None:
        c = {"pool": {}, "gps": "unset"}
        cluster_sampling["_np_cache"] = c
    return c


def _gps_arrays(cluster_sampling):
    """pair_to_gps as dense numpy arrays indexed by pair idx (NaN = no GPS)."""
    cache = _np_cache(cluster_sampling)
    if cache["gps"] == "unset":
        p2g = cluster_sampling.get("pair_to_gps") or {}
        if p2g:
            n = max(int(k) for k in p2g.keys()) + 1
            lat = np.full(n, np.nan, dtype=np.float64)
            lon = np.full(n, np.nan, dtype=np.float64)
            for k, (la, lo) in p2g.items():
                lat[int(k)] = la
                lon[int(k)] = lo
            cache["gps"] = (lat, lon)
        else:
            cache["gps"] = (None, None)
    return cache["gps"]


def _cluster_pool_arrays(cluster_sampling, cluster_id):
    """(idx array [N], embedding matrix [N,D]) for one cluster — built ONCE per
    clustering pass instead of re-stacked per query (the main 80 s/batch culprit)."""
    cache = _np_cache(cluster_sampling)
    key = ("c", int(cluster_id))
    if key not in cache["pool"]:
        emb = cluster_sampling["embedding_lookup"]
        pool = cluster_sampling["cluster_members"].get(int(cluster_id), [])
        idx = [int(i) for i in pool if int(i) in emb]
        if idx:
            mat = torch.stack([emb[i] for i in idx], dim=0).float()
            cache["pool"][key] = (np.asarray(idx, dtype=np.int64), mat)
        else:
            cache["pool"][key] = (np.empty(0, dtype=np.int64), torch.zeros(0, 1))
    return cache["pool"][key]


def _multi_cluster_pool_arrays(cluster_sampling, cluster_ids):
    """Concatenated pool arrays for several clusters (e.g. 3 nearest of a query).

    NOT cached by combination: cluster_ids is per-query (each query's own
    nearest-K clusters), so with K clusters there are up to C(K,3) distinct
    combinations. Caching those in the shared unbounded `_np_cache["pool"]`
    dict (as before) leaked without bound over an epoch -- confirmed via a
    remote RAM watcher: single-process RSS reached 33.5GB and the kernel
    OOM-killed the run within ~19 batches on GTA-UAV's 123k-pair, K=16
    setting. The underlying per-cluster arrays are still cheaply cached
    (bounded to K entries, see _cluster_pool_arrays), so concatenating them
    fresh here is just an index/tensor copy, not a similarity recompute.
    """
    parts = [_cluster_pool_arrays(cluster_sampling, c) for c in cluster_ids]
    parts = [(i, m) for i, m in parts if i.size]
    if parts:
        idx = np.concatenate([p[0] for p in parts])
        mat = torch.cat([p[1] for p in parts], dim=0)
        return idx, mat
    return np.empty(0, dtype=np.int64), torch.zeros(0, 1)


def _all_pool_arrays(cluster_sampling):
    """Full-bank pool arrays (every sample), cached. Used by the fallback when a
    query's cluster pools cannot fill the negative quota after exclusion. Built
    once per clustering pass — the previous per-query torch.stack over the whole
    bank allocated ~0.5 GB per fallback query and OOM-killed large-dataset runs
    (GTA-UAV, 123k pairs) on both Linux and Windows."""
    cache = _np_cache(cluster_sampling)
    key = ("all",)
    if key not in cache["pool"]:
        emb = cluster_sampling["embedding_lookup"]
        idx = [int(i) for i in cluster_sampling["all_indices"] if int(i) in emb]
        if idx:
            mat = torch.stack([emb[i] for i in idx], dim=0).float()
            cache["pool"][key] = (np.asarray(idx, dtype=np.int64), mat)
        else:
            cache["pool"][key] = (np.empty(0, dtype=np.int64), torch.zeros(0, 1))
    return cache["pool"][key]


def _sample_hard_cached(cluster_sampling, idx_arr, emb_mat, count, exclude_set, query_idx):
    """Vectorized equivalent of sample_hard_from_pool: rank the (cached) pool by
    cosine similarity to the query, keep top count*FACTOR, sample count of them."""
    if count <= 0 or idx_arr.size == 0:
        return []
    q = cluster_sampling["embedding_lookup"].get(int(query_idx))
    mask = np.ones(idx_arr.shape[0], dtype=bool)
    if exclude_set:
        mask = ~np.isin(idx_arr, np.fromiter(exclude_set, dtype=np.int64,
                                             count=len(exclude_set)))
    n_avail = int(mask.sum())
    if n_avail == 0:
        return []
    if q is None:
        pick = np.random.choice(np.nonzero(mask)[0], size=count, replace=n_avail < count)
        return idx_arr[pick].tolist()
    sims = torch.mv(emb_mat, q.float()).numpy().copy()
    sims[~mask] = -np.inf
    pool_size = min(count * HARD_NEGATIVE_POOL_FACTOR, n_avail)
    top = np.argpartition(-sims, pool_size - 1)[:pool_size]
    pick = np.random.permutation(top.shape[0])[:count]
    hard = idx_arr[top[pick]].tolist()
    if len(hard) < count:
        extra = np.random.choice(np.nonzero(mask)[0], size=count - len(hard), replace=True)
        hard.extend(idx_arr[extra].tolist())
    return hard[:count]


def sample_structured_negative_indices(
    query_indices,
    cluster_assignments,
    cluster_sampling,
    negatives_per_query=NEGATIVES_PER_QUERY,
):
    primary_ratio, _ = negative_sampling_ratios()
    primary_count = max(1, round(negatives_per_query * primary_ratio))
    global_count = negatives_per_query - primary_count

    rows = []
    temperature_rows = []
    all_indices = cluster_sampling["all_indices"]

    for query_idx in query_indices.tolist():
        cluster_id = int(cluster_assignments[query_idx])
        cluster_pool = cluster_sampling["cluster_members"].get(cluster_id, [])
        selected = []
        temperatures = []

        # MES exclusion (shared with the GUI hard-negative preview): self, same
        # tile, twin/adjacent tiles, same drone, and GPS-proximate pairs.
        exclude_set = compute_mes_exclude_set(query_idx, cluster_sampling, cluster_pool)

        # Primary negatives from the query's own cluster — cached pool matrix,
        # single matmul per query (was: torch.stack over the whole pool per query).
        p_idx, p_mat = _cluster_pool_arrays(cluster_sampling, cluster_id)
        primary_selected = _sample_hard_cached(
            cluster_sampling, p_idx, p_mat, primary_count, exclude_set, int(query_idx))
        selected.extend(primary_selected)
        temperatures.extend([PRIMARY_NEGATIVE_TAU] * len(primary_selected))

        # Sample from nearest clusters (other high-similarity clusters)
        if global_count > 0:
            centroids = cluster_sampling.get("centroids")
            embeddings = cluster_sampling.get("embedding_lookup")
            if centroids is not None and embeddings is not None and int(query_idx) in embeddings:
                query_emb = torch.as_tensor(embeddings[int(query_idx)], dtype=torch.float32)
                centroid_sims = torch.nn.functional.cosine_similarity(
                    query_emb.unsqueeze(0), centroids, dim=1
                )
                nearest_k = min(3, len(centroids) - 1)  # 3 nearest clusters for cross-cluster negs
                _, nearest_cluster_ids = torch.topk(centroid_sims, nearest_k + 1)
                nearest_cluster_ids = [
                    int(cid) for cid in nearest_cluster_ids if int(cid) != cluster_id
                ][:nearest_k]

                n_idx, n_mat = _multi_cluster_pool_arrays(cluster_sampling, nearest_cluster_ids)
                if n_idx.size:
                    global_selected = _sample_hard_cached(
                        cluster_sampling, n_idx, n_mat, global_count, exclude_set, int(query_idx))
                else:
                    global_selected = sample_from_pool(all_indices, global_count, exclude=exclude_set)
            else:
                global_selected = sample_from_pool(all_indices, global_count, exclude=exclude_set)
        else:
            global_selected = []

        selected.extend(global_selected)
        temperatures.extend([GLOBAL_NEGATIVE_TAU] * len(global_selected))

        if len(selected) < negatives_per_query:
            # First fallback: sample from all clusters (cross-cluster) with hard
            # mining, via the cached full-bank matrix — never re-stacked per query.
            remaining = negatives_per_query - len(selected)
            a_idx, a_mat = _all_pool_arrays(cluster_sampling)
            if a_idx.size:
                fallback_selected = _sample_hard_cached(
                    cluster_sampling, a_idx, a_mat, remaining, exclude_set, int(query_idx))
                selected.extend(fallback_selected)
                temperatures.extend([GLOBAL_NEGATIVE_TAU] * len(fallback_selected))

        if len(selected) < negatives_per_query:
            # Second fallback: repeat from current selection (allows duplicates)
            # This ensures we always get exactly negatives_per_query negatives
            missing = negatives_per_query - len(selected)
            repeat_idx = 0
            while missing > 0:
                if selected:
                    selected.append(selected[repeat_idx % len(selected)])
                    temperatures.append(temperatures[repeat_idx % len(temperatures)])
                    repeat_idx += 1
                    missing -= 1
                else:
                    # Fallback of last resort: use query itself
                    selected.append(query_idx)
                    temperatures.append(GLOBAL_NEGATIVE_TAU)
                    missing -= 1
        rows.append(selected[:negatives_per_query])
        temperature_rows.append(temperatures[:negatives_per_query])
    return torch.tensor(rows, dtype=torch.long), torch.tensor(temperature_rows, dtype=torch.float32)


def negative_sampling_ratios():
    return (STAGE1_PRIMARY_NEGATIVE_RATIO, STAGE1_GLOBAL_NEGATIVE_RATIO)


def negative_sampling_mix():
    primary_ratio, global_ratio = negative_sampling_ratios()
    return f"{primary_ratio:.0%} same cluster, {global_ratio:.0%} nearest clusters"


def sample_from_pool(pool, count, exclude=None):
    """exclude: int, set[int], or None."""
    if count <= 0:
        return []
    excl = exclude if isinstance(exclude, set) else ({int(exclude)} if exclude is not None else set())
    candidates = [int(idx) for idx in pool if int(idx) not in excl]
    if not candidates:
        return []
    candidate_tensor = torch.tensor(candidates, dtype=torch.long)
    if len(candidates) >= count:
        order = torch.randperm(len(candidates))[:count]
        return candidate_tensor[order].tolist()
    repeats = torch.randint(0, len(candidates), (count - len(candidates),), dtype=torch.long)
    return candidates + candidate_tensor[repeats].tolist()


HARD_NEGATIVE_POOL_FACTOR = 3

def sample_hard_from_pool(embedding_lookup, pool, count, exclude=None, query_idx=None):
    """
    exclude: int, set[int], or None — all filtered out of the candidate pool.
    query_idx: the single int whose embedding is used for similarity ranking.
               Defaults to exclude when exclude is a plain int (backward compat).
    """
    if count <= 0:
        return []
    excl = exclude if isinstance(exclude, set) else ({int(exclude)} if exclude is not None else set())
    _qidx = query_idx if query_idx is not None else (int(exclude) if isinstance(exclude, (int, float)) else None)
    candidates = [int(idx) for idx in pool if int(idx) not in excl and int(idx) in embedding_lookup]
    if not candidates:
        return []
    query_embedding = embedding_lookup.get(_qidx) if _qidx is not None else None
    if query_embedding is None:
        return sample_from_pool(candidates, count)
    candidate_embeddings = torch.stack([embedding_lookup[idx] for idx in candidates], dim=0)
    similarities = torch.matmul(candidate_embeddings, query_embedding)
    order = torch.argsort(similarities, descending=True)
    pool_size = min(count * HARD_NEGATIVE_POOL_FACTOR, len(candidates))
    top_pool = [candidates[int(pos)] for pos in order[:pool_size]]
    perm = torch.randperm(len(top_pool))[:count]
    hard = [top_pool[int(i)] for i in perm]
    if len(hard) < count:
        hard.extend(sample_from_pool(candidates, count - len(hard)))
    return hard[:count]


def lookup_negative_global_embeddings(embedding_lookup, negative_indices):
    embeddings = []
    for sample_idx in negative_indices.tolist():
        embeddings.append(embedding_lookup[int(sample_idx)])
    return torch.stack(embeddings, dim=0)



