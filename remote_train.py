#!/usr/bin/env python3
"""Headless training runner driven by the local training GUI over SSH.

The GUI uploads a training_params JSON, launches
    python3 -u remote_train.py --config <cfg.json> --run-dir <dir>
and parses one JSON object per stdout line:

    {"event": "status",  "message": str}
    {"event": "model_created", "model_size": str}
    {"event": "loss",    "metrics": {...}, "step": int, "epoch": int,
                         "batch_step": int}
    {"event": "cluster", "stats": {...}, "epoch": int}
    {"event": "checkpoint", "path": str, "epoch": int}
    {"event": "finished", "status": "finished"|"stopped"|"failed",
                          "error": str?}

Stop: the GUI touches <run-dir>/stop.flag; a watcher thread sets the same
stop_event trainer.train() already honors (stops after the current batch).

The checkpoint written here uses the exact dict layout of the GUI's
write_checkpoint(), so pulled checkpoints load in the local eval GUIs.
"""
import argparse
import hashlib
import json
import os
import threading
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import (SwinEmbedding, backbone_img_size,
                   CLUSTER_DESCRIPTOR_DIM)
from dataset import loader_kwargs
from trainer import train, even_batch_size
import trainer as trainer_mod


def emit(event, **kw):
    print(json.dumps({"event": event, **kw}, default=str), flush=True)


def status(message):
    emit("status", message=str(message))


def _json_safe(value):
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 2000:
        return [_json_safe(v) for v in value]
    if isinstance(value, dict) and len(value) <= 2000:
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, set) and len(value) <= 2000:
        return [_json_safe(v) for v in value]
    return None


def sanitize(d):
    return {k: _json_safe(v) for k, v in d.items() if _json_safe(v) is not None}


def backbone_stages(backbone):
    """Per-architecture stage container (mirrors model.py's hook logic):
    Swin/SwinV2 -> .layers, ConvNeXt -> .stages, ViT/DINOv2 -> .blocks.
    The old layers-only check made unfreeze a SILENT no-op on ConvNeXt —
    zero trainable params -> backward() crash on no-head variants."""
    if hasattr(backbone, "layers") and not hasattr(backbone, "blocks"):
        return list(backbone.layers)
    if hasattr(backbone, "stages"):
        return list(backbone.stages)
    if hasattr(backbone, "blocks"):
        return list(backbone.blocks)
    return []


def freeze_backbone(model, params):
    n_unfreeze = params.get("unfreeze_backbone_layers", 0)
    for p in model.backbone.parameters():
        p.requires_grad = False
    model.backbone.eval()
    stages = backbone_stages(model.backbone)
    if n_unfreeze > 0 and stages:
        num_stages = len(stages)
        tail = stages[max(0, num_stages - n_unfreeze):]
        if hasattr(model.backbone, "norm"):
            tail.append(model.backbone.norm)
        for mod in tail:
            mod.train()
            for p in mod.parameters():
                p.requires_grad = True
        unfrozen = sum(p.numel() for m in tail for p in m.parameters())
        status(f"Backbone: last {n_unfreeze} stage(s) unfrozen "
               f"({unfrozen/1e6:.1f}M trainable backbone params).")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    status(f"Total trainable params: {trainable/1e6:.1f}M")


def configure_trainable_heads(model):
    if getattr(model, "_use_head", True) and hasattr(model, "cluster_head"):
        for p in model.cluster_head.parameters():
            p.requires_grad = True


def build_dataset(params, img_size):
    ds_type = params.get("dataset_type", "")
    group_size = params.get("group_size", 1)
    augment = params.get("augment", True)

    def root(key):
        # The GUI sends roots like "~/Thinghiem/data/..." verbatim; the shell
        # never sees them (they go via config JSON), and pathlib treats "~" as
        # a literal directory — expand it here.
        return str(Path(params[key]).expanduser())

    if ds_type == "sues200":
        from dataset import Sues200Dataset
        return Sues200Dataset(
            train_root=root("sues200_train_root"), group_size=group_size,
            img_size=img_size, augment=augment)
    if ds_type == "denseuav":
        from dataset import DenseUAVDataset
        return DenseUAVDataset(
            train_root=root("denseuav_train_root"), group_size=group_size,
            img_size=img_size, augment=augment,
            cross_altitude=params.get("denseuav_cross_altitude", True),
            altitude_weight_tau=(None if params.get("denseuav_altitude_full_strength", False)
                                 else params.get("denseuav_altitude_weight_tau", 20.0)))
    if ds_type == "university1652":
        from dataset import University1652Dataset
        return University1652Dataset(
            train_root=root("u1652_train_root"), group_size=group_size,
            img_size=img_size, augment=augment)
    if ds_type == "game4loc":
        from dataset import GtaUavDataset
        return GtaUavDataset(
            data_root=root("gta_data_root"),
            pairs_meta_file=params.get("gta_json", ""),
            mode=params.get("gta_mode", "pos_semipos"),
            group_size=group_size,
            img_size=img_size,
            augment_positives=params.get("gta_augment_pos", False),
            augment=augment)
    raise ValueError(f"Remote training does not support dataset_type={ds_type!r} yet")


def filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    for key, value in state_dict.items():
        if key.startswith("cluster_heads."):
            continue
        target = (f"cluster_head.{key[len('global_head.'):]}"
                  if key.startswith("global_head.") else key)
        if target in model_state and model_state[target].shape == value.shape:
            compatible[target] = value
    return compatible


class Runner:
    def __init__(self, params, run_dir):
        self.params = params
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.losses = []
        self.loss_records = []
        self.kmeans_centroids = None
        self.model = None
        self.optimizer = None
        self.ckpt_path = self.run_dir / "checkpoint.pt"

    # ---- stop/pause flag watcher ------------------------------------------
    def watch_stop_flag(self):
        """stop.flag: stop after the current batch (existing behavior).
        pause.flag: in-place freeze at the next batch boundary while the flag
        exists; removing it resumes at the exact next batch — the epoch is NOT
        restarted (trainer.train just busy-waits inside the batch loop)."""
        stop_flag = self.run_dir / "stop.flag"
        pause_flag = self.run_dir / "pause.flag"
        while not self.stop_event.is_set():
            if stop_flag.exists():
                status("Stop flag detected; stopping after current batch...")
                self.stop_event.set()
                return
            paused_now = pause_flag.exists()
            if paused_now != self.pause_event.is_set():
                if paused_now:
                    self.pause_event.set()
                else:
                    self.pause_event.clear()
            time.sleep(2)

    # ---- callbacks --------------------------------------------------------
    def on_loss(self, metrics, step, epoch, batch_step):
        m = sanitize(metrics)
        loss = m.get("total_loss", 0.0)
        self.losses.append(loss)
        self.loss_records.append({"step": step, "epoch": epoch,
                                  "batch_step": batch_step, **m})
        emit("loss", metrics=m, step=step, epoch=epoch, batch_step=batch_step)

    def on_cluster(self, stats, epoch):
        emit("cluster", stats=sanitize(stats), epoch=epoch)

    def on_cluster_sampling(self, cluster_sampling):
        """The FULL cluster data (per-sample embeddings, cluster_members,
        embedding_lookup, MES exclusion maps — everything the local GUI's
        hard-negative/distinctiveness preview needs), unlike on_cluster's
        lightweight summary stats. Too large for the JSON-line stdout stream
        (~100MB+ for a full SUES-200/DenseUAV epoch's embeddings) — saved to
        disk instead (same atomic tmp+replace pattern as save_checkpoint) and
        pulled via scp by the local side, mirroring the checkpoint pull path.
        User explicitly opted into this extra per-epoch transfer cost."""
        path = self.run_dir / "cluster_sampling.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(cluster_sampling, tmp)
        tmp.replace(path)
        emit("cluster_sampling", path=str(path))

    def on_centroids(self, centroids):
        try:
            self.kmeans_centroids = centroids.detach().cpu()
        except AttributeError:
            self.kmeans_centroids = centroids

    def save_checkpoint(self, epoch=None):
        if self.model is None:
            return
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": (self.optimizer.state_dict()
                                     if self.optimizer else None),
            "losses": self.losses,
            "loss_records": self.loss_records,
            "dataset_root": self.params.get("dataset_root", ""),
            "training_params": self.params,
            "kmeans_centroids": self.kmeans_centroids,
            "retrieval_centroids": None,
            "last_training_mode": self.params.get("training_mode"),
        }
        tmp = self.ckpt_path.with_suffix(".tmp")
        torch.save(checkpoint, tmp)
        tmp.replace(self.ckpt_path)
        self._register_in_ckpt_cache()
        emit("checkpoint", path=str(self.ckpt_path), epoch=epoch)

    def _register_in_ckpt_cache(self):
        """Hardlink this checkpoint into ../ckpt_cache/<md5>.pt (relative to
        code_dir, our cwd — matches train.py's remote_train_loop upload path).

        Without this, the cache only ever gets populated on UPLOAD, never on
        PULL: checkpoint_path_for() is one fixed local file that gets
        overwritten every epoch by the auto-pull, so by the time a NEW run
        resumes from it, its content was never indexed under its own hash —
        even though it's bit-identical to what this exact server just wrote,
        forcing a pointless ~1GB re-upload of a file the server already has.
        Registering here closes that gap: a later resume from the pulled-
        unchanged file will find its hash already cached and skip re-upload
        entirely. Best-effort (upload just falls back to re-sending if this
        fails) and prunes to the 3 newest entries, same retention as the
        local upload path — otherwise every epoch's distinct checkpoint would
        accumulate here forever across a long run."""
        try:
            h = hashlib.md5()
            with open(self.ckpt_path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            cache_dir = Path("ckpt_cache")
            cache_dir.mkdir(exist_ok=True)
            cache_path = cache_dir / f"{h.hexdigest()}.pt"
            if not cache_path.exists():
                os.link(self.ckpt_path, cache_path)
            entries = sorted(cache_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime,
                             reverse=True)
            for stale in entries[3:]:
                stale.unlink(missing_ok=True)
        except Exception as exc:
            status(f"ckpt_cache registration skipped: {exc}")

    def on_epoch_end(self, epoch, model):
        self.save_checkpoint(epoch=epoch)

    # ---- resume -----------------------------------------------------------
    def maybe_resume(self, model, optimizer, device):
        if not self.params.get("resume_checkpoint"):
            status("Checkpoint resume disabled; starting from initialized weights.")
            return
        if not self.ckpt_path.exists():
            status(f"No checkpoint at {self.ckpt_path}; starting fresh.")
            return
        ckpt = torch.load(self.ckpt_path, map_location=device, weights_only=False)
        state = filter_compatible_state_dict(model, ckpt["model_state_dict"])
        if (self.params.get("freeze_backbone", True)
                and self.params.get("unfreeze_backbone_layers", 0) == 0):
            state = {k: v for k, v in state.items()
                     if not k.startswith("backbone.")}
        model.load_state_dict(state, strict=False)
        opt_state = ckpt.get("optimizer_state_dict")
        if (opt_state is not None
                and ckpt.get("last_training_mode") == self.params.get("training_mode")):
            try:
                optimizer.load_state_dict(opt_state)
                status("Loaded optimizer state from checkpoint.")
            except Exception as exc:
                status(f"Skipped incompatible optimizer state: {exc}")
        if ckpt.get("kmeans_centroids") is not None:
            self.kmeans_centroids = ckpt["kmeans_centroids"].cpu()
        self.losses = list(ckpt.get("losses", []))
        self.loss_records = list(ckpt.get("loss_records", []))
        status(f"Resumed from {self.ckpt_path} "
               f"({len(state)} tensors, {len(self.loss_records)} prior steps).")

    # ---- main -------------------------------------------------------------
    def run(self):
        params = self.params
        device = torch.device(params.get("device", "cuda"))
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        backbone = params.get("backbone", "swin_t")
        img_size = backbone_img_size(backbone)

        model = SwinEmbedding(
            embed_dim=CLUSTER_DESCRIPTOR_DIM,
            pretrained=params.get("pretrained", True),
            num_clusters=params.get("cluster_count", 16),
            backbone=backbone,
        ).to(device)
        self.model = model
        freeze_backbone(model, params)
        configure_trainable_heads(model)
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError(
                "No trainable parameters after freeze/unfreeze configuration "
                f"(backbone={params.get('backbone')}, unfreeze="
                f"{params.get('unfreeze_backbone_layers', 0)}). No-head "
                "backbones need unfreeze >= 1; if unfreeze IS set, the "
                "backbone's stage container wasn't recognized.")
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=params["learning_rate"], weight_decay=params["weight_decay"])
        self.optimizer = optimizer
        self.maybe_resume(model, optimizer, device)
        emit("model_created",
             model_size=f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

        dataset = build_dataset(params, img_size)
        status(f"Loaded {len(dataset)} pairs "
               f"(dataset_type={params.get('dataset_type')}, img_size={img_size}).")
        loader = DataLoader(
            dataset,
            batch_size=even_batch_size(len(dataset), params["batch_size"]),
            shuffle=params.get("shuffle", True),
            **loader_kwargs(device.type,
                            num_workers=params.get("num_workers", 8),
                            persistent=False))
        status(f"DataLoader workers: {loader.num_workers}")

        trainer_mod.STAGE1_LABEL_SMOOTHING = params.get("label_smoothing", 0.05)
        trainer_mod.STAGE1_GROUP_SIZE = params.get("group_size", 1)

        # The GUI creates a fresh run_dir per launch, so any checkpoint present
        # at startup is an UPLOADED fine-tune starting point, not a partially
        # completed remote run — its loss_records carry epoch history from
        # prior (possibly local, different-dataset) training and must NOT
        # shrink this run's epoch budget. (The old
        # `epochs - max(recorded epoch)` logic made a 40-epoch fine-tune from
        # a 40-epoch checkpoint run exactly 1 epoch.)
        epochs_done = max((r["epoch"] for r in self.loss_records), default=0)
        epochs_left = params["epochs"]
        if epochs_done:
            status(f"Starting checkpoint carries {epochs_done} prior recorded "
                   f"epoch(s) (fine-tune upload); running the full "
                   f"{epochs_left} epoch(s) of this run regardless.")

        if (self.kmeans_centroids is not None
                and self.kmeans_centroids.shape[0] != params.get("cluster_count", 16)):
            # Requested K changed vs the uploaded checkpoint: stale centroids
            # can't seed faiss K-means at a different K — recluster fresh.
            status(f"Saved centroids ({self.kmeans_centroids.shape[0]} clusters) "
                   f"!= requested cluster_count={params.get('cluster_count', 16)} — "
                   f"discarding; fresh clustering.")
            self.kmeans_centroids = None

        threading.Thread(target=self.watch_stop_flag, daemon=True).start()

        completed = train(
            model, loader, optimizer, device,
            epochs=epochs_left,
            use_amp=params.get("use_amp", True),
            cluster_count=params.get("cluster_count", 16),
            cluster_every=params.get("cluster_every", 1),
            stop_event=self.stop_event,
            loss_callback=self.on_loss,
            cluster_callback=self.on_cluster,
            cluster_sampling_callback=self.on_cluster_sampling,
            centroids_callback=self.on_centroids,
            training_mode=params.get("training_mode", "cluster_head_512"),
            status_callback=status,
            train_microbatch_size=params.get("microbatch_size", 24),
            initial_centroids=(None if params.get("auto_cluster_k")
                               else self.kmeans_centroids),
            cluster_consistency_weight=params.get("cluster_consistency_weight", 1.5),
            negative_weight=params.get("negative_weight", 10.0),
            auto_k=params.get("auto_cluster_k", False),
            epoch_end_callback=self.on_epoch_end,
            hard_mining=params.get("hard_mining", False),
            enable_clustering=params.get("enable_clustering", True),
            pause_event=self.pause_event,
        )
        self.save_checkpoint()
        self.stop_event.set()          # ends the stop-flag watcher
        return "finished" if completed else "stopped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    params = json.loads(Path(args.config).read_text())
    runner = Runner(params, args.run_dir)
    try:
        result = runner.run()
        emit("finished", status=result)
    except Exception as exc:
        emit("finished", status="failed", error=str(exc),
             traceback=traceback.format_exc())
        raise SystemExit(1)


if __name__ == "__main__":
    main()
