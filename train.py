import json
import re
import shutil
import subprocess
import sys
import random
import time
from datetime import datetime
from pathlib import Path
# Import torch before PyQt5 to avoid a Windows DLL initialization conflict.
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal, qInstallMessageHandler
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor
from PyQt5.QtWidgets import *
import threading
from model import CLUSTER_DESCRIPTOR_DIM, GLOBAL_DESCRIPTOR_DIM, NUM_CLUSTERS, BACKBONE_CONFIGS

DATASET_ROOT = r"D:\UAV_DATASET\PAIRS"
# Two independent profiles so training locally and remotely at the same time
# (two GUI windows) doesn't have one instance's save clobber the other's
# settings in a single shared file — each window keeps its own local vs.
# remote config, and toggling "Train on remote (SSH)" auto-saves whichever
# profile you're leaving and auto-loads the one you're switching to.
UI_SETTINGS_PATH = Path("training_ui_settings.json")
UI_SETTINGS_PATH_REMOTE = Path("training_ui_settings_remote.json")
# Suppress console window flashes when the GUI spawns ssh/scp on Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def qt_message_handler(*args):
    msg = args[2]
    if "Cannot queue arguments of type 'QVector<int>'" not in msg:
        sys.stderr.write(f"{msg}\n")


qInstallMessageHandler(qt_message_handler)

class App(QWidget):
    log_message = pyqtSignal(str)
    loss_recorded = pyqtSignal(dict, int, int, int)
    clusters_updated = pyqtSignal(dict, int)
    training_finished = pyqtSignal(str)
    model_created = pyqtSignal(dict)
    dead_cluster_preview_updated = pyqtSignal(dict, int)
    cluster_stats_updated = pyqtSignal(dict, int)
    cluster_sampling_ready = pyqtSignal(object)
    cluster_data_build_finished = pyqtSignal(str)
    auto_k_found = pyqtSignal(int)
    # Scale head tab signals
    sh_log_signal  = pyqtSignal(str)
    sh_loss_signal = pyqtSignal(int, float)   # epoch, loss
    sh_done_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAV Localization Trainer")
        self.resize(900, 650)

        self.stop_event = threading.Event()
        self.training_thread = None
        self.model = None
        self.optimizer = None
        self.model_lock = threading.Lock()
        self.losses = []
        self.loss_records = []
        self.training_params = {}
        self.kmeans_centroids = None
        self.retrieval_centroids = None
        self.cluster_image_paths = {}
        self._last_dead_cluster_ids = set()
        self.last_cluster_sampling = None
        self.excluded_pair_orig_idx: set = set()   # union of cluster-level + individual exclusions
        self._cluster_table_data: dict = {}         # cluster_id → {"size", "loss", "variance"}
        # Scale head tab state
        self.sh_thread: threading.Thread | None = None
        self.sh_stop_event = threading.Event()
        self.sh_losses: list[float] = []
        self.sh_epoch_boundaries: list[int] = []
        self.sh_current_epoch = 0

        self.log_message.connect(self.log_append)
        self.loss_recorded.connect(self.add_loss)
        self.clusters_updated.connect(self.update_cluster_status)
        self.training_finished.connect(self.on_training_finished)
        self.model_created.connect(self.on_model_created)
        self.dead_cluster_preview_updated.connect(self.on_dead_cluster_preview_updated)
        self.cluster_stats_updated.connect(self.on_cluster_stats_updated)
        self.cluster_sampling_ready.connect(self.on_cluster_sampling_ready)
        self.cluster_data_build_finished.connect(self.on_cluster_data_build_finished)
        self.auto_k_found.connect(self.on_auto_k_found)
        self.sh_log_signal.connect(self._sh_on_log)
        self.sh_loss_signal.connect(self._sh_on_loss)
        self.sh_done_signal.connect(self._sh_on_done)

        layout = QVBoxLayout()
        tabs = QTabWidget()
        self.tabs = tabs
        input_tab = QWidget()
        input_layout = QVBoxLayout()
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout()
        controls = QHBoxLayout()
        params = QGroupBox("Training Parameters")
        _params_left = QFormLayout()
        _params_right = QFormLayout()

        self.train_btn = QPushButton("Start Training")
        self.train_btn.clicked.connect(self.start_training)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setCheckable(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setToolTip(
            "Pause training at the next batch boundary (GPU idles, nothing is\n"
            "lost; press again to resume). While paused, remote checkpoint\n"
            "downloads are also skipped — the newest pending checkpoint is\n"
            "remembered and pulled on resume.")
        self.pause_btn.toggled.connect(self.toggle_pause)
        self.pause_event = threading.Event()

        self.reattach_btn = QPushButton("Reattach")
        self.reattach_btn.setEnabled(False)
        self.reattach_btn.setToolTip(
            "Resume GUI visibility into the LAST remote run this session\n"
            "launched, without touching it — no re-sync, no re-upload, no\n"
            "relaunch. Use this if a remote run gets reported 'failed' or\n"
            "'stopped' but you suspect it's actually still training fine on\n"
            "the server (e.g. the local launch/monitor SSH connection itself\n"
            "hung or died) — check the server directly first if unsure.")
        self.reattach_btn.clicked.connect(self.start_reattach_remote)
        self._last_remote_run_file = Path(__file__).parent / "last_remote_run.json"
        self._last_remote_host, self._last_remote_run_dir = self._load_last_remote_run()
        if self._last_remote_run_dir:
            self.reattach_btn.setEnabled(True)

        self.save_btn = QPushButton("Save Model")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_model)

        self.preview_aug_btn = QPushButton("Preview Positive Augmentations")
        self.preview_aug_btn.clicked.connect(self.preview_positive_augmentations)

        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 1000)
        self.epochs_input.setValue(1)

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 256)
        self.batch_size_input.setValue(64)

        self.microbatch_size_input = QSpinBox()
        self.microbatch_size_input.setRange(1, 64)
        self.microbatch_size_input.setValue(8)
        self.microbatch_size_input.setToolTip(
            "CUDA chunk size used inside each batch. Lower this if training runs out of GPU memory."
        )

        self.lr_input = QDoubleSpinBox()
        self.lr_input.setDecimals(7)
        self.lr_input.setRange(0.0000001, 1.0)
        self.lr_input.setSingleStep(0.0001)
        self.lr_input.setValue(0.0001)

        self.weight_decay_input = QDoubleSpinBox()
        self.weight_decay_input.setDecimals(7)
        self.weight_decay_input.setRange(0.0, 1.0)
        self.weight_decay_input.setSingleStep(0.0001)
        self.weight_decay_input.setValue(0.01)

        self.device_input = QComboBox()
        self.device_input.addItem("CPU", "cpu")
        if torch.cuda.is_available():
            self.device_input.addItem(f"CUDA - {torch.cuda.get_device_name(0)}", "cuda")
            self.device_input.setCurrentIndex(1)
        else:
            self.device_input.setToolTip(f"CUDA unavailable in torch {torch.__version__}")

        self.num_workers_input = QSpinBox()
        self.num_workers_input.setRange(0, 16)
        self.num_workers_input.setValue(0)
        self.num_workers_input.setToolTip("Use 0 on Windows/PyQt to avoid DataLoader worker hangs.")

        self.amp_input = QCheckBox()
        self.amp_input.setChecked(torch.cuda.is_available())

        self.backbone_input = QComboBox()
        self.backbone_input.addItem("Swin-T  (28 M, 224×224) — default", "swin_t")
        self.backbone_input.addItem("Swin-B  (88 M, 384×384) — Game4Loc", "swin_b")
        self.backbone_input.addItem("Swin-B no head  (88 M, 384×384, 1024-D) — Game4Loc exact", "swin_b_nohead")
        self.backbone_input.addItem("SwinV2-B  (88 M, 384×384) — IN-22K→384", "swinv2_b")
        self.backbone_input.addItem("ConvNeXt-B  (89 M, 384×384, 1024-D) — Game4Loc exact", "convnext_b")
        self.backbone_input.addItem("ConvNeXt-T  (28 M, 384×384, 768-D) — small, same recipe", "convnext_t")
        self.backbone_input.addItem("ViT-B/16  (86 M, 224×224) — IN-21K", "vit_b")
        self.backbone_input.addItem("DINOv2-B/14  (86 M, 224×224) — LVD-142M", "dinov2_b")
        self.backbone_input.setToolTip(
            "Swin-T: swin_tiny_patch4_window7_224 · 28 M params · 224×224 · 512-D output.\n"
            "Swin-B: same backbone + Linear(1024→512)+GELU head · 384×384 · 512-D output.\n"
            "Swin-B no head: raw 1024-D L2-norm backbone output · matches Game4Loc exactly.\n"
            "SwinV2-B: swinv2_base_window12to24_192to384 (IN-22K→384) · 88 M · 384×384 · 1024-D → 512-D head.\n"
            "ConvNeXt-B: convnext_base.fb_in22k_ft_in1k_384 · 89 M · 384×384 · raw 1024-D L2-norm · matches Game4Loc exactly.\n"
            "ConvNeXt-T: convnext_tiny.fb_in22k_ft_in1k_384 · 28 M · 384×384 · raw 768-D L2-norm · same recipe, 3x smaller.\n"
            "ViT-B/16: vit_base_patch16_224 (augreg2 IN-21K) · 86 M · 224×224 · 768-D → 512-D head.\n"
            "DINOv2-B/14: vit_base_patch14_dinov2 (LVD-142M) · 86 M · 224×224 · 768-D → 512-D head.\n"
            "Switching backbone resets the model — disable 'Resume checkpoint' when changing."
        )

        self.pretrained_input = QCheckBox()
        self.pretrained_input.setChecked(True)
        self.pretrained_input.setToolTip(
            "Load ImageNet-pretrained backbone weights.\n"
            "Recommended: keep checked unless fine-tuning from a saved checkpoint."
        )

        self.unfreeze_backbone_layers_input = QSpinBox()
        self.unfreeze_backbone_layers_input.setRange(0, 4)
        self.unfreeze_backbone_layers_input.setValue(0)
        self.unfreeze_backbone_layers_input.setToolTip(
            "Number of backbone stages to unfreeze from the end.\n"
            "0 = all frozen (fastest, lowest risk of catastrophic forgetting).\n"
            "1 = unfreeze last stage (~2.4 M Swin-T / ~8.0 M Swin-B params).\n"
            "2 = also unfreeze second-to-last stage.\n"
            "Use a lower learning rate when unfreezing backbone layers."
        )

        self.resume_checkpoint_input = QCheckBox()
        self.resume_checkpoint_input.setChecked(True)
        self.resume_checkpoint_input.setToolTip(
            "Load the checkpoint below before training (or the default\n"
            "checkpoints/latest_{backbone}.pt if the field is left blank).\n"
            "Turn off for a clean run from pretrained weights."
        )

        self.checkpoint_path_input = QLineEdit()
        self.checkpoint_path_input.setPlaceholderText(
            "(default: checkpoints/latest_<backbone>.pt)")
        self.checkpoint_path_input.setToolTip(
            "Checkpoint file to resume from AND to save/autosave to during this run.\n"
            "Leave blank to use the backbone-derived default\n"
            "(checkpoints/latest_swin_t.pt, latest_swin_b.pt, etc).\n"
            "Set this to load/continue a specific checkpoint — e.g. a renamed\n"
            "backup, a run from 'checkpoints/olds/', or a checkpoint on another drive."
        )
        _ckpt_browse_btn = QPushButton("Browse...")
        _ckpt_browse_btn.clicked.connect(
            lambda: self._browse_file(self.checkpoint_path_input,
                                      "Checkpoint (*.pt *.pth);;All files (*.*)"))
        checkpoint_path_row = QWidget()
        _ckpt_row_layout = QHBoxLayout()
        _ckpt_row_layout.setContentsMargins(0, 0, 0, 0)
        _ckpt_row_layout.addWidget(self.checkpoint_path_input)
        _ckpt_row_layout.addWidget(_ckpt_browse_btn)
        checkpoint_path_row.setLayout(_ckpt_row_layout)

        self.shuffle_input = QCheckBox()
        self.shuffle_input.setChecked(True)

        self.augment_input = QCheckBox()
        self.augment_input.setChecked(True)
        self.augment_input.setToolTip(
            "Apply training-time image augmentation (JPEG, color jitter,\n"
            "blur/sharpen, 90° rotation, coarse dropout).\n"
            "Uncheck for a clean Resize→ToTensor→Normalize pipeline\n"
            "(same as eval) — useful for debugging or clean-signal training."
        )

        self.hard_mining_input = QCheckBox()
        self.hard_mining_input.setChecked(False)
        self.hard_mining_input.setToolTip(
            "Curriculum hard-example mining. After each epoch, mask the easiest 10%\n"
            "of positives (by margin = positive_sim − hardest_negative_sim) out of\n"
            "the next epoch, so the model focuses on hard cases. When >70% is masked,\n"
            "reset to 100% so easy cases aren't forgotten. Speeds up training and\n"
            "improves hard-case performance."
        )

        self.cluster_model_input = QLineEdit()
        self.cluster_model_input.setPlaceholderText("(none — cluster with main backbone)")
        self.cluster_model_input.setToolTip(
            "Optional SimpleClusterCNN checkpoint (from cluster_train_gui.py).\n"
            "When set, per-epoch clustering / negative-sampling embeddings come\n"
            "from this cheap frozen CNN instead of the heavy backbone — much\n"
            "faster, and cached across epochs (re-clustering becomes near-free).\n"
            "The cluster-consistency loss is auto-disabled (routing is externalized)."
        )
        self.cluster_model_browse = QPushButton("…")
        self.cluster_model_browse.setFixedWidth(28)
        self.cluster_model_browse.clicked.connect(self._browse_cluster_model)
        self.cluster_model_row = QWidget()
        _cmrl = QHBoxLayout(self.cluster_model_row)
        _cmrl.setContentsMargins(0, 0, 0, 0)
        _cmrl.addWidget(self.cluster_model_input)
        _cmrl.addWidget(self.cluster_model_browse)

        self.cluster_count_input = QSpinBox()
        self.cluster_count_input.setRange(1, 1024)
        self.cluster_count_input.setValue(32)
        self.cluster_count_input.setToolTip(
            "Number of K-means embedding clusters used during training."
        )

        self.auto_cluster_k_input = QCheckBox("Auto")
        self.auto_cluster_k_input.setToolTip(
            "Automatically find optimal K via silhouette score sweep before the first epoch.\n"
            "Tries K ∈ {4, 8, 12, 16, 24, 32, 48, 64} on a 2000-sample subsample."
        )
        self.auto_cluster_k_input.toggled.connect(
            lambda checked: self.cluster_count_input.setEnabled(not checked)
        )

        cluster_count_row = QWidget()
        cluster_count_layout = QHBoxLayout()
        cluster_count_layout.setContentsMargins(0, 0, 0, 0)
        cluster_count_layout.addWidget(self.cluster_count_input)
        cluster_count_layout.addWidget(self.auto_cluster_k_input)
        cluster_count_row.setLayout(cluster_count_layout)

        self.cluster_every_input = QSpinBox()
        self.cluster_every_input.setRange(1, 100)
        self.cluster_every_input.setValue(1)
        self.cluster_every_input.setToolTip(
            "Rebuild embedding clusters every N epochs. Use 1 while the shared head is changing; higher values reuse stale clusters for speed."
        )

        self.enable_clustering_input = QCheckBox()
        self.enable_clustering_input.setChecked(True)
        self.enable_clustering_input.setToolTip(
            "ON  (default): per-epoch K-means clustering, cluster-consistency loss, and\n"
            "structured hard negatives (same/nearest cluster) are all active.\n"
            "OFF: pure pairwise training — InfoNCE + variance only. No K-means pass,\n"
            "no consistency loss, no hard-negative bank. Epochs run faster (skips the\n"
            "per-epoch embedding extraction); useful as an ablation baseline or for\n"
            "quick fine-tuning runs. Cluster-dependent previews need cluster data."
        )

        self.cluster_consistency_weight_input = QDoubleSpinBox()
        self.cluster_consistency_weight_input.setDecimals(2)
        self.cluster_consistency_weight_input.setRange(0.0, 20.0)
        self.cluster_consistency_weight_input.setSingleStep(0.5)
        self.cluster_consistency_weight_input.setValue(2.0)
        self.cluster_consistency_weight_input.setToolTip(
            "Weight of the cluster consistency (symmetric KL) loss term.\n"
            "0 = disabled. Default 2.0. Increase to enforce tighter cluster routing."
        )

        self.negative_weight_input = QDoubleSpinBox()
        self.negative_weight_input.setDecimals(2)
        self.negative_weight_input.setRange(0.0, 50.0)
        self.negative_weight_input.setSingleStep(1.0)
        self.negative_weight_input.setValue(10.0)
        self.negative_weight_input.setToolTip(
            "Weight of the hard-negative loss term.\n"
            "Default 10.0. Set to 0 to disable in-cluster hard negatives.\n"
            "Increase to push hard negatives harder; decrease to reduce collapse risk."
        )

        dataset_row = QWidget()
        dataset_row_layout = QHBoxLayout()
        dataset_row_layout.setContentsMargins(0, 0, 0, 0)
        self.dataset_root_input = QLineEdit(DATASET_ROOT)
        self.dataset_root_input.setToolTip("Root directory of the training dataset (folder containing image pairs).")
        dataset_browse_btn = QPushButton("Browse...")
        dataset_browse_btn.clicked.connect(self.browse_dataset_root)
        dataset_row_layout.addWidget(self.dataset_root_input)
        dataset_row_layout.addWidget(dataset_browse_btn)
        dataset_row.setLayout(dataset_row_layout)

        # Game4Loc / GTA-UAV dataset controls (shown when dataset_type == "game4loc")
        self.dataset_type_input = QComboBox()
        self.dataset_type_input.addItem("Crop Pairs (anchor/ + positive/)", "crop_pairs")
        self.dataset_type_input.addItem("Game4Loc JSON (GTA-UAV / VisLoc)", "game4loc")
        self.dataset_type_input.addItem("University-1652  (drone → satellite)", "university1652")
        self.dataset_type_input.addItem("DenseUAV (multi-scale drone → satellite)", "denseuav")
        self.dataset_type_input.addItem("SUES-200  (drone → satellite, use train split)", "sues200")
        self.dataset_type_input.setToolTip(
            "Crop Pairs: classic anchor/positive directory pairs.\n"
            "Game4Loc JSON: drone+satellite pairs from a Game4Loc-format JSON file\n"
            "  (GTA-UAV, UAV-VisLoc).  Uses IoU weights for WeightedInfoNCE.\n"
            "University-1652: building geo-localization (701 train locations, D→S only).\n"
            "DenseUAV: multi-altitude dataset (H80/H90/H100) from Zhejiang universities."
        )

        self.gta_data_root_input = QLineEdit(r"D:\UAV_DATASET\GTA-UAV-LR")
        self.gta_data_root_input.setToolTip("Root directory of the GTA-UAV / UAV-VisLoc dataset.")
        gta_browse_btn = QPushButton("Browse...")
        gta_browse_btn.clicked.connect(self._browse_gta_root)
        gta_root_row = QWidget()
        gta_root_layout = QHBoxLayout()
        gta_root_layout.setContentsMargins(0, 0, 0, 0)
        gta_root_layout.addWidget(self.gta_data_root_input)
        gta_root_layout.addWidget(gta_browse_btn)
        gta_root_row.setLayout(gta_root_layout)

        self.u1652_root_input = QLineEdit(
            r"D:\UAV_DATASET\university-1652\University-Release\train")
        self.u1652_root_input.setToolTip(
            "University-1652 train directory — must contain drone/ and satellite/ sub-folders.")
        u1652_browse_btn = QPushButton("Browse...")
        u1652_browse_btn.clicked.connect(self._browse_u1652_root)
        u1652_root_row = QWidget()
        u1652_root_layout = QHBoxLayout()
        u1652_root_layout.setContentsMargins(0, 0, 0, 0)
        u1652_root_layout.addWidget(self.u1652_root_input)
        u1652_root_layout.addWidget(u1652_browse_btn)
        u1652_root_row.setLayout(u1652_root_layout)

        self.denseuav_root_input = QLineEdit(
            r"D:\UAV_DATASET\DenseUAV\DenseUAV\train")
        self.denseuav_root_input.setToolTip(
            "DenseUAV train directory — must contain drone/ and satellite/ sub-folders.")
        denseuav_browse_btn = QPushButton("Browse...")
        denseuav_browse_btn.clicked.connect(self._browse_denseuav_root)
        denseuav_root_row = QWidget()
        denseuav_root_layout = QHBoxLayout()
        denseuav_root_layout.setContentsMargins(0, 0, 0, 0)
        denseuav_root_layout.addWidget(self.denseuav_root_input)
        denseuav_root_layout.addWidget(denseuav_browse_btn)
        denseuav_root_row.setLayout(denseuav_root_layout)

        self.sues200_root_input = QLineEdit(
            r"D:\UAV_DATASET\SUES-200-split-official\train")
        self.sues200_root_input.setToolTip(
            "SUES-200 TRAIN SPLIT directory — must contain drone_view_512/ and\n"
            "satellite-view/ sub-folders. Use SUES-200-split-official/train —\n"
            "the FIXED 120-location split from the official benchmark repo\n"
            "(github.com/Reza-Zhu/SUES-200-Benchmark, script/indexs.yaml), NOT\n"
            "the old SUES-200-split (a random seed-42 split that assigns a\n"
            "DIFFERENT 120/80 partition — checkpoints trained on it may have\n"
            "trained on locations the official protocol reserves for testing).\n"
            "Also NOT the raw unsplit 200-location dataset — evaluating on\n"
            "trained locations makes the eval meaningless either way.")
        sues200_browse_btn = QPushButton("Browse...")
        sues200_browse_btn.clicked.connect(
            lambda: self._browse_dir_into(self.sues200_root_input))
        sues200_root_row = QWidget()
        sues200_root_layout = QHBoxLayout()
        sues200_root_layout.setContentsMargins(0, 0, 0, 0)
        sues200_root_layout.addWidget(self.sues200_root_input)
        sues200_root_layout.addWidget(sues200_browse_btn)
        sues200_root_row.setLayout(sues200_root_layout)

        # DenseUAV multi-positive (cross-altitude weighting) controls
        self.denseuav_cross_alt_input = QCheckBox("Cross-altitude multi-positive")
        self.denseuav_cross_alt_input.setChecked(True)
        self.denseuav_cross_alt_input.setToolTip(
            "Pair each drone image with the satellite tile at EVERY altitude of the\n"
            "same location (all are the same place → all valid positives), each\n"
            "weighted by altitude match w = exp(-|Δalt_m| / tau).\n"
            "Off = same-altitude-only (H80↔H80), one positive per anchor.")
        self.denseuav_alt_tau_input = QDoubleSpinBox()
        self.denseuav_alt_tau_input.setRange(1.0, 200.0)
        self.denseuav_alt_tau_input.setSingleStep(5.0)
        self.denseuav_alt_tau_input.setValue(20.0)
        self.denseuav_alt_tau_input.setToolTip(
            "Altitude weight falloff tau (metres). Smaller = sharper penalty on\n"
            "scale gaps. At tau=20 with H80/H90/H100: Δ0→1.00, Δ10→0.61, Δ20→0.37.\n"
            "Disabled (greyed) when 'Full-strength (disable weighting)' is checked.")
        self.denseuav_alt_full_strength_input = QCheckBox("Full-strength (disable weighting)")
        self.denseuav_alt_full_strength_input.setChecked(False)
        self.denseuav_alt_full_strength_input.setToolTip(
            "Give every altitude combo w=1.0 (no label-smoothing attenuation for\n"
            "cross-altitude pairs) — matches the official DenseUAV baseline, which\n"
            "trains every drone/satellite altitude combination at full strength\n"
            "with no distance-based softening (CE/triplet/KL, all same class).\n"
            "Use this to A/B against the default weighted mode.")
        self.denseuav_alt_full_strength_input.toggled.connect(
            lambda checked: self.denseuav_alt_tau_input.setEnabled(not checked))
        denseuav_mp_row = QWidget()
        denseuav_mp_layout = QHBoxLayout()
        denseuav_mp_layout.setContentsMargins(0, 0, 0, 0)
        denseuav_mp_layout.addWidget(self.denseuav_cross_alt_input)
        denseuav_mp_layout.addWidget(QLabel("weight tau (m):"))
        denseuav_mp_layout.addWidget(self.denseuav_alt_tau_input)
        denseuav_mp_layout.addWidget(self.denseuav_alt_full_strength_input)
        denseuav_mp_row.setLayout(denseuav_mp_layout)

        self.gta_json_input = QComboBox()
        self.gta_json_input.setEditable(True)
        for jf in ["pairs_train.json",
                   "same-area-drone2sate-train.json",
                   "cross-area-drone2sate-train.json"]:
            self.gta_json_input.addItem(jf)
        self.gta_json_input.setToolTip(
            "JSON pairs file (relative to data root, or absolute path).\n"
            "Use 'pairs_train.json' for synthetic datasets from dataset_gen.py.\n"
            "Use 'same-area-drone2sate-train.json' for UAV-VisLoc."
        )
        gta_json_browse = QPushButton("…")
        gta_json_browse.setFixedWidth(28)
        gta_json_browse.setFixedHeight(24)
        gta_json_browse.clicked.connect(self._browse_train_json)
        gta_json_row = QWidget()
        gta_json_layout = QHBoxLayout()
        gta_json_layout.setContentsMargins(0, 0, 0, 0)
        gta_json_layout.addWidget(self.gta_json_input)
        gta_json_layout.addWidget(gta_json_browse)
        gta_json_row.setLayout(gta_json_layout)

        self.gta_mode_input = QComboBox()
        self.gta_mode_input.addItem("pos + semi-positive (IoU-weighted)", "pos_semipos")
        self.gta_mode_input.addItem("positive only", "pos")
        self.gta_mode_input.setToolTip(
            "pos_semipos: includes semi-positive pairs with IoU weights → richer signal.\n"
            "pos: only strong-IoU pairs."
        )

        self.gta_augment_pos_input = QCheckBox()
        self.gta_augment_pos_input.setToolTip(
            "After loading JSON pairs, add extra positives where tile recall ≥ 40%:\n"
            "tiles with ≥40% of their area inside the drone footprint but IoU < 0.39.\n"
            "Catches adjacent partially-overlapping tiles that appear as hard negatives.\n"
            "Requires {seq}/{seq}.csv altitude+heading metadata.  VisLoc only."
        )

        self.filter_pos_input = QCheckBox()
        self.filter_pos_input.setChecked(False)
        self.filter_pos_input.setToolTip(
            "Before training, prune non-distinctive positives using the\n"
            "distinctiveness ratio = sim(tile, anchor) / avg sim(tile, negatives).\n"
            "Removes featureless/generic positives (e.g. open water) that match\n"
            "the query no better than random tiles. Writes pairs_train_distinctive.json\n"
            "and trains on it. Uses the STARTING model to embed — resume from a\n"
            "trained checkpoint for a meaningful ranking."
        )
        self.filter_pos_thresh_input = QDoubleSpinBox()
        self.filter_pos_thresh_input.setRange(0.5, 5.0)
        self.filter_pos_thresh_input.setSingleStep(0.1)
        self.filter_pos_thresh_input.setDecimals(2)
        self.filter_pos_thresh_input.setValue(1.0)
        self.filter_pos_thresh_input.setToolTip(
            "Minimum distinctiveness ratio to keep a positive.\n"
            "1.0 = tile must match its anchor better than the average negative.\n"
            "Higher = stricter. The single best positive per query is always kept."
        )
        self.filter_pos_max_input = QDoubleSpinBox()
        self.filter_pos_max_input.setRange(0.0, 10.0)
        self.filter_pos_max_input.setSingleStep(0.1)
        self.filter_pos_max_input.setDecimals(2)
        self.filter_pos_max_input.setValue(0.0)
        self.filter_pos_max_input.setToolTip(
            "Maximum distinctiveness ratio (0 = disabled).\n"
            "Removes trivially-easy positives (featureless water/sky) whose ratio\n"
            "is inflated because avg_neg is near-zero. Unlike the min floor, the\n"
            "'best always kept' rule does NOT apply — featureless anchors are\n"
            "dropped entirely. Typical value: 1.5."
        )
        self.filter_pos_row = QWidget()
        _fpl = QHBoxLayout(self.filter_pos_row)
        _fpl.setContentsMargins(0, 0, 0, 0)
        _fpl.addWidget(self.filter_pos_input)
        _fpl.addWidget(QLabel("min ratio"))
        _fpl.addWidget(self.filter_pos_thresh_input)
        _fpl.addWidget(QLabel("max ratio"))
        _fpl.addWidget(self.filter_pos_max_input)
        _fpl.addStretch()

        self.gta_test_json_input = QComboBox()
        self.gta_test_json_input.setEditable(True)
        self.gta_test_json_input.addItem("")   # empty = skip per-epoch eval
        for jf in ["same-area-drone2sate-test.json",
                   "cross-area-drone2sate-test.json"]:
            self.gta_test_json_input.addItem(jf)
        self.gta_test_json_input.setToolTip(
            "Optional test JSON for per-epoch evaluation (Game4Loc style).\n"
            "Leave empty to skip. When set, Recall@1, SDM@1 and Dis@1 are logged\n"
            "after each epoch."
        )
        gta_test_json_browse = QPushButton("…")
        gta_test_json_browse.setFixedWidth(28)
        gta_test_json_browse.setFixedHeight(24)
        gta_test_json_browse.clicked.connect(self._browse_test_json)
        gta_test_json_row = QWidget()
        gta_test_json_layout = QHBoxLayout()
        gta_test_json_layout.setContentsMargins(0, 0, 0, 0)
        gta_test_json_layout.addWidget(self.gta_test_json_input)
        gta_test_json_layout.addWidget(gta_test_json_browse)
        gta_test_json_row.setLayout(gta_test_json_layout)

        def _toggle_gta_controls():
            ds = self.dataset_type_input.currentData()
            is_gta      = ds == "game4loc"
            is_u1652    = ds == "university1652"
            is_denseuav = ds == "denseuav"
            gta_root_row.setVisible(is_gta)
            gta_json_row.setVisible(is_gta)
            self.gta_mode_input.setVisible(is_gta)
            self.gta_augment_pos_input.setVisible(is_gta)
            gta_test_json_row.setVisible(is_gta)
            dataset_row.setVisible(ds == "crop_pairs")
            u1652_root_row.setVisible(is_u1652)
            denseuav_root_row.setVisible(is_denseuav)
            denseuav_mp_row.setVisible(is_denseuav)
            sues200_root_row.setVisible(ds == "sues200")
        self.dataset_type_input.currentIndexChanged.connect(_toggle_gta_controls)
        gta_root_row.setVisible(False)
        gta_json_row.setVisible(False)
        self.gta_mode_input.setVisible(False)
        self.gta_augment_pos_input.setVisible(False)
        gta_test_json_row.setVisible(False)
        u1652_root_row.setVisible(False)
        denseuav_root_row.setVisible(False)
        denseuav_mp_row.setVisible(False)
        sues200_root_row.setVisible(False)

        _params_left.addRow("Epochs", self.epochs_input)
        _params_left.addRow("Batch size", self.batch_size_input)
        _params_left.addRow("GPU micro-batch", self.microbatch_size_input)
        _params_left.addRow("Learning rate", self.lr_input)
        # Weight decay / Device / Mixed precision / Shuffle dataset are fixed to
        # their defaults (0.01 / auto-CUDA / AMP-on / shuffle-on) and not exposed
        # in the UI — those widgets remain as hidden holders.
        _params_left.addRow("DataLoader workers", self.num_workers_input)
        _params_left.addRow("Backbone", self.backbone_input)
        _params_left.addRow("Pretrained", self.pretrained_input)
        _params_left.addRow("Unfreeze backbone tail (stages)", self.unfreeze_backbone_layers_input)
        _params_left.addRow("Resume checkpoint", self.resume_checkpoint_input)
        _params_left.addRow("Checkpoint path", checkpoint_path_row)
        _params_left.addRow("Augmentation", self.augment_input)
        _params_left.addRow("Hard-example mining", self.hard_mining_input)
        _params_left.addRow("Cluster model (fast)", self.cluster_model_row)
        _params_left.addRow("Enable clustering", self.enable_clustering_input)
        _params_left.addRow("Target clusters", cluster_count_row)
        _params_left.addRow("Cluster every N epochs", self.cluster_every_input)
        _params_left.addRow("Cluster consistency weight", self.cluster_consistency_weight_input)
        _params_right.addRow("Negative loss weight", self.negative_weight_input)

        self.label_smoothing_input = QDoubleSpinBox()
        self.label_smoothing_input.setDecimals(3)
        self.label_smoothing_input.setRange(0.0, 0.5)
        self.label_smoothing_input.setSingleStep(0.01)
        self.label_smoothing_input.setValue(0.05)
        self.label_smoothing_input.setToolTip(
            "Label smoothing ε for WeightedInfoNCE / GroupInfoNCE (Game4Loc style).\n"
            "0 = hard InfoNCE (original behaviour). ~0.05 adds soft uniform pressure.\n"
            "High-IoU pairs auto-reduce ε toward 0; low-overlap pairs stay smooth."
        )
        _params_right.addRow("Label smoothing", self.label_smoothing_input)

        self.group_size_input = QSpinBox()
        self.group_size_input.setRange(1, 4)
        self.group_size_input.setValue(1)
        self.group_size_input.setToolTip(
            "Group size N for GroupInfoNCE whole_slice (Game4Loc style).\n"
            "1 = WeightedInfoNCE (standard paired training).\n"
            "2 = each pair produces 2 independent augmented views; the loss treats\n"
            "    both views of the same pair as mutual positives, giving richer\n"
            "    contrastive signal within each group."
        )
        _params_right.addRow("Group size (NCE)", self.group_size_input)

        _params_right.addRow("Dataset type", self.dataset_type_input)
        _params_right.addRow("Dataset root", dataset_row)
        _params_right.addRow("GTA-UAV / VisLoc root", gta_root_row)
        _params_right.addRow("Train JSON", gta_json_row)
        _params_right.addRow("Pair mode", self.gta_mode_input)
        _params_right.addRow("Augment positives (tile recall ≥ 40%)", self.gta_augment_pos_input)
        _params_right.addRow("Filter positives (distinctive)", self.filter_pos_row)
        _params_right.addRow("Test JSON (eval/epoch)", gta_test_json_row)
        _params_right.addRow("University-1652 train root", u1652_root_row)
        _params_right.addRow("DenseUAV train root", denseuav_root_row)
        _params_right.addRow("SUES-200 train root", sues200_root_row)
        _params_right.addRow("DenseUAV multi-positive", denseuav_mp_row)

        self.epoch_eval_input = QCheckBox()
        self.epoch_eval_input.setChecked(False)
        self.epoch_eval_input.setToolTip(
            "Run a VisLoc/GTA evaluation on the Test JSON after EVERY training epoch.\n"
            "Off by default — per-epoch eval is slow. The checkpoint is always saved\n"
            "after each epoch regardless of this setting."
        )
        _params_right.addRow("Eval after each epoch", self.epoch_eval_input)

        self.quick_eval_denseuav_input = QCheckBox()
        self.quick_eval_denseuav_input.setChecked(False)
        self.quick_eval_denseuav_input.setToolTip(
            "Every N epochs (right), run a lightweight DenseUAV test-set eval (exact-ID\n"
            "R@1/R@5/R@10/mAP, full confusion gallery) using the CURRENT in-training\n"
            "weights and log the result. Only applies when Dataset type = DenseUAV.\n"
            "Useful because training loss can plateau while retrieval quality keeps\n"
            "improving — this gives a periodic ground-truth signal without waiting\n"
            "for the run to finish. Does not touch eval history/cache; independent\n"
            "of 'Eval after each epoch' (which is GTA/VisLoc-only).")
        self.quick_eval_every_n_input = QSpinBox()
        self.quick_eval_every_n_input.setRange(1, 100)
        self.quick_eval_every_n_input.setValue(10)
        _qe_row = QWidget()
        _qe_lay = QHBoxLayout(); _qe_lay.setContentsMargins(0, 0, 0, 0)
        _qe_lay.addWidget(self.quick_eval_denseuav_input)
        _qe_lay.addWidget(QLabel("every"))
        _qe_lay.addWidget(self.quick_eval_every_n_input)
        _qe_lay.addWidget(QLabel("epochs"))
        _qe_lay.addStretch()
        _qe_row.setLayout(_qe_lay)
        _params_right.addRow("Quick eval (DenseUAV)", _qe_row)

        self.auto_eval_input = QCheckBox()
        self.auto_eval_input.setToolTip(
            "After each training run, automatically evaluate the saved checkpoint using\n"
            "the same settings as eval run #62 (seq=04, two_stage_cluster, cluster_mah,\n"
            "eps=0.5, K=16, 400m/130m). Results are stored in train_history_runs.json\n"
            "and train_history.html is regenerated."
        )
        _params_right.addRow("Auto eval after training", self.auto_eval_input)

        self.auto_sh_input = QCheckBox()
        self.auto_sh_input.setToolTip(
            "After backbone training finishes, automatically run scale head training\n"
            "on the same model using the settings in the Scale Head tab.\n"
            "The checkpoint is saved again after scale head training completes.\n"
            "Has no effect if the backbone has no scale head (e.g. swin_b_nohead)."
        )
        _params_right.addRow("Auto-train scale head after", self.auto_sh_input)

        self.training_label_input = QLineEdit()
        self.training_label_input.setPlaceholderText("e.g. same area, cross area")
        self.training_label_input.setToolTip("Optional label saved to train_history_runs.json to identify this run.")
        _params_right.addRow("Run label", self.training_label_input)

        # ── Remote training over SSH ──
        self.remote_train_input = QCheckBox()
        self.remote_train_input.setToolTip(
            "Run training on a remote GPU server over SSH instead of this machine.\n"
            "Local code (dataset/model/trainer/loss/clustering) is synced to the\n"
            "remote before each run; metrics stream back live into this GUI, and\n"
            "the checkpoint is pulled to the local checkpoint path after every\n"
            "epoch so local eval GUIs can use it. Stop works as usual.\n"
            "Requires passwordless SSH (key auth) to the host alias."
        )
        _params_right.addRow("Train on remote (SSH)", self.remote_train_input)
        self.remote_host_input = QLineEdit("nvidia5090")
        self.remote_host_input.setToolTip("SSH host alias or user@host (must be key-authenticated).")
        _params_right.addRow("Remote host", self.remote_host_input)
        self.remote_code_dir_input = QLineEdit("~/Thinghiem/uav_code")
        self.remote_code_dir_input.setToolTip("Directory on the remote where code is synced and runs are stored.")
        _params_right.addRow("Remote code dir", self.remote_code_dir_input)
        self.remote_data_root_input = QLineEdit("~/Thinghiem/data/SUES-200-split-official/train")
        self.remote_data_root_input.setToolTip(
            "Train-split root ON THE REMOTE for the selected dataset type\n"
            "(replaces the local dataset root when remote training is enabled).")
        _params_right.addRow("Remote dataset root", self.remote_data_root_input)
        # Per-dataset remote roots: the field auto-swaps when the dataset type
        # changes, so switching e.g. SUES-200 -> DenseUAV can't silently launch
        # a remote run pointing at the previous dataset's path (run #224
        # failure mode). Edits are remembered per dataset type.
        self._remote_roots_by_ds = {
            "sues200": "~/Thinghiem/data/SUES-200-split-official/train",
            "denseuav": "~/Thinghiem/data/DenseUAV/DenseUAV/train",
            "university1652": "~/Thinghiem/data/university-1652/University-Release/train",
            "game4loc": "~/Thinghiem/data/GTA-UAV-LR",
        }
        self._rr_prev_ds = self.dataset_type_input.currentData()
        self.dataset_type_input.currentIndexChanged.connect(self._on_ds_remote_root_swap)

        _params_cols = QHBoxLayout()
        _params_cols.addLayout(_params_left)
        _params_cols.addSpacing(16)
        _params_cols.addLayout(_params_right)
        params.setLayout(_params_cols)

        model_info = QGroupBox("Model Size")
        model_info_layout = QFormLayout()
        self.backbone_size_label = QLabel()
        self.head_size_label = QLabel()
        self.total_size_label = QLabel()
        self.model_memory_label = QLabel()
        model_info_layout.addRow("Backbone params", self.backbone_size_label)
        model_info_layout.addRow("Head params", self.head_size_label)
        model_info_layout.addRow("Total params", self.total_size_label)
        model_info_layout.addRow("FP32 weights", self.model_memory_label)
        model_info.setLayout(model_info_layout)

        cluster_info = QGroupBox("Embedding Clusters")
        cluster_info_layout = QFormLayout()
        self.cluster_epoch_label = QLabel("Not generated")
        self.cluster_target_label = QLabel("-")
        self.cluster_non_empty_label = QLabel("-")
        self.cluster_samples_label = QLabel("-")
        self.cluster_pair_probability_label = QLabel("-")
        self.dead_cluster_count_label = QLabel("-")
        self.dead_sample_count_label = QLabel("-")
        cluster_info_layout.addRow("Last update", self.cluster_epoch_label)
        cluster_info_layout.addRow("Target clusters", self.cluster_target_label)
        cluster_info_layout.addRow("Non-empty clusters", self.cluster_non_empty_label)
        cluster_info_layout.addRow("Embedded samples", self.cluster_samples_label)
        cluster_info_layout.addRow("Anchor-positive same cluster", self.cluster_pair_probability_label)
        cluster_info_layout.addRow("Dead clusters", self.dead_cluster_count_label)
        cluster_info_layout.addRow("Dead cluster samples", self.dead_sample_count_label)
        cluster_info.setLayout(cluster_info_layout)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.total_loss_figure = Figure(figsize=(5, 2.5))
        self.total_loss_canvas = FigureCanvas(self.total_loss_figure)
        self.total_loss_ax = self.total_loss_figure.add_subplot(111)
        self.total_loss_ax.set_title("Total Loss")
        self.total_loss_ax.set_xlabel("Step")
        self.total_loss_ax.set_ylabel("Loss")
        self.total_loss_ax.grid(True)
        self.total_loss_figure.tight_layout()

        self.figure = Figure(figsize=(5, 2.5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Loss Elements")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Value")
        self.ax.grid(True)
        self.figure.tight_layout()

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Latest value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setFixedHeight(170)

        controls.addWidget(self.train_btn)
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.stop_btn)
        controls.addWidget(self.reattach_btn)
        controls.addWidget(self.save_btn)
        controls.addWidget(self.preview_aug_btn)
        controls.addStretch()

        input_layout.addLayout(controls)
        input_layout.addWidget(params)
        input_layout.addWidget(model_info)
        input_layout.addWidget(cluster_info)
        input_layout.addStretch()
        input_tab.setLayout(input_layout)

        monitor_layout.addWidget(self.total_loss_canvas, stretch=2)
        monitor_layout.addWidget(self.canvas, stretch=2)
        monitor_layout.addWidget(self.metrics_table)
        monitor_layout.addWidget(self.log, stretch=2)
        monitor_tab.setLayout(monitor_layout)

        dead_cluster_tab = QWidget()
        dead_cluster_layout = QVBoxLayout()
        self.dead_cluster_status_label = QLabel("Run training to see dead cluster samples.")
        self.cluster_variance_figure = Figure(figsize=(7, 2.5))
        self.cluster_variance_canvas = FigureCanvas(self.cluster_variance_figure)
        self.cluster_variance_ax = self.cluster_variance_figure.add_subplot(111)
        self.cluster_variance_ax.set_title("Per-cluster descriptor variance (click a bar to browse samples)")
        self.cluster_variance_ax.set_xlabel("Cluster ID")
        self.cluster_variance_ax.set_ylabel("Mean per-dim variance")
        self.cluster_variance_figure.tight_layout()
        self.cluster_variance_canvas.mpl_connect("button_press_event", self.on_variance_bar_clicked)
        self.dead_cluster_scroll = QScrollArea()
        self.dead_cluster_scroll.setWidgetResizable(True)
        self.dead_cluster_grid_widget = QWidget()
        self.dead_cluster_grid = QGridLayout()
        self.dead_cluster_grid.setSpacing(4)
        self.dead_cluster_grid_widget.setLayout(self.dead_cluster_grid)
        self.dead_cluster_scroll.setWidget(self.dead_cluster_grid_widget)
        dead_cluster_layout.addWidget(self.dead_cluster_status_label)
        dead_cluster_layout.addWidget(self.cluster_variance_canvas)
        dead_cluster_layout.addWidget(self.dead_cluster_scroll, stretch=1)
        dead_cluster_tab.setLayout(dead_cluster_layout)

        # ── Cluster exclusion table + sample preview (inside Cluster Statistics tab) ──
        self.cluster_excl_table = QTableWidget(0, 5)
        self.cluster_excl_table.setHorizontalHeaderLabels(
            ["Include", "Cluster", "Size", "Loss", "Variance"])
        self.cluster_excl_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.cluster_excl_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.cluster_excl_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_excl_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.cluster_excl_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cluster_excl_table.setMaximumHeight(180)
        self.cluster_excl_table.itemSelectionChanged.connect(self._on_cluster_table_selection_changed)

        self.cluster_excl_status = QLabel("Run training to populate cluster table.")
        self.cluster_excl_status.setStyleSheet("color:#555;font-style:italic;")

        excl_ctrl_row = QWidget()
        _ecrl = QHBoxLayout(excl_ctrl_row)
        _ecrl.setContentsMargins(0, 0, 0, 0)
        self.build_cluster_data_btn = QPushButton("Build Cluster Data")
        self.build_cluster_data_btn.setToolTip(
            "Compute cluster assignments from the current model and K-means centroids. "
            "Runs automatically after each training epoch."
        )
        self.build_cluster_data_btn.clicked.connect(self.start_build_cluster_data)
        self.clear_excl_btn = QPushButton("Clear All Exclusions")
        self.clear_excl_btn.clicked.connect(self._clear_all_exclusions)
        _ecrl.addWidget(self.build_cluster_data_btn)
        _ecrl.addWidget(self.clear_excl_btn)
        _ecrl.addStretch()

        dead_cluster_layout.addWidget(excl_ctrl_row)
        dead_cluster_layout.addWidget(self.cluster_excl_table)
        dead_cluster_layout.addWidget(self.cluster_excl_status)
        dead_cluster_layout.addWidget(self.dead_cluster_scroll, stretch=1)
        dead_cluster_tab.setLayout(dead_cluster_layout)
        # ─────────────────────────────────────────────────────────────────────

        tabs.addTab(input_tab, "Input & Parameters")
        tabs.addTab(monitor_tab, "Loss, Results & Log")
        tabs.addTab(dead_cluster_tab, "Cluster Statistics")
        tabs.addTab(self._build_scale_head_tab(), "Scale Head")

        layout.addWidget(tabs)

        self.setLayout(layout)
        self._default_palette = self.palette()
        self._load_ui_settings()
        self._apply_remote_theme(self.remote_train_input.isChecked())
        # Connected AFTER the initial load (not at checkbox-creation time):
        # _load_ui_settings() above may itself call setChecked() while
        # restoring last session's state, and that must NOT re-trigger the
        # profile auto-switch handler mid-startup.
        self.remote_train_input.toggled.connect(self._on_remote_toggle)
        self.remote_train_input.toggled.connect(self._apply_remote_theme)
        self.update_model_size_preview()

    def _apply_remote_theme(self, remote: bool):
        """Tints the window light green while "Train on remote (SSH)" is
        checked, so it's visually obvious at a glance which config a given
        open GUI window is pointed at (useful when running local + remote
        training side by side)."""
        if remote:
            pal = self.palette()
            pal.setColor(QPalette.Window, QColor("#d6f5d6"))
            pal.setColor(QPalette.Base, QColor("#d6f5d6"))
        else:
            pal = self._default_palette
        self.setPalette(pal)
        self.setAutoFillBackground(True)
        self.tabs.setPalette(pal)
        self.tabs.setAutoFillBackground(True)
        for i in range(self.tabs.count()):
            page = self.tabs.widget(i)
            page.setPalette(pal)
            page.setAutoFillBackground(True)

    def start_training(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            self.log.append("Training is already running.")
            return

        self.losses.clear()
        self.loss_records.clear()
        self.update_plot()
        self.update_metrics_table({})
        self.reset_cluster_status()
        self.stop_event.clear()
        self._save_ui_settings()
        self.training_params = self.read_training_params()
        # Zero-trainable-params guard: a no-head backbone has no projection
        # head, so the backbone itself is the only thing that CAN train —
        # unfreeze=0 leaves nothing requiring grad and the first backward()
        # dies with "element 0 of tensors does not require grad" (seen live
        # with convnext_t on the remote). Catch it before launch instead.
        if (self.training_params["training_mode"] == "backbone_raw_nohead"
                and self.training_params.get("unfreeze_backbone_layers", 0) < 1):
            self.log.append(
                "ERROR: backbone "
                f"'{self.backbone_input.currentData()}' has no projection head — "
                "training optimizes the backbone directly, but 'Unfreeze backbone "
                "layers' is 0, so NO parameters would train. Set unfreeze >= 1 "
                "and start again.")
            return
        self.log.append(f"DEBUG: UI dataset_type={repr(self.training_params.get('dataset_type'))}, dataset_root={repr(self.training_params.get('dataset_root'))}")
        self.set_parameter_inputs_enabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.pause_btn.blockSignals(False)
        self.pause_btn.setEnabled(True)
        self.pause_event.clear()
        self._paused_pull_pending = None
        ckpt_path = self.checkpoint_path_for()
        self.log.append(
            "Training started with "
            f"epochs={self.training_params['epochs']}, "
            f"batch_size={self.training_params['batch_size']}, "
            f"microbatch={self.training_params['microbatch_size']}, "
            f"lr={self.training_params['learning_rate']}, "
            f"device={self.training_params['device']}, "
            f"workers={self.training_params['num_workers']}, "
            f"amp={self.training_params['use_amp']}, "
            f"mode={self.training_params['training_mode']}, "
            f"resume={self.training_params['resume_checkpoint']}, "
            f"clusters={self.training_params['cluster_count']}, "
            f"checkpoint={ckpt_path}."
        )

        if self.training_params.get("remote_train"):
            # run_dir isn't known yet (generated inside remote_train_loop),
            # but enabling here (main thread — remote_train_loop runs on a
            # background thread, where touching widgets directly isn't safe)
            # is harmless: start_reattach_remote defensively checks
            # _last_remote_run_dir is actually set before doing anything.
            self.reattach_btn.setEnabled(True)
        target = (self.remote_train_loop if self.training_params.get("remote_train")
                  else self.train_loop)
        self.training_thread = threading.Thread(target=target, daemon=True)
        self.training_thread.start()

    def start_reattach_remote(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            return
        if not self._last_remote_run_dir:
            self.log.append("No remote run to reattach to yet — launch one first.")
            return
        self.log.append(
            f"Reattaching to {self._last_remote_host}:{self._last_remote_run_dir} — "
            f"NOT touching the remote job, just resuming visibility into it.")
        self.set_parameter_inputs_enabled(False)
        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.pause_btn.blockSignals(False)
        self.pause_btn.setEnabled(True)
        self.stop_event.clear()
        self.pause_event.clear()
        self._paused_pull_pending = None
        host, run_dir = self._last_remote_host, self._last_remote_run_dir
        self.training_thread = threading.Thread(
            target=lambda: self.reattach_remote_run(host, run_dir), daemon=True)
        self.training_thread.start()

    def browse_dataset_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Root", self.dataset_root_input.text() or DATASET_ROOT
        )
        if path:
            self.dataset_root_input.setText(path)

    def _browse_gta_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select GTA-UAV / VisLoc Root", self.gta_data_root_input.text()
        )
        if path:
            self.gta_data_root_input.setText(path)

    def _browse_u1652_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select University-1652 train directory", self.u1652_root_input.text()
        )
        if path:
            self.u1652_root_input.setText(path)

    def _browse_denseuav_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select DenseUAV train directory", self.denseuav_root_input.text()
        )
        if path:
            self.denseuav_root_input.setText(path)

    def _browse_dir_into(self, line_edit):
        path = QFileDialog.getExistingDirectory(
            self, "Select directory", line_edit.text())
        if path:
            line_edit.setText(path)

    def _browse_cluster_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SimpleClusterCNN checkpoint",
            self.cluster_model_input.text() or str(Path(__file__).parent),
            "PyTorch checkpoint (*.pt);;All files (*)")
        if path:
            self.cluster_model_input.setText(path)

    def _browse_file(self, line_edit, file_filter="All files (*.*)"):
        """Generic open-file dialog that writes the chosen path into line_edit."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select file", line_edit.text() or str(Path(__file__).parent),
            file_filter)
        if path:
            line_edit.setText(path)

    def _browse_json_file(self, combo):
        """Open file dialog for a JSON pairs file; store relative path if inside data root."""
        start = self.gta_data_root_input.text() or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON pairs file", start, "JSON (*.json);;All files (*)"
        )
        if not path:
            return
        # Try to store relative path (relative to data root) — falls back to absolute
        root = self.gta_data_root_input.text().strip()
        if root:
            try:
                rel = Path(path).relative_to(Path(root))
                path = str(rel)
            except ValueError:
                pass  # different drive or not under root — keep absolute
        combo.setCurrentText(path)

    def _browse_train_json(self):
        self._browse_json_file(self.gta_json_input)

    def _browse_test_json(self):
        self._browse_json_file(self.gta_test_json_input)

    def toggle_pause(self, paused):
        """Freeze/unfreeze training at the next batch boundary — in place, NOT
        a stop/resume cycle: the process stays alive, the loader iterator keeps
        its position, and resume continues at the exact next batch of the same
        epoch (no epoch restart). While paused, remote checkpoint downloads
        are deferred (newest pending pull is remembered, flushed on resume)."""
        self.pause_btn.setText("Resume" if paused else "Pause")
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()
        if getattr(self, "_remote_run_dir", None):
            host, run_dir = self._remote_host, self._remote_run_dir
            cmd = (f"touch {run_dir}/pause.flag" if paused
                   else f"rm -f {run_dir}/pause.flag")
            threading.Thread(
                target=lambda: subprocess.run(
                    ["ssh", host, cmd], creationflags=_NO_WINDOW),
                daemon=True).start()
            self.log.append(
                "Pause flag sent to remote; pausing after the current batch "
                "(checkpoint downloads deferred)." if paused
                else "Resume sent to remote.")
        else:
            self.log.append("Pausing after the current batch..." if paused
                            else "Resuming...")
        if not paused:
            pending = getattr(self, "_paused_pull_pending", None)
            self._paused_pull_pending = None
            if pending:
                self._pull_remote_checkpoint_async(*pending)

    def stop_training(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            self.stop_event.set()
            self.pause_btn.blockSignals(True)
            self.pause_btn.setChecked(False)
            self.pause_btn.setText("Pause")
            self.pause_btn.blockSignals(False)
            self.pause_event.clear()   # a paused trainer must wake up to see the stop
            self.stop_btn.setEnabled(False)
            if getattr(self, "_remote_run_dir", None):
                host, run_dir = self._remote_host, self._remote_run_dir
                threading.Thread(
                    target=lambda: subprocess.run(
                        ["ssh", host, f"touch {run_dir}/stop.flag"],
                        creationflags=_NO_WINDOW),
                    daemon=True).start()
                self.log.append("Stop flag sent to remote; stopping after the current batch...")
            else:
                self.log.append("Stopping training after the current batch...")

    def closeEvent(self, event):
        """Closing the window mid-remote-run must not orphan the ssh stream:
        Python never kills child processes on exit, and a leftover ssh.exe
        both hangs the launching console (it inherits stdin) and dooms the
        remote run anyway (its stdout pipe loses its reader and fills up,
        blocking the remote process mid-print). Signal a clean remote stop
        and terminate the local ssh before closing."""
        proc = getattr(self, "_remote_proc", None)
        if proc is not None and proc.poll() is None:
            if getattr(self, "_remote_run_dir", None):
                host, run_dir = self._remote_host, self._remote_run_dir
                try:
                    subprocess.run(["ssh", host, f"touch {run_dir}/stop.flag"],
                                   timeout=10, capture_output=True,
                                   creationflags=_NO_WINDOW)
                except Exception:
                    pass
            try:
                proc.terminate()
            except Exception:
                pass
        event.accept()

    def save_model(self):
        if self.model is None:
            self.log.append("No model is available to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Model",
            "uav_localization_model.pt",
            "PyTorch Model (*.pt *.pth)"
        )
        if not path:
            return

        with self.model_lock:
            self.write_checkpoint(path)

        self.log.append(f"Saved model to {path}")

    def preview_positive_augmentations(self):
        try:
            from PIL import Image
        except Exception as exc:
            self.log.append(f"Could not import PIL: {exc}")
            return

        try:
            ds_type = self.dataset_type_input.currentData()
            if ds_type == "game4loc":
                from dataset import GtaUavDataset
                dataset = GtaUavDataset(
                    data_root=self.gta_data_root_input.text(),
                    pairs_meta_file=self.gta_json_input.currentText(),
                    mode=self.gta_mode_input.currentData() or "pos_semipos",
                    augment_positives=self.gta_augment_pos_input.isChecked(),
                    augment=self.augment_input.isChecked(),
                )
            elif ds_type == "university1652":
                from dataset import University1652Dataset
                dataset = University1652Dataset(self.u1652_root_input.text())
            elif ds_type == "sues200":
                from dataset import Sues200Dataset
                dataset = Sues200Dataset(
                    train_root=self.sues200_root_input.text(),
                    augment=self.augment_input.isChecked(),
                )
            elif ds_type == "denseuav":
                from dataset import DenseUAVDataset
                from model import backbone_img_size
                backbone_name = self.backbone_input.currentData()
                img_size = backbone_img_size(backbone_name)  # Call function, not .get()
                dataset = DenseUAVDataset(
                    train_root=self.denseuav_root_input.text(),
                    group_size=1,
                    img_size=img_size,
                    augment=self.augment_input.isChecked(),
                    cross_altitude=self.denseuav_cross_alt_input.isChecked(),
                    altitude_weight_tau=(None if self.denseuav_alt_full_strength_input.isChecked()
                                        else self.denseuav_alt_tau_input.value()),
                )
            else:
                from dataset import SatCropDataset
                dataset = SatCropDataset(self.dataset_root_input.text() or DATASET_ROOT)
        except Exception as exc:
            self.log.append(f"Could not load training dataset: {exc}")
            return
        if not dataset.pairs:
            self.log.append("No training pairs available for augmentation preview.")
            return

        # Whether positive augmentation is actually enabled for THIS dataset type in training
        # Photometric augmentation of positives is governed by the "Augmentation"
        # checkbox for ALL dataset types. (gta_augment_pos_input is unrelated: it
        # adds extra positive TILES by recall geometry, not image augmentation.)
        augment_enabled = self.augment_input.isChecked()

        sample_idx = random.randrange(len(dataset.pairs))
        pair = dataset.pairs[sample_idx]
        drone_path, positive_path = Path(pair[0]), Path(pair[1])
        try:
            drone_image = Image.open(drone_path).convert("RGB")
            original    = Image.open(positive_path).convert("RGB")
        except Exception as exc:
            self.log.append(f"Could not open pair images: {exc}")
            return

        from dataset import make_train_transform, DEFAULT_AUG

        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"Augmentation Preview — {drone_path.name}  →  {positive_path.name}"
        )
        dialog.resize(1100, 820)
        root_layout = QVBoxLayout()

        # ── Controls ────────────────────────────────────────────────────────
        ctrl_group = QGroupBox("Augmentation parameters")
        ctrl_form  = QFormLayout()
        ctrl_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        def _dspin(lo, hi, step, val):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi); sb.setSingleStep(step); sb.setValue(val)
            sb.setFixedWidth(80)
            return sb

        def _ispin(lo, hi, val):
            sb = QSpinBox()
            sb.setRange(lo, hi); sb.setValue(val)
            sb.setFixedWidth(80)
            return sb

        a = DEFAULT_AUG
        ctrl = {
            "brightness":  _dspin(0.0, 1.0, 0.01, a["brightness"]),
            "contrast":    _dspin(0.0, 1.0, 0.01, a["contrast"]),
            "saturation":  _dspin(0.0, 1.0, 0.01, a["saturation"]),
            "hue":         _dspin(0.0, 0.5, 0.01, a["hue"]),
            "jitter_p":    _dspin(0.0, 1.0, 0.05, a["jitter_p"]),
            "dropout_p":   _dspin(0.0, 1.0, 0.05, a["dropout_p"]),
            "min_holes":   _ispin(0, 50, a["min_holes"]),
            "max_holes":   _ispin(0, 50, a["max_holes"]),
            "min_frac":    _dspin(0.0, 0.5, 0.01, a["min_frac"]),
            "max_frac":    _dspin(0.0, 0.5, 0.01, a["max_frac"]),
        }
        ctrl_form.addRow("Brightness",      ctrl["brightness"])
        ctrl_form.addRow("Contrast",        ctrl["contrast"])
        ctrl_form.addRow("Saturation",      ctrl["saturation"])
        ctrl_form.addRow("Hue",             ctrl["hue"])
        ctrl_form.addRow("Jitter prob",     ctrl["jitter_p"])
        ctrl_form.addRow("Dropout prob",    ctrl["dropout_p"])
        ctrl_form.addRow("Min holes",       ctrl["min_holes"])
        ctrl_form.addRow("Max holes",       ctrl["max_holes"])
        ctrl_form.addRow("Min hole frac",   ctrl["min_frac"])
        ctrl_form.addRow("Max hole frac",   ctrl["max_frac"])
        ctrl_group.setLayout(ctrl_form)

        btn_row = QHBoxLayout()
        refresh_btn = QPushButton("Refresh previews")
        save_btn    = QPushButton("Apply to training")
        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(save_btn)

        # ── Image area ──────────────────────────────────────────────────────
        scroll   = QScrollArea(); scroll.setWidgetResizable(True)
        content  = QWidget()
        cont_lay = QVBoxLayout(); cont_lay.setSpacing(10)
        content.setLayout(cont_lay)
        scroll.setWidget(content)

        # Positive / augmentation section
        pos_box  = QGroupBox("Anchor  +  Positive  +  Augmented positives")
        pos_grid = QGridLayout(); pos_grid.setSpacing(6)
        pos_box.setLayout(pos_grid)
        cont_lay.addWidget(pos_box)

        # Scale crop section — drone image at simulated altitudes
        scale_box  = QGroupBox("Scale crop augmentation (drone/query) — simulates altitude diversity for scale head training")
        scale_grid = QGridLayout(); scale_grid.setSpacing(6)
        scale_box.setLayout(scale_grid)
        cont_lay.addWidget(scale_box)

        # Distinctiveness section — all positives of this anchor, kept vs removed
        dist_box   = QGroupBox("Positives of this anchor — kept (green) vs removed (red) by distinctiveness")
        dist_vlay  = QVBoxLayout(); dist_box.setLayout(dist_vlay)
        dist_status = QLabel("Start/resume training so a model is loaded, then Refresh.")
        dist_status.setStyleSheet("color:#888;font-style:italic;")
        dist_grid_w = QWidget()
        dist_grid   = QGridLayout(); dist_grid.setSpacing(6)
        dist_grid_w.setLayout(dist_grid)
        dist_vlay.addWidget(dist_status)
        dist_vlay.addWidget(dist_grid_w)
        cont_lay.addWidget(dist_box)

        # Negative section
        N_NEG = 9  # 6 from same cluster + 3 from nearest clusters
        neg_box    = QGroupBox("Hard negatives (6 same + 3 nearest clusters), ranked by cosine similarity")
        neg_vlay   = QVBoxLayout()
        neg_box.setLayout(neg_vlay)
        neg_status = QLabel("No cluster data — run Build Cluster Data first.")
        neg_status.setStyleSheet("color:#888;font-style:italic;")
        neg_grid_w = QWidget()
        neg_grid   = QGridLayout(); neg_grid.setSpacing(6)
        neg_grid_w.setLayout(neg_grid)
        neg_vlay.addWidget(neg_status)
        neg_vlay.addWidget(neg_grid_w)
        cont_lay.addWidget(neg_box)
        cont_lay.addStretch()

        orig_idx = int(pair[-1])   # dataset-level index of the selected sample

        def _make_panel(title, pixmap):
            panel = QWidget()
            pl = QVBoxLayout(); pl.setContentsMargins(2, 2, 2, 2)
            lbl_t = QLabel(title); lbl_t.setAlignment(Qt.AlignCenter)
            lbl_i = QLabel(); lbl_i.setAlignment(Qt.AlignCenter)
            lbl_i.setPixmap(pixmap.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            pl.addWidget(lbl_t); pl.addWidget(lbl_i)
            panel.setLayout(pl)
            return panel

        def _clear_grid(g):
            while g.count():
                item = g.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        def _build_pos_grid(transform):
            _clear_grid(pos_grid)
            drone_loc_id = drone_path.parent.name

            # Gather ALL positives that share this anchor (drone) image, with their
            # per-pair positive weight (3rd tuple element; 1.0 if absent).
            drone_key = str(drone_path)
            uniq_pos, seen = [], set()
            for p in dataset.pairs:
                if str(Path(p[0])) == drone_key:
                    s = str(Path(p[1]))
                    if s not in seen:
                        w = float(p[2]) if len(p) > 2 else 1.0
                        seen.add(s); uniq_pos.append((Path(p[1]), w))
            if not uniq_pos:
                uniq_pos = [(positive_path, 1.0)]
            # Highest-weight (hardest) positive first
            uniq_pos.sort(key=lambda x: -x[1])

            if augment_enabled:
                pos_box.setTitle(
                    f"Anchor  +  {len(uniq_pos)} weighted positive(s)  +  augmentations")
            else:
                pos_box.setTitle(
                    f"Anchor  +  {len(uniq_pos)} weighted positive(s)  "
                    f"(augmentation disabled in training)")

            panels = [
                (f"Anchor (drone)\n{drone_loc_id}\n{drone_path.name}",
                 self.pil_to_pixmap(drone_image)),
            ]
            for k, (pp, w) in enumerate(uniq_pos):
                try:
                    pos_img = Image.open(pp).convert("RGB")
                except Exception:
                    continue
                panels.append(
                    (f"Positive #{k+1}  w={w:.2f}\n{pp.parent.name}\n{pp.name}",
                     self.pil_to_pixmap(pos_img)))
                # Only show augmentations when training actually augments positives
                if augment_enabled:
                    for i in range(3):
                        panels.append(
                            (f"  ↳ Pos #{k+1} Aug #{i+1}",
                             self.tensor_to_pixmap(transform(pos_img))))
            for i, (title, px) in enumerate(panels):
                pos_grid.addWidget(_make_panel(title, px), i // 3, i % 3)

        def _center_square(img):
            """The center square the model actually sees (matches make_eval_transform:
            shorter-side resize + center crop). This is the true 'crop 100%' reference."""
            W, H = img.size
            m = min(W, H)
            left = (W - m) // 2
            top  = (H - m) // 2
            return img.crop((left, top, left + m, top + m))

        def _pil_scale_crop(img):
            """Square crop of side crop_ratio*min(W,H) at a random position — identical
            to scale_head_trainer._crop_augment. Returned as a SQUARE (no stretch back
            to the non-square frame), so 'crop 99%' looks ~99% of the center square."""
            W, H = img.size
            m = min(W, H)
            crop_ratio = random.uniform(0.70, 1.00)   # matches training range
            size = max(1, int(m * crop_ratio))
            left = random.randint(0, W - size)
            top  = random.randint(0, H - size)
            cropped = img.crop((left, top, left + size, top + size))
            return cropped, size / m

        def _build_scale_grid():
            _clear_grid(scale_grid)
            try:
                # 'Original' = center square (what model sees at 100%), NOT the full 4:3 frame
                panels = [("Original\n(center square, 100%)",
                           self.pil_to_pixmap(_center_square(drone_image)))]
                for i in range(5):
                    aug, frac = _pil_scale_crop(drone_image)
                    panels.append((f"Scale #{i+1}\n(crop {frac*100:.0f}%)",
                                   self.pil_to_pixmap(aug)))
                for i, (title, px) in enumerate(panels):
                    scale_grid.addWidget(_make_panel(title, px), i // 3, i % 3)
            except Exception as exc:
                self.log.append(f"Scale crop preview failed: {exc}")
                scale_grid.addWidget(QLabel(f"Scale crop preview failed: {exc}"), 0, 0)

        def _make_bordered_panel(title, pixmap, color):
            panel = _make_panel(title, pixmap)
            panel.setStyleSheet(f"border:2px solid {color}; border-radius:3px;")
            return panel

        def _build_distinct_grid():
            _clear_grid(dist_grid)
            if not self._ensure_local_model_loaded():
                dist_status.setText(
                    "No model loaded and no checkpoint file found — "
                    "start/resume training first.")
                return
            model = self.model
            thr = self.filter_pos_thresh_input.value()
            thr_max_raw = self.filter_pos_max_input.value()
            thr_max = thr_max_raw if thr_max_raw > 0 else None
            try:
                import numpy as np
                from dataset import make_eval_transform
                from model import backbone_img_size
                dev = next(model.parameters()).device
                tf = make_eval_transform(
                    img_size=backbone_img_size(getattr(model, "backbone_name", "swin_t")))

                @torch.inference_mode()
                def _emb(pil):
                    x = tf(pil).unsqueeze(0).to(dev)
                    e = model.encode_cluster_head(x)
                    return torch.nn.functional.normalize(e.float(), dim=-1)[0].cpu().numpy()

                drone_name = drone_path.name
                same = [p for p in dataset.pairs if Path(p[0]).name == drone_name]
                others = [p for p in dataset.pairs if Path(p[0]).name != drone_name]
                random.Random(0).shuffle(others)
                # negative reference pool
                neg = []
                for p in others[:64]:
                    try:
                        neg.append(_emb(Image.open(p[1]).convert("RGB")))
                    except Exception:
                        pass
                neg = np.array(neg) if neg else None

                a = _emb(drone_image)
                results = []
                for p in same:
                    try:
                        tv = _emb(Image.open(p[1]).convert("RGB"))
                    except Exception:
                        continue
                    sim_a = float(tv @ a)
                    avg_neg = float((neg @ tv).mean()) if neg is not None else 0.1
                    ratio = sim_a / max(avg_neg, 0.05)
                    results.append((Path(p[1]), ratio))
            except Exception as exc:
                dist_status.setText(f"Distinctiveness preview failed: {exc}")
                return

            if not results:
                dist_status.setText("No positives found for this anchor.")
                return
            results.sort(key=lambda x: -x[1])
            best = results[0][0]
            def _keep(pp, r):
                if thr_max is not None and r > thr_max:
                    return False      # trivially easy — no rescue
                if r >= thr:
                    return True
                return pp == best     # rescue only from drop_low
            kept   = [(pp, r) for pp, r in results if _keep(pp, r)]
            removed = [(pp, r) for pp, r in results if not _keep(pp, r)]
            dist_status.setText(
                f"{drone_name}: {len(results)} positives  |  "
                f"keep {len(kept)} (ratio ≥ {thr:g}), remove {len(removed)}  "
                f"(best always kept)")
            dist_status.setStyleSheet("color:#333;")

            col = 0
            for pp, r in kept:
                try:
                    px = self.pil_to_pixmap(Image.open(pp).convert("RGB"))
                except Exception:
                    continue
                dist_grid.addWidget(
                    _make_bordered_panel(f"keep {r:.2f}\n{pp.name}", px, "#27ae60"), 0, col)
                col += 1
            col = 0
            for pp, r in removed:
                try:
                    px = self.pil_to_pixmap(Image.open(pp).convert("RGB"))
                except Exception:
                    continue
                dist_grid.addWidget(
                    _make_bordered_panel(f"remove {r:.2f}\n{pp.name}", px, "#c0392b"), 1, col)
                col += 1

        def _build_neg_grid():
            _clear_grid(neg_grid)
            cs = getattr(self, 'last_cluster_sampling', None)
            if cs is None:
                # Remote runs save cluster_sampling.pt server-side every
                # rebuild, but the pull only triggers on the streamed event —
                # a GUI (re)started mid-run never saw it. The file persists in
                # the run dir, so fetch it directly instead of demanding a
                # from-scratch local Build Cluster Data.
                host = getattr(self, "_last_remote_host", None)
                run_dir = getattr(self, "_last_remote_run_dir", None)
                if host and run_dir:
                    self._pull_remote_cluster_sampling_async(
                        host, f"{run_dir}/cluster_sampling.pt")
                    neg_status.setText(
                        "No cluster data in memory — pulling it from the last "
                        "remote run in the background (a few minutes on a slow "
                        "link). Press 'Refresh previews' after the '[remote] "
                        "Cluster data synced' message appears in the training "
                        "log. (Or run Build Cluster Data to rebuild locally.)")
                else:
                    neg_status.setText("No cluster data — run Build Cluster Data first.")
                return

            # Find which cluster contains orig_idx
            cluster_id = None
            for cid, members in cs["cluster_members"].items():
                if orig_idx in members:
                    cluster_id = cid
                    break
            if cluster_id is None:
                neg_status.setText(
                    f"Sample #{orig_idx} not found in cluster data — rebuild clusters."
                )
                return

            # MES exclusion — use the SAME helper as the training negative sampler
            # (trainer.compute_mes_exclude_set) so the preview never diverges from
            # what training actually excludes: self, same tile, twin/adjacent tiles
            # (Chebyshev≤2, pair-ID normalized), same drone, and GPS-proximate pairs.
            from trainer import compute_mes_exclude_set
            cluster_pool = cs["cluster_members"][cluster_id]
            exclude_set = compute_mes_exclude_set(orig_idx, cs, cluster_pool)
            emb_lookup = cs["embedding_lookup"]
            anchor_emb = emb_lookup.get(orig_idx)
            if anchor_emb is not None:
                anchor_emb = torch.as_tensor(anchor_emb, dtype=torch.float32)

            def _rank(members):
                """Members minus MES-excluded, hardest (most similar) first."""
                kept = [m for m in members if m not in exclude_set]
                if anchor_emb is None:
                    return kept
                valid = [(m, emb_lookup[m]) for m in kept if m in emb_lookup]
                if not valid:
                    return kept
                idxs, embs = zip(*valid)
                embs = torch.stack([torch.as_tensor(e, dtype=torch.float32) for e in embs])
                order = torch.argsort(torch.mv(embs, anchor_emb), descending=True)
                return [idxs[int(k)] for k in order]

            # Primary negatives: same cluster (6). Matches training's primary_count.
            primary_pool = _rank(cluster_pool)

            # Global negatives: 3 nearest clusters by centroid cosine similarity —
            # identical selection to sample_structured_negative_indices, so the
            # preview never diverges from what training samples.
            nearest_pool = []
            near_ids = []
            centroids = cs.get("centroids")
            if centroids is not None and anchor_emb is not None:
                centroids = torch.as_tensor(centroids, dtype=torch.float32)
                csims = torch.nn.functional.cosine_similarity(
                    anchor_emb.unsqueeze(0), centroids, dim=1)
                k = min(3, centroids.shape[0] - 1)
                _, top = torch.topk(csims, k + 1)
                near_ids = [int(c) for c in top if int(c) != cluster_id][:k]
                near_members = []
                for nc in near_ids:
                    near_members.extend(cs["cluster_members"].get(nc, []))
                nearest_pool = _rank(near_members)

            # 6 from same cluster + 3 from nearest clusters (falls back gracefully:
            # if the same cluster is emptied by MES, nearest-cluster negatives still show).
            tagged = ([(m, "same") for m in primary_pool]
                      + [(m, "near") for m in nearest_pool])

            # Stage-by-stage diagnostic — pinpoints WHERE candidates vanish
            # (stale cluster data, missing embeddings/centroids, or MES wipeout).
            max_pool_idx = max((int(m) for m in cluster_pool), default=-1)
            stale = max_pool_idx >= len(dataset.pairs)
            self.log.append(
                f"[Hard Neg Preview] cluster#{cluster_id}: pool={len(cluster_pool)}, "
                f"MES-excluded={len(exclude_set)}, primary={len(primary_pool)}, "
                f"near_ids={near_ids}→{len(nearest_pool)}, "
                f"anchor_emb={'ok' if anchor_emb is not None else 'MISSING'}, "
                f"centroids={'ok' if centroids is not None else 'MISSING'}, "
                f"dataset_pairs={len(dataset.pairs)}, max_pool_idx={max_pool_idx}"
                + (" | STALE cluster data — built on a different dataset; retrain or "
                   "rebuild clusters!" if stale else ""))

            if not tagged:
                neg_status.setText(
                    f"Cluster #{cluster_id}: 0 candidates after MES "
                    f"(pool={len(cluster_pool)}, excluded={len(exclude_set)}, "
                    f"anchor_emb={'ok' if anchor_emb is not None else 'missing'}"
                    + (", STALE cluster data" if stale else "") + ") — see log.")
                neg_status.setStyleSheet("color:#c00;font-weight:bold;")
                return

            neg_status.setText(
                f"Cluster #{cluster_id}  |  {len(primary_pool)} same-cluster + "
                f"{len(nearest_pool)} nearest-cluster (#{near_ids}) candidates after MES"
            )
            neg_status.setStyleSheet("color:#333;")

            shown = 0
            cap   = {"same": 6, "near": 3}
            count = {"same": 0, "near": 0}
            seen_locs = set()  # Track by location, not by tile (allows different altitudes)
            # Location identity is dataset-layout dependent: loc-per-directory
            # datasets (U-1652/DenseUAV/SUES) use the parent dir name; flat
            # tiled galleries (game4loc VisLoc/GTA: every tile in one
            # directory, location encoded in the FILENAME) must use the tile
            # stem — parent.name would be the constant "satellite", collapsing
            # every candidate into one "seen" location (seen_tile=898 bug).
            _flat_tiles = self.dataset_type_input.currentData() == "game4loc"
            if _flat_tiles:
                anchor_loc_id = Path(dataset.pairs[orig_idx][1]).stem
            else:
                anchor_loc_id = drone_path.parent.name
            debug_skipped = {"out_of_range": 0, "seen_tile": 0, "same_loc": 0, "load_fail": 0}
            for neg_idx, src in tagged:
                if shown >= N_NEG or count[src] >= cap[src]:
                    continue
                if neg_idx >= len(dataset.pairs):
                    debug_skipped["out_of_range"] += 1
                    continue
                neg_sat_path = Path(dataset.pairs[neg_idx][1])
                neg_loc_id = (neg_sat_path.stem if _flat_tiles
                              else neg_sat_path.parent.name)

                # Exclude same location ID (different altitudes are not hard negatives)
                if neg_loc_id == anchor_loc_id:
                    debug_skipped["same_loc"] += 1
                    continue
                if neg_loc_id in seen_locs:
                    debug_skipped["seen_tile"] += 1
                    continue
                seen_locs.add(neg_loc_id)
                try:
                    img = Image.open(neg_sat_path).convert("RGB")
                except Exception:
                    debug_skipped["load_fail"] += 1
                    continue
                title = f"Neg #{shown+1} [{src}]\n{neg_loc_id}\n{neg_sat_path.name}"
                neg_grid.addWidget(
                    _make_panel(title, self.pil_to_pixmap(img)),
                    shown // 3, shown % 3,
                )
                count[src] += 1
                shown += 1

            msg = (f"[Hard Neg Preview] Showed {shown}/{N_NEG} "
                   f"(same={count['same']}, near={count['near']}). "
                   f"Skipped: same_loc={debug_skipped['same_loc']}, "
                   f"seen_tile={debug_skipped['seen_tile']}, load_fail={debug_skipped['load_fail']}, "
                   f"out_of_range={debug_skipped['out_of_range']}")
            self.log.append(msg)
            if shown < N_NEG:
                neg_status.setText(
                    f"Cluster #{cluster_id}  |  same={count['same']}/6 near={count['near']}/3  |  "
                    f"Showing {shown}/{N_NEG} (skipped: same_loc={debug_skipped['same_loc']}, "
                    f"seen_tile={debug_skipped['seen_tile']}, load_fail={debug_skipped['load_fail']})"
                )
                neg_status.setStyleSheet("color:#c00;font-weight:bold;")

        def _current_aug():
            return {k: w.value() for k, w in ctrl.items()}

        def _refresh():
            aug = _current_aug()
            _build_pos_grid(make_train_transform(aug=aug))
            _build_scale_grid()
            _build_distinct_grid()
            _build_neg_grid()

        def _apply():
            aug = _current_aug()
            import dataset as _ds
            _ds.DEFAULT_AUG.update(aug)
            if hasattr(dataset, "transform"):
                dataset.transform = make_train_transform(aug=aug)
            self.log.append(
                f"Augmentation applied: contrast={aug['contrast']:.2f} "
                f"brightness={aug['brightness']:.2f} saturation={aug['saturation']:.2f} "
                f"hue={aug['hue']:.2f} dropout_p={aug['dropout_p']:.2f} "
                f"holes={aug['min_holes']}-{aug['max_holes']} "
                f"frac={aug['min_frac']:.2f}-{aug['max_frac']:.2f}"
            )

        refresh_btn.clicked.connect(_refresh)
        save_btn.clicked.connect(_apply)

        # side-by-side: controls left, scroll right
        body = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(ctrl_group)
        left.addLayout(btn_row)
        left.addStretch()
        body.addLayout(left, 0)
        body.addWidget(scroll, 1)

        root_layout.addLayout(body)
        dialog.setLayout(root_layout)

        # Build anchor + positive, scale crops, and hard negatives initially;
        # distinctiveness (needs a loaded model) waits for "Refresh previews".
        _build_pos_grid(make_train_transform())
        _build_scale_grid()
        _build_neg_grid()
        dialog.exec_()

    def pil_to_pixmap(self, image):
        image = image.convert("RGB")
        data = image.tobytes("raw", "RGB")
        qimage = QImage(data, image.width, image.height, image.width * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimage)

    def tensor_to_pixmap(self, tensor):
        tensor = tensor.detach().cpu()
        if tensor.ndim != 3:
            raise ValueError(f"Expected CHW image tensor, got shape {tuple(tensor.shape)}")
        # Undo ImageNet normalization so the image is visible
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor * std + mean).clamp(0.0, 1.0)
        array = (tensor.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        height, width, _ = array.shape
        data = array.tobytes()
        qimage = QImage(data, width, height, width * 3, QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimage)

    def checkpoint_path_for(self):
        override = self.checkpoint_path_input.text().strip()
        if override:
            return Path(override)
        backbone = self.training_params.get("backbone", "swin_t")
        return Path("checkpoints") / f"latest_{backbone}.pt"

    def _load_last_remote_run(self):
        """(host, run_dir) of the last remote run this machine launched, or
        (None, None). Persisted to disk (not just an instance attribute) so
        Reattach still works after restarting the GUI — the exact scenario
        that motivated it: a stuck local launch/monitor connection may force
        a restart while the remote job itself keeps training untouched."""
        try:
            data = json.loads(self._last_remote_run_file.read_text(encoding="utf-8"))
            return data.get("host"), data.get("run_dir")
        except Exception:
            return None, None

    def _save_last_remote_run(self, host, run_dir):
        try:
            self._last_remote_run_file.write_text(
                json.dumps({"host": host, "run_dir": run_dir}), encoding="utf-8")
        except Exception:
            pass   # best-effort; Reattach just won't survive a restart if this fails

    # Autosave runs once per epoch (see _make_epoch_end_callback), and each
    # backup is a full model+optimizer checkpoint (~1GB for Swin-B) — with no
    # limit, a single 40-epoch run alone adds ~40GB to checkpoints/olds/, and
    # it never gets cleaned up across runs/days (observed: 89 files, 85GB,
    # nearly filling the SSD). Keep only the N most recent per checkpoint name.
    CHECKPOINT_BACKUP_RETENTION = 5

    def write_checkpoint(self, path):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "losses": self.losses,
            "loss_records": self.loss_records,
            "dataset_root": DATASET_ROOT,
            "training_params": self.training_params,
            "kmeans_centroids": self.kmeans_centroids,
            "retrieval_centroids": self.retrieval_centroids,
            "last_training_mode": self.training_params.get("training_mode"),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            backup_dir = path.parent / "olds"
            backup_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{path.stem}_{ts}{path.suffix}"
            shutil.copy2(path, backup_path)
            self.log_message.emit(f"Checkpoint backed up to {backup_path.name}")
            self._prune_old_backups(backup_dir, path.stem, path.suffix)
        torch.save(checkpoint, path)

    def _prune_old_backups(self, backup_dir: Path, stem: str, suffix: str):
        """Delete all but the CHECKPOINT_BACKUP_RETENTION most recent timestamped
        backups for this checkpoint name. Matches the EXACT
        '{stem}_YYYYMMDD_HHMMSS{suffix}' pattern (not a loose glob) so pruning
        "latest_swin_b"'s backups can't accidentally sweep up
        "latest_swin_b_nohead_all_h_full_strength"'s — several checkpoint
        variants in this project share a common name prefix.
        """
        pattern = re.compile(rf"^{re.escape(stem)}_\d{{8}}_\d{{6}}{re.escape(suffix)}$")
        backups = sorted(p for p in backup_dir.iterdir() if pattern.match(p.name))
        excess = backups[:-self.CHECKPOINT_BACKUP_RETENTION] if len(backups) > self.CHECKPOINT_BACKUP_RETENTION else []
        for old in excess:
            try:
                old.unlink()
                self.log_message.emit(f"Pruned old backup {old.name}")
            except OSError as exc:
                self.log_message.emit(f"Could not prune {old.name}: {exc}")

    def _save_train_history(self, status, start_time, error_message=None, error_traceback=None):
        history_path = Path("train_history_runs.json")
        runs = []
        if history_path.exists():
            try:
                runs = json.loads(history_path.read_text(encoding="utf-8"))
            except Exception:
                runs = []

        params = self.training_params or {}
        losses = self.losses or []
        loss_records = self.loss_records or []
        epochs_completed = max((r["epoch"] for r in loss_records), default=0)
        last_epoch_losses = [r["total_loss"] for r in loss_records if r["epoch"] == epochs_completed] if loss_records else []

        run_number = max((r.get("run_number", 0) for r in runs), default=0) + 1
        self._last_train_run_number = run_number
        entry = {
            "run_number": run_number,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_s": round(time.perf_counter() - start_time, 1),
            "status": status,
            "training_label": params.get("training_label", ""),
            "dataset_root": params.get("dataset_root", ""),
            "dataset_type": params.get("dataset_type", ""),
            "epochs": params.get("epochs", 0),
            "epochs_completed": epochs_completed,
            "cluster_count": params.get("cluster_count", 0),
            "batch_size": params.get("batch_size", 0),
            "learning_rate": params.get("learning_rate", 0),
            "weight_decay": params.get("weight_decay", 0),
            "backbone": params.get("backbone", "swin_t"),
            "training_mode": params.get("training_mode", ""),
            "resume_checkpoint": params.get("resume_checkpoint", False),
            "unfreeze_backbone_layers": params.get("unfreeze_backbone_layers", 0),
            "total_steps": len(loss_records),
            "final_loss": round(losses[-1], 6) if losses else None,
            "min_loss": round(min(losses), 6) if losses else None,
            "mean_loss_last_epoch": round(sum(last_epoch_losses) / len(last_epoch_losses), 6) if last_epoch_losses else None,
            **({"error_message": error_message} if error_message else {}),
            **({"error_traceback": error_traceback} if error_traceback else {}),
        }
        runs.append(entry)
        history_path.write_text(json.dumps(runs, indent=2), encoding="utf-8")
        self.log_message.emit(f"Training run #{run_number} saved to {history_path}.")
        try:
            from post_training_eval import render_train_history_html
            html = render_train_history_html(runs)
            Path("train_history.html").write_text(html, encoding="utf-8")
            self.log_message.emit("train_history.html updated.")
        except Exception as exc:
            self.log_message.emit(f"train_history.html update failed: {exc}")

    def _make_epoch_end_callback(self, params, device):
        """Return a callback(epoch, model) that saves checkpoint + optionally runs VisLoc/GTA eval."""
        def _cb(epoch, model):
            # Quick checkpoint save
            self.autosave_checkpoint()

            # Skip expensive evaluation if stop requested
            if self.stop_event.is_set():
                return

            # Periodic lightweight DenseUAV eval — independent of the GTA/VisLoc
            # "epoch_eval" option below, since training loss alone can plateau
            # while retrieval quality keeps improving (see project memory).
            if (params.get("quick_eval_denseuav", False)
                    and params.get("dataset_type") == "denseuav"
                    and epoch % max(1, params.get("quick_eval_every_n", 10)) == 0):
                try:
                    from general_eval_gui import quick_eval_denseuav
                    self.log_message.emit(f"[Epoch {epoch}] Quick DenseUAV eval...")
                    quick_eval_denseuav(
                        model, device, params.get("denseuav_train_root", ""),
                        batch_size=min(64, params.get("batch_size", 32)),
                        stop_event=self.stop_event,
                        log=self.log_message.emit)
                except Exception as exc:
                    self.log_message.emit(f"[Epoch {epoch}] Quick eval error: {exc}")

            if not params.get("epoch_eval", False):
                return   # per-epoch eval disabled (checkpoint still saved above)
            test_json = params.get("gta_test_json", "").strip()
            if not test_json or params.get("dataset_type") != "game4loc":
                return
            try:
                import json as _json
                from pathlib import Path as _Path
                from dataset import make_eval_transform as _make_tf
                from visloc_eval import evaluate_split as _eval_split

                data_root = _Path(params["gta_data_root"])
                meta_path = data_root / test_json
                if not meta_path.exists():
                    self.log_message.emit(f"[Epoch {epoch}] Test JSON not found: {meta_path}")
                    return

                sample = _json.load(open(meta_path, encoding="utf-8"))[0]
                ds_name = "visloc" if "drone_loc_lat_lon" in sample else "gta_uav"

                self.log_message.emit(f"[Epoch {epoch}] Evaluating {test_json} ({ds_name}, two_stage_cluster + cluster_mah)...")
                from model import backbone_img_size as _bimg
                model.eval()
                result = _eval_split(
                    model, device, _make_tf(img_size=_bimg(getattr(model, "backbone_name", "swin_t"))),
                    data_root, test_json, "satellite", "pos",
                    batch_size=128,
                    dataset_name=ds_name,
                    search_mode="two_stage_cluster",
                    scoring_mode="cluster_mah",
                    cluster_count=16,
                    cluster_mah_eps=0.5,
                )
                model.train()

                if result:
                    r1   = result["recall_at_1"] * 100
                    r5   = result["recall_at_5"] * 100
                    sdm1 = result["sdm"].get("sdm@1", 0.0)
                    dis1 = result["dis"].get("dis@1", 0.0)
                    self.log_message.emit(
                        f"[Epoch {epoch}] R@1={r1:.2f}%  R@5={r5:.2f}%  "
                        f"SDM@1={sdm1:.4f}  Dis@1={dis1:.1f}"
                    )
            except Exception as exc:
                self.log_message.emit(f"[Epoch {epoch}] Eval error: {exc}")
        return _cb

    def autosave_checkpoint(self):
        with self.model_lock:
            if self.model is None:
                return
            path = self.checkpoint_path_for()
            self.write_checkpoint(path)
        self.log_message.emit(f"Autosaved checkpoint to {path}")

    def load_existing_checkpoint(self, model, optimizer, device, params):
        checkpoint_path = self.checkpoint_path_for()
        if not checkpoint_path.exists():
            self.log_message.emit(
                f"No existing checkpoint found at {checkpoint_path}; starting from new weights."
            )
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            saved_params = checkpoint.get("training_params", {})
            saved_embed_dim = saved_params.get("embed_dim")
            saved_cluster_count = saved_params.get("cluster_count")
            if saved_embed_dim is not None and saved_embed_dim != CLUSTER_DESCRIPTOR_DIM:
                self.log_message.emit(
                    "Existing checkpoint ignored because embedding dim differs "
                    f"({saved_embed_dim} != {CLUSTER_DESCRIPTOR_DIM})."
                )
                return False

            state_dict = self.filter_compatible_state_dict(
                model,
                checkpoint["model_state_dict"],
                saved_cluster_count,
            )
            n_unfreeze = params.get("unfreeze_backbone_layers", 0)
            if params.get("freeze_backbone", True) and n_unfreeze == 0:
                # Fully frozen backbone: skip backbone weights so pretrained weights are kept.
                state_dict_to_load = {
                    key: value
                    for key, value in state_dict.items()
                    if not key.startswith("backbone.")
                }
                missing, unexpected = model.load_state_dict(state_dict_to_load, strict=False)
                skipped = sum(1 for key in checkpoint["model_state_dict"] if key.startswith("backbone."))
                loaded_cluster = sum(1 for key in state_dict_to_load if key.startswith("cluster_head."))
                self.log_message.emit(
                    "Loaded checkpoint heads only; "
                    f"cluster_head tensors={loaded_cluster}, "
                    f"skipped {skipped} backbone tensors (backbone frozen, keeping pretrained weights)."
                )
                non_backbone_missing = [k for k in missing if not k.startswith("backbone.")]
                if non_backbone_missing:
                    self.log_message.emit(
                        f"Missing checkpoint keys initialized from new model: {non_backbone_missing}"
                    )
                if unexpected:
                    self.log_message.emit(f"Unexpected checkpoint keys ignored: {len(unexpected)}")
            else:
                # Partially or fully unfrozen backbone: load all weights including backbone.
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                loaded_cluster = sum(1 for key in state_dict if key.startswith("cluster_head."))
                loaded_backbone = sum(1 for key in state_dict if key.startswith("backbone."))
                self.log_message.emit(
                    f"Loaded checkpoint weights; backbone tensors={loaded_backbone}, "
                    f"cluster_head tensors={loaded_cluster}, "
                    f"missing={len(missing)}, unexpected={len(unexpected)}."
                )

            last_mode = checkpoint.get("last_training_mode")
            current_mode = params.get("training_mode")

            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                if last_mode != current_mode:
                    self.log_message.emit(
                        f"Skipped optimizer state: training mode changed from "
                        f"{last_mode!r} to {current_mode!r} (momentum buffers would have "
                        f"wrong shapes for the current head)."
                    )
                else:
                    try:
                        optimizer.load_state_dict(optimizer_state)
                        self.log_message.emit("Loaded optimizer state from checkpoint.")
                    except Exception as exc:
                        self.log_message.emit(f"Skipped optimizer state because it is incompatible: {exc}")

            saved_centroids = checkpoint.get("kmeans_centroids")
            if saved_centroids is not None:
                self.kmeans_centroids = saved_centroids.cpu()
                self.log_message.emit(
                    f"Loaded K-means centroids from checkpoint "
                    f"(shape={tuple(saved_centroids.shape)})."
                )
            else:
                self.kmeans_centroids = None
            saved_retrieval_centroids = checkpoint.get("retrieval_centroids")
            if saved_retrieval_centroids is not None:
                self.retrieval_centroids = saved_retrieval_centroids.cpu()
                self.log_message.emit(
                    f"Loaded retrieval centroids from checkpoint "
                    f"(shape={tuple(saved_retrieval_centroids.shape)})."
                )
            else:
                self.retrieval_centroids = None

            self.log_message.emit(f"Loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as exc:
            self.log_message.emit(f"Could not load existing checkpoint: {exc}")
            return False

    def filter_compatible_state_dict(self, model, state_dict, saved_cluster_count=None):
        model_state = model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            migrated_keys = []
            if key.startswith("global_head."):
                suffix = key[len("global_head."):]
                migrated_keys = [f"cluster_head.{suffix}"]
            elif key.startswith("cluster_heads."):
                skipped.append(key)
                continue
            else:
                migrated_keys = [key]
            migrated = False
            for migrated_key in migrated_keys:
                if migrated_key in model_state and model_state[migrated_key].shape == value.shape:
                    compatible[migrated_key] = value
                    migrated = True
            if migrated:
                continue
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
            else:
                skipped.append(key)
        if skipped:
            self.log_message.emit(
                f"Skipped {len(skipped)} incompatible checkpoint tensors "
                f"(likely old cluster-head count {saved_cluster_count} -> {NUM_CLUSTERS})."
            )
        return compatible

    def freeze_backbone(self, model):
        n_unfreeze = self.training_params.get("unfreeze_backbone_layers", 0)
        for param in model.backbone.parameters():
            param.requires_grad = False
        model.backbone.eval()

        # Per-architecture stage container: Swin/SwinV2 -> .layers,
        # ConvNeXt -> .stages, ViT/DINOv2 -> .blocks. The old layers-only
        # check made unfreeze a SILENT no-op on ConvNeXt (zero trainable
        # params -> backward() crash on no-head variants, runs #226/#227).
        if hasattr(model.backbone, "layers") and not hasattr(model.backbone, "blocks"):
            stages = list(model.backbone.layers)
        elif hasattr(model.backbone, "stages"):
            stages = list(model.backbone.stages)
        elif hasattr(model.backbone, "blocks"):
            stages = list(model.backbone.blocks)
        else:
            stages = []
        if n_unfreeze > 0 and stages:
            num_stages = len(stages)
            tail_modules = stages[max(0, num_stages - n_unfreeze):]
            if hasattr(model.backbone, "norm"):
                tail_modules.append(model.backbone.norm)
            for mod in tail_modules:
                mod.train()
                for param in mod.parameters():
                    param.requires_grad = True
            unfrozen = sum(p.numel() for mod in tail_modules for p in mod.parameters())
            self.log_message.emit(
                f"Backbone: last {n_unfreeze} stage(s) unfrozen "
                f"({self.format_params(unfrozen)} trainable backbone params)."
            )

        frozen = sum(p.numel() for p in model.backbone.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log_message.emit(
            f"Backbone frozen params={self.format_params(frozen)}, "
            f"total trainable params={self.format_params(trainable)}."
        )

    def configure_trainable_heads(self, model):
        has_head = getattr(model, "_use_head", True) and hasattr(model, "cluster_head")
        if has_head:
            for param in model.cluster_head.parameters():
                param.requires_grad = True
        trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
        self.log_message.emit(
            f"Trainable heads configured: cluster_head={'True' if has_head else 'N/A (no head)'},"
            f" trainable params={self.format_params(trainable)}."
        )

    def _make_dead_cluster_preview_callback(self, dataset):
        def callback(dead_info, epoch):
            dead_sample_indices = dead_info.get("dead_sample_indices", [])
            selected = random.sample(dead_sample_indices, min(16, len(dead_sample_indices))) if dead_sample_indices else []
            dead_image_paths = []
            for idx in selected:
                if idx < len(dataset.pairs):
                    anchor_path = dataset.pairs[idx][0]
                    dead_image_paths.append(str(anchor_path))
            cluster_sample_indices = dead_info.get("cluster_sample_indices", {})
            cluster_image_paths = {}
            for cid, indices in cluster_sample_indices.items():
                sampled = random.sample(indices, min(16, len(indices))) if indices else []
                paths = []
                for idx in sampled:
                    if idx < len(dataset.pairs):
                        anchor_path = dataset.pairs[idx][0]
                        paths.append(str(anchor_path))
                cluster_image_paths[cid] = paths
            self.dead_cluster_preview_updated.emit(
                {
                    "dead_cluster_ids": list(dead_info.get("dead_cluster_ids", set())),
                    "image_paths": dead_image_paths,
                    "cluster_variances": dead_info.get("cluster_variances", {}),
                    "variance_threshold": dead_info.get("variance_threshold", 0.0),
                    "global_mean_var": dead_info.get("global_mean_var", 0.0),
                    "cluster_image_paths": cluster_image_paths,
                    "cluster_positive_probability": dead_info.get("cluster_positive_probability", {}),
                    "cluster_training_pos_prob": dead_info.get("cluster_training_pos_prob", {}),
                },
                epoch,
            )
        return callback

    def _make_cluster_stats_callback(self, dataset):
        def callback(stats, epoch):
            cluster_image_paths = {}
            for cid, indices in stats.get("cluster_sample_indices", {}).items():
                paths = []
                for idx in indices:
                    if idx < len(dataset.pairs):
                        anchor_path = dataset.pairs[idx][0]
                        paths.append(str(anchor_path))
                cluster_image_paths[int(cid)] = paths
            merged = dict(stats)
            merged["cluster_image_paths"] = cluster_image_paths
            self.cluster_stats_updated.emit(merged, epoch)
        return callback

    def on_centroids_updated(self, centroids):
        self.kmeans_centroids = centroids.cpu()

    def on_retrieval_centroids_updated(self, centroids):
        self.retrieval_centroids = centroids.cpu()

    def on_cluster_stats_updated(self, stats):
        cluster_loss = {int(k): v for k, v in stats.get("cluster_loss", {}).items()}
        if not cluster_loss:
            return
        self._cluster_table_data["cluster_loss"] = cluster_loss
        self._rebuild_cluster_excl_table()
        cluster_image_paths = {int(k): v for k, v in stats.get("cluster_image_paths", {}).items()}
        if cluster_image_paths:
            self.cluster_image_paths = cluster_image_paths
        self.cluster_variance_canvas.draw_idle()

    def on_dead_cluster_preview_updated(self, preview_info, epoch):
        self._cluster_table_data["cluster_variances"] = {
            int(k): v for k, v in preview_info.get("cluster_variances", {}).items()}
        dead_cluster_ids = set(preview_info.get("dead_cluster_ids", []))
        n_dead = len(dead_cluster_ids)
        image_paths = preview_info.get("image_paths", [])
        cluster_variances = preview_info.get("cluster_variances", {})
        variance_threshold = preview_info.get("variance_threshold", 0.0)
        global_mean_var = preview_info.get("global_mean_var", 0.0)
        self.cluster_image_paths = {
            int(k): v for k, v in preview_info.get("cluster_image_paths", {}).items()
        }
        self._last_dead_cluster_ids = set(int(i) for i in dead_cluster_ids)
        self.dead_cluster_status_label.setText(
            f"Epoch {epoch} — {n_dead} dead cluster(s) (threshold {variance_threshold:.2e}). "
            "Click a bar to browse cluster samples."
        )
        self.cluster_variance_ax.clear()
        if cluster_variances:
            cids = sorted(cluster_variances.keys())
            variances = [cluster_variances[c] for c in cids]
            colors = ["#d62728" if c in dead_cluster_ids else "#1f77b4" for c in cids]
            self.cluster_variance_ax.bar(cids, variances, color=colors, width=0.8)
            if variance_threshold > 0:
                self.cluster_variance_ax.axhline(
                    variance_threshold, color="#ff7f0e", linestyle="--", linewidth=1.2,
                    label=f"10% threshold ({variance_threshold:.2e})"
                )
                self.cluster_variance_ax.axhline(
                    global_mean_var, color="#2ca02c", linestyle=":", linewidth=1.0,
                    label=f"cluster mean ({global_mean_var:.2e})"
                )
                self.cluster_variance_ax.legend(fontsize=8)
        self.cluster_variance_ax.set_title(f"Per-cluster descriptor variance — Epoch {epoch}")
        self.cluster_variance_ax.set_xlabel("Cluster ID")
        self.cluster_variance_ax.set_ylabel("Mean per-dim variance")
        self.cluster_variance_figure.tight_layout()
        self.cluster_variance_canvas.draw_idle()
        self._render_image_grid(image_paths)

    def _render_image_grid(self, paths):
        while self.dead_cluster_grid.count():
            item = self.dead_cluster_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        cols = 4
        for i, path in enumerate(paths):
            container = QWidget()
            container_layout = QVBoxLayout()
            container_layout.setContentsMargins(2, 2, 2, 2)
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap(path)
            if not pixmap.isNull():
                img_label.setPixmap(pixmap.scaled(160, 160, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img_label.setText("(load error)")
            name_label = QLabel(Path(path).name)
            name_label.setAlignment(Qt.AlignCenter)
            name_label.setWordWrap(True)
            container_layout.addWidget(img_label)
            container_layout.addWidget(name_label)
            container.setLayout(container_layout)
            container.setToolTip(path)
            self.dead_cluster_grid.addWidget(container, i // cols, i % cols)

    def on_variance_bar_clicked(self, event):
        if event.inaxes is not self.cluster_variance_ax or event.xdata is None:
            return
        cluster_id = int(round(event.xdata))
        paths = self.cluster_image_paths.get(cluster_id)
        if paths is None:
            return
        kind = "dead" if cluster_id in self._last_dead_cluster_ids else "live"
        self.dead_cluster_status_label.setText(
            f"Cluster {cluster_id} ({kind}) — {len(paths)} sample image(s). "
            "Click another bar to switch cluster."
        )
        self._render_image_grid(paths)

    def train_loop(self):
        _train_start   = time.perf_counter()
        _MAX_RETRIES   = 3
        _RECOVERY_WAIT = 15          # seconds to wait after CUDA crash
        _micro_batch   = self.training_params.get("microbatch_size", 8)
        _device_str    = self.training_params.get("device", "cpu")

        for _attempt in range(_MAX_RETRIES + 1):
            self._train_loop_attempt(_train_start, _attempt, _micro_batch)
            # _train_loop_attempt either returns (success/stop/non-cuda-fail)
            # or sets self._cuda_retry_needed = True for CUDA errors
            if not getattr(self, "_cuda_retry_needed", False):
                return
            # CUDA recovery: reduce micro-batch, wait, reset GPU
            _micro_batch = max(1, _micro_batch // 2)
            self.log_message.emit(
                f"[Auto-recovery] Waiting {_RECOVERY_WAIT}s for GPU driver recovery…"
            )
            time.sleep(_RECOVERY_WAIT)
            try:
                torch.cuda.empty_cache()
                torch.cuda.init()
                _ = torch.zeros(1, device=torch.device(_device_str))
                del _
                self.log_message.emit(
                    f"[Auto-recovery] GPU driver recovered. "
                    f"Resuming with micro-batch={_micro_batch} "
                    f"(attempt {_attempt + 2}/{_MAX_RETRIES + 1})."
                )
            except Exception as _ce:
                self.log_message.emit(f"[Auto-recovery] GPU reset failed: {_ce}. Giving up.")
                self._save_train_history("failed", _train_start,
                                         error_message="CUDA reset failed after crash",
                                         error_traceback=str(_ce))
                self.training_finished.emit("failed")
                return
        # All retries exhausted — should not normally reach here (handled inside attempt)
        if getattr(self, "_cuda_retry_needed", False):
            self.log_message.emit("[Auto-recovery] All retry attempts exhausted. Training failed.")
            self._save_train_history("failed", _train_start,
                                     error_message="CUDA recovery failed after max retries")
            self.training_finished.emit("failed")

    def _train_loop_attempt(self, _train_start, _attempt, _micro_batch):
        """Single training attempt; sets self._cuda_retry_needed on recoverable CUDA errors."""
        self._cuda_retry_needed = False
        try:
            from trainer import train, even_batch_size
            from model import SwinEmbedding, backbone_img_size
            from dataset import SatCropDataset
            from torch.utils.data import DataLoader
            import torch

            # On retries: resume from last checkpoint and skip completed epochs
            base_params  = self.training_params
            params = {
                **base_params,
                "microbatch_size": _micro_batch,
                **({"resume_checkpoint": True} if _attempt > 0 else {}),
            }
            device = torch.device(params["device"])
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            cluster_count = params["cluster_count"]
            backbone = params.get("backbone", "swin_t")
            img_size = backbone_img_size(backbone)
            model = SwinEmbedding(
                embed_dim=CLUSTER_DESCRIPTOR_DIM,
                pretrained=params.get("pretrained", True),
                num_clusters=cluster_count,
                backbone=backbone,
            ).to(device)
            self.freeze_backbone(model)
            self.configure_trainable_heads(model)
            optimizer = torch.optim.AdamW(
                (param for param in model.parameters() if param.requires_grad),
                lr=params["learning_rate"],
                weight_decay=params["weight_decay"],
            )
            if params["resume_checkpoint"]:
                self.load_existing_checkpoint(model, optimizer, device, params)
            else:
                self.log_message.emit("Checkpoint resume disabled; starting from initialized weights.")
                self.kmeans_centroids = None
                self.retrieval_cluster_stats = None
            with self.model_lock:
                self.model = model
                self.optimizer = optimizer
            model_size = self.model_size_from_model(model)
            self.model_created.emit(model_size)

            group_size = params.get("group_size", 1)
            gta_json = params.get("gta_json", "")
            ds_type = params.get("dataset_type", "")
            self.log_message.emit(f"DEBUG: dataset_type={repr(ds_type)}, dataset_root={repr(params.get('dataset_root'))}")
            if ds_type == "university1652":
                from dataset import University1652Dataset
                dataset = University1652Dataset(
                    train_root=params.get("u1652_train_root",
                                         r"D:\UAV_DATASET\university-1652\University-Release\train"),
                    group_size=group_size,
                    img_size=img_size,
                    augment=params.get("augment", True),
                )
                self.log_message.emit(
                    f"Loaded {len(dataset)} University-1652 D→S pairs "
                    f"(group_size={group_size}, augment={params.get('augment', True)})."
                )
            elif ds_type == "denseuav":
                from dataset import DenseUAVDataset
                dataset = DenseUAVDataset(
                    train_root=params.get("denseuav_train_root",
                                         r"D:\UAV_DATASET\DenseUAV\DenseUAV\train"),
                    group_size=group_size,
                    img_size=img_size,
                    augment=params.get("augment", True),
                    cross_altitude=params.get("denseuav_cross_altitude", True),
                    altitude_weight_tau=(None if params.get("denseuav_altitude_full_strength", False)
                                        else params.get("denseuav_altitude_weight_tau", 20.0)),
                )
                self.log_message.emit(
                    f"Loaded {len(dataset)} DenseUAV D→S pairs "
                    f"(group_size={group_size}, augment={params.get('augment', True)}, "
                    f"cross_altitude={params.get('denseuav_cross_altitude', True)})."
                )
            elif ds_type == "sues200":
                from dataset import Sues200Dataset
                dataset = Sues200Dataset(
                    train_root=params.get("sues200_train_root",
                                         r"D:\UAV_DATASET\SUES-200-split\train"),
                    group_size=group_size,
                    img_size=img_size,
                    augment=params.get("augment", True),
                )
                self.log_message.emit(
                    f"Loaded {len(dataset)} SUES-200 D→S pairs "
                    f"(group_size={group_size}, augment={params.get('augment', True)})."
                )
            elif ds_type == "game4loc":
                from dataset import GtaUavDataset
                # Optional: prune non-distinctive positives before training.
                # Uses the freshly-loaded (resumed/pretrained) model to embed, so
                # resume from a trained checkpoint for a meaningful ranking.
                if params.get("filter_positives"):
                    try:
                        from filter_positives import filter_outlier_positives
                        from dataset import make_eval_transform
                        _tf = make_eval_transform(img_size=img_size)
                        _out = "pairs_train_distinctive.json"
                        thr = params.get("filter_distinctive_min", 1.0)
                        thr_max_raw = params.get("filter_distinctive_max", 0.0)
                        thr_max = thr_max_raw if thr_max_raw > 0 else None
                        cap_str = f" ≤ {thr_max}" if thr_max else ""
                        self.log_message.emit(
                            f"Filtering positives (distinctive ≥ {thr}{cap_str}) → {_out} ...")
                        model.eval()
                        st = filter_outlier_positives(
                            params["gta_data_root"], gta_json, model, _tf, device,
                            metric="distinctive", distinctive_min=thr,
                            distinctive_max=thr_max,
                            min_positives=3, out_json=_out,
                            log_fn=lambda s: self.log_message.emit(str(s)))
                        model.train()
                        gta_json = _out
                        self.log_message.emit(
                            f"Distinctiveness filter: removed {st['tiles_removed']} "
                            f"of {st['tiles_before']} positives "
                            f"({st.get('queries_zeroed', 0)} anchors fully dropped) "
                            f"→ training on {_out}.")
                    except Exception as _fe:
                        self.log_message.emit(
                            f"WARN: distinctiveness filter failed ({_fe}); "
                            f"training on original JSON.")
                        gta_json = params.get("gta_json", "")
                dataset = GtaUavDataset(
                    data_root=params["gta_data_root"],
                    pairs_meta_file=gta_json,
                    mode=params.get("gta_mode", "pos_semipos"),
                    group_size=group_size,
                    img_size=img_size,
                    augment_positives=params.get("gta_augment_pos", False),
                    augment=params.get("augment", True),
                )
                self.log_message.emit(
                    f"Loaded {len(dataset)} GTA-UAV/VisLoc pairs from "
                    f"{params['gta_data_root']}/{gta_json} "
                    f"(mode={params.get('gta_mode','pos_semipos')}, group_size={group_size}, "
                    f"augment={params.get('augment', True)})."
                )
                _excl = frozenset(self.excluded_pair_orig_idx)
                if _excl:
                    before = len(dataset.samples)
                    dataset.samples = [p for p in dataset.samples if p[3] not in _excl]
                    self.log_message.emit(
                        f"Cluster exclusion: removed {before - len(dataset.samples)} "
                        f"of {before} pairs ({len(_excl)} excluded indices)."
                    )
            else:
                dataset = SatCropDataset(params["dataset_root"], group_size=group_size,
                                         img_size=img_size, augment=params.get("augment", True))
                self.log_message.emit(
                    f"Loaded {len(dataset)} training pairs from {params['dataset_root']} "
                    f"(group_size={group_size})."
                )
            self.log_message.emit(f"Using device: {device}")
            if (self.kmeans_centroids is not None
                    and self.kmeans_centroids.shape[0] != cluster_count):
                # Requested K changed vs the checkpoint (e.g. 8 -> 16): the
                # saved centroids can't seed faiss K-means at a different K —
                # discard them and recluster fresh at the new K.
                self.log_message.emit(
                    f"Saved K-means centroids ({self.kmeans_centroids.shape[0]} clusters) "
                    f"don't match requested Target clusters={cluster_count} — "
                    f"discarding them; fresh clustering at K={cluster_count}.")
                self.kmeans_centroids = None
            if self.kmeans_centroids is not None:
                self.log_message.emit(
                    f"Using saved K-means centroids as initial clustering: shape={tuple(self.kmeans_centroids.shape)}."
                )
            from dataset import loader_kwargs
            # persistent=False is required: hard-mining mutates the dataset between
            # epochs, and persistent workers would keep serving a stale pickled copy.
            loader = DataLoader(
                dataset,
                batch_size=even_batch_size(len(dataset), params["batch_size"]),
                shuffle=params["shuffle"],
                **loader_kwargs(device.type,
                                num_workers=params["num_workers"],
                                persistent=False),
            )
            self.log_message.emit(
                f"DataLoader workers: {loader.num_workers} "
                f"(override with UAV_NUM_WORKERS env var)")

            # Patch trainer module-level constants so UI values take effect immediately.
            import trainer as _trainer_mod
            _trainer_mod.STAGE1_LABEL_SMOOTHING = params.get("label_smoothing", 0.05)
            _trainer_mod.STAGE1_GROUP_SIZE = params.get("group_size", 1)

            # On recovery, skip epochs already completed in previous attempts.
            _epochs_done = max((r["epoch"] for r in self.loss_records), default=0)
            _epochs_left = max(1, params["epochs"] - _epochs_done)
            if _attempt > 0 and _epochs_done > 0:
                self.log_message.emit(
                    f"[Auto-recovery] Skipping {_epochs_done} completed epoch(s), "
                    f"running {_epochs_left} remaining."
                )

            # Optional fast clustering: load a frozen SimpleClusterCNN to drive
            # per-epoch clustering / negative sampling instead of the heavy backbone.
            _cluster_model = None
            _cm_ckpt = params.get("cluster_model_ckpt", "").strip()
            if _cm_ckpt:
                try:
                    from cluster_model import load_cluster_model
                    _cluster_model, _cm_cfg = load_cluster_model(_cm_ckpt, device=device)
                    self.log_message.emit(
                        f"Fast clustering: loaded cluster model {_cm_ckpt} "
                        f"({_cm_cfg.get('out_dim')}-D, in_size={_cm_cfg.get('in_size')}). "
                        f"Consistency loss auto-disabled; embeddings cached across epochs."
                    )
                except Exception as e:
                    self.log_message.emit(f"WARN: could not load cluster model ({e}); "
                                          f"falling back to backbone clustering.")
                    _cluster_model = None

            completed = train(
                model,
                loader,
                optimizer,
                device,
                epochs=_epochs_left,
                use_amp=params["use_amp"],
                cluster_count=cluster_count,
                cluster_every=params["cluster_every"],
                stop_event=self.stop_event,
                pause_event=self.pause_event,
                loss_callback=lambda metrics, step, epoch, batch_step: self.loss_recorded.emit(
                    metrics, step, epoch, batch_step
                ),
                cluster_callback=lambda stats, epoch: self.clusters_updated.emit(stats, epoch),
                centroids_callback=self.on_centroids_updated,
                model_lock=self.model_lock,
                training_mode=params["training_mode"],
                status_callback=lambda message: self.log_message.emit(message),
                train_microbatch_size=params["microbatch_size"],
                initial_centroids=None if params.get("auto_cluster_k") else self.kmeans_centroids,
                dead_clusters_callback=None,
                dead_cluster_preview_callback=self._make_dead_cluster_preview_callback(dataset),
                cluster_epoch_stats_callback=self._make_cluster_stats_callback(dataset),
                cluster_sampling_callback=lambda cs: self.cluster_sampling_ready.emit(cs),
                cluster_consistency_weight=params["cluster_consistency_weight"],
                negative_weight=params.get("negative_weight", 10.0),
                auto_k=params.get("auto_cluster_k", False),
                auto_k_callback=lambda k: self.auto_k_found.emit(k),
                epoch_end_callback=self._make_epoch_end_callback(params, device),
                cluster_model=_cluster_model,
                hard_mining=params.get("hard_mining", False),
                enable_clustering=params.get("enable_clustering", True),
            )

            self.autosave_checkpoint()

            if completed and params.get("auto_train_scale_head") and hasattr(self.model, "scale_head"):
                self.log_message.emit("\n── Auto-training scale head ──")
                from scale_head_trainer import train_scale_head
                train_scale_head(
                    model          = self.model,
                    data_root      = params["sh_data_root"],
                    dataset_type   = params.get("sh_dataset_type", "visloc"),
                    device         = device,
                    epochs         = params.get("sh_epochs", 10),
                    batch_size     = params.get("sh_batch_size", 32),
                    lr             = params.get("sh_lr", 1e-3),
                    weight_decay   = params.get("sh_weight_decay", 1e-4),
                    amp_enabled    = params.get("sh_amp", True),
                    status_callback= lambda m: self.log_message.emit(m),
                    stop_event     = self.stop_event,
                    model_lock     = self.model_lock,
                )
                self.autosave_checkpoint()
                self.log_message.emit("── Scale head training complete ──\n")

            status = "finished" if completed else "stopped"
            self._save_train_history(status, _train_start)
            self.training_finished.emit(status)
        except Exception as exc:
            import traceback as _tb
            tb = _tb.format_exc()
            _is_cuda = ("CUDA error" in str(exc) or "AcceleratorError" in str(exc)
                        or "out of memory" in str(exc).lower())
            if _is_cuda:
                self.log_message.emit(
                    f"[Auto-recovery] CUDA error: {exc}\n"
                    f"Will attempt recovery (micro-batch will be halved)."
                )
                self.autosave_checkpoint()
                self._cuda_retry_needed = True
            else:
                self.log_message.emit(f"Training error: {exc}\n{tb}")
                self._save_train_history("failed", _train_start, error_message=str(exc), error_traceback=tb)
                self.training_finished.emit("failed")

    # ── Remote training over SSH ─────────────────────────────────────────
    # Code files synced to the remote before every run (keeps remote in step
    # with local edits; a few hundred KB, so always pushed).
    REMOTE_CODE_FILES = ("dataset.py", "model.py", "trainer.py", "loss.py",
                         "clustering.py", "remote_train.py")
    def _ssh_run(self, args, timeout=30, check=True, retries=3, **kw):
        """subprocess.run wrapper for ssh/scp control commands.

        This SSH path (Windows OpenSSH client -> this specific host) has a
        real, recurring transient-failure rate — confirmed via `ssh -v`
        showing "Exceeded MaxStartups" (server-side cap on concurrent
        unauthenticated connections, tripped by our own back-to-back
        ssh/scp calls with no connection reuse) and separately "kex_exchange
        ...: Software caused connection abort" on otherwise-idle isolated
        commands. ControlMaster multiplexing was tried as the real fix but
        is unusable on this client (the control socket itself fails to
        establish every time, a Windows/NTFS colon-in-path issue with the
        MSYS2 ssh build — reverted). So: retry with backoff instead.
        Two failure shapes get retried:
          - a hang (TimeoutExpired)
          - ssh exiting 255 fast (its code for "client itself failed to
            connect/auth", distinct from the remote command's own exit code)
        A backoff sleep before each retry matters: an immediate back-to-back
        retry can land in the same MaxStartups window and fail again
        (observed once). Only use for idempotent commands."""
        kw.setdefault("capture_output", True)
        kw.setdefault("creationflags", _NO_WINDOW)
        kw.setdefault("stdin", subprocess.DEVNULL)
        last_exc = None
        for attempt in range(retries + 1):
            if attempt > 0:
                time.sleep(3.0 * attempt)
            try:
                result = subprocess.run(args, check=False, timeout=timeout, **kw)
            except subprocess.TimeoutExpired as e:
                last_exc = e
                self.log_message.emit(
                    f"[remote] {' '.join(args[:2])} command wedged "
                    f"({timeout}s timeout)"
                    + (" — retrying on a fresh connection..."
                       if attempt < retries else " — giving up."))
                continue
            if result.returncode == 255 and attempt < retries:
                self.log_message.emit(
                    f"[remote] {' '.join(args[:2])} ssh transport failed "
                    f"(exit 255) — retrying on a fresh connection...")
                last_exc = subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr)
                continue
            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, args, result.stdout, result.stderr)
            return result
        raise last_exc

    _REMOTE_ROOT_KEYS = {"sues200": "sues200_train_root",
                         "denseuav": "denseuav_train_root",
                         "university1652": "u1652_train_root",
                         "game4loc": "gta_data_root"}

    def _stream_remote_output(self, host, run_dir):
        """Yield lines from the detached remote run's output.log, reconnecting
        through network drops instead of giving up. The training process is
        launched detached (setsid+nohup) specifically so THIS is the only
        thing a dropped connection can kill — training keeps running on the
        server regardless, and we just need to catch back up on its log.

        Reconnects via `tail -n +{offset+1} -f` (skip already-seen lines,
        keep following). If the stream ends before the caller has broken out
        on a 'finished' event, that means either a network drop (reconnect)
        or the remote process genuinely died (an unhandled crash bypassing
        remote_train.py's own try/except, e.g. an OOM-kill) — the two are
        told apart with a `pgrep` check: still running -> reconnect; gone
        with no 'finished' ever seen -> stop retrying and report it."""
        offset = 0
        attempt = 0
        max_attempts = 60
        while True:
            if self.stop_event.is_set():
                return
            proc = subprocess.Popen(
                # -F (not -f): retries if the file doesn't exist yet, closing
                # the narrow race between the detached launch returning and
                # this first tail connection.
                ["ssh", host, f"tail -n +{offset + 1} -F {run_dir}/output.log"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=_NO_WINDOW)
            self._remote_proc = proc
            try:
                for raw in proc.stdout:
                    offset += 1
                    yield raw
            except GeneratorExit:
                proc.terminate()
                raise
            finally:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except Exception:
                    pass

            if self.stop_event.is_set():
                return
            attempt += 1
            if attempt > max_attempts:
                self.log_message.emit(
                    f"[remote] Log stream: giving up after {attempt} reconnect attempts.")
                return
            time.sleep(min(5 * attempt, 30))
            alive = subprocess.run(
                ["ssh", host,
                 f"pgrep -f 'remote_train.py --config {run_dir}/config.json' "
                 f">/dev/null && echo yes || echo no"],
                capture_output=True, text=True, creationflags=_NO_WINDOW)
            if alive.stdout.strip() != "yes":
                self.log_message.emit(
                    "[remote] Remote training process is no longer running and "
                    "never reported completion — treating this run as failed "
                    "rather than reconnecting forever.")
                return
            self.log_message.emit(
                f"[remote] Log stream dropped; training is still alive on the "
                f"server — reconnecting (attempt {attempt})...")

    def remote_train_loop(self):
        """Counterpart of train_loop that drives remote_train.py over SSH.

        Streams JSON-line events from the remote process into the same Qt
        signals the local trainer uses, so plots/log/cluster panels behave
        identically. Checkpoints are pulled after every epoch."""
        _train_start = time.perf_counter()
        params = dict(self.training_params)
        host = params.get("remote_host") or "nvidia5090"
        code_dir = (params.get("remote_code_dir") or "~/Thinghiem/uav_code").rstrip("/")
        ds_type = params.get("dataset_type", "")
        root_key = self._REMOTE_ROOT_KEYS.get(ds_type)
        final_status = "failed"
        try:
            if root_key is None:
                raise ValueError(
                    f"Remote training supports dataset types "
                    f"{sorted(self._REMOTE_ROOT_KEYS)}, not {ds_type!r}.")
            remote_root = params.get("remote_data_root", "").strip()
            if remote_root:
                params[root_key] = remote_root
            params["device"] = "cuda"

            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = f"{code_dir}/runs/{run_id}"
            self._remote_host = host
            self._remote_run_dir = run_dir
            # Survives past this run's own `finally` clearing _remote_run_dir —
            # lets "Reattach" find this job again even after the GUI's own
            # monitoring connection dies/hangs and the run is reported failed.
            self._last_remote_host = host
            self._last_remote_run_dir = run_dir
            self._save_last_remote_run(host, run_dir)

            # Pre-flight: fail fast (before any sync/upload) if the remote
            # dataset root doesn't exist — a stale root from a previously
            # selected dataset type otherwise burns a full launch cycle just
            # to die in build_dataset on the server (runs #224/#225).
            if remote_root:
                # Existence of the root alone isn't enough — a stale root from
                # another dataset type EXISTS but lacks this type's expected
                # drone subdirectory (mirrors dataset.py's own checks).
                _drone_subdir = {"sues200": "drone_view_512",
                                 "denseuav": "drone",
                                 "university1652": "drone",
                                 "game4loc": "drone"}[ds_type]
                _check_dir = f"{remote_root.rstrip('/')}/{_drone_subdir}"
                # _ssh_run itself now retries transport-level failures (exit
                # 255) on a fresh connection, so a returncode surviving that
                # is trustworthy: either the shell's own test -d result (0/1)
                # or a repeated transport failure worth surfacing as-is.
                chk = self._ssh_run(
                    ["ssh", host, f"test -d {_check_dir}"], check=False)
                if chk.returncode != 0:
                    raise FileNotFoundError(
                        f"Remote dataset root is missing {ds_type}'s expected "
                        f"'{_drone_subdir}/' subdirectory on {host}: {_check_dir}\n"
                        f"The 'Remote dataset root' field likely still points at "
                        f"a different dataset type's path.")
                self.log_message.emit(
                    f"[remote] Dataset root verified: {host}:{_check_dir}")

            here = Path(__file__).parent
            cfg_path = here / "remote_train_config.json"
            cfg_path.write_text(json.dumps(params, indent=2, default=str),
                                encoding="utf-8")
            self.log_message.emit(f"[remote] Syncing code to {host}:{code_dir} ...")
            self._ssh_run(["ssh", host, f"mkdir -p {code_dir} {run_dir}"])
            files = [str(here / f) for f in self.REMOTE_CODE_FILES
                     if (here / f).exists()]
            self._ssh_run(["scp", "-O", "-q", *files, f"{host}:{code_dir}/"],
                          timeout=120)
            self._ssh_run(["scp", "-O", "-q", str(cfg_path),
                           f"{host}:{run_dir}/config.json"], timeout=60)

            # Upload the local checkpoint so the remote can actually resume /
            # fine-tune from it: remote_train.py's maybe_resume only looks at
            # <run_dir>/checkpoint.pt, and the run_dir is fresh per run — without
            # this upload, "Resume checkpoint" silently trains from ImageNet
            # weights instead. check=True on purpose: a failed upload must abort
            # the run rather than silently fine-tune from scratch.
            # scp -C: measured ~8% transfer saving on a real 1GB checkpoint
            # (fp32 weights are nearly incompressible; zlib-1 actually INFLATES
            # to 105%, zlib-6 gives 91.9% at 51MB/s — far above the link speed,
            # so inline compression is free but explicit gzip staging is not
            # worth the extra steps).
            if params.get("resume_checkpoint"):
                local_ckpt = self.checkpoint_path_for()
                if local_ckpt.exists():
                    # Content-addressed remote cache: run dirs are fresh per
                    # launch, so uploading straight into <run_dir> re-sends the
                    # same ~1GB on every retry (e.g. re-running right after a
                    # config fix). Instead upload once to ckpt_cache/<md5>.pt
                    # and hardlink into each run dir (instant, same filesystem).
                    import hashlib
                    h = hashlib.md5()
                    with open(local_ckpt, "rb") as f:
                        for chunk in iter(lambda: f.read(1 << 22), b""):
                            h.update(chunk)
                    digest = h.hexdigest()
                    cache_dir = f"{code_dir}/ckpt_cache"
                    cache_path = f"{cache_dir}/{digest}.pt"
                    have = self._ssh_run(
                        ["ssh", host, f"test -f {cache_path} && echo yes || echo no"],
                        check=False, text=True)
                    if have.stdout.strip() == "yes":
                        self.log_message.emit(
                            f"[remote] Checkpoint already cached on remote "
                            f"({digest[:12]}) — skipping upload.")
                    else:
                        sz_mb = local_ckpt.stat().st_size / 1e6
                        self.log_message.emit(
                            f"[remote] Uploading checkpoint for resume "
                            f"({local_ckpt.name}, {sz_mb:.0f} MB — takes a few "
                            f"minutes on a slow link)...")
                        t0 = time.perf_counter()
                        # Upload to .part then mv: a half-transferred file must
                        # never be mistaken for a valid cache entry.
                        self._ssh_run(["ssh", host, f"mkdir -p {cache_dir}"])
                        # 1GB over a slow link legitimately takes minutes —
                        # generous timeout, no retry (a re-send is another
                        # multi-minute upload; let the user re-launch instead).
                        self._ssh_run(
                            ["scp", "-O", "-q", "-C", str(local_ckpt),
                             f"{host}:{cache_path}.part"],
                            timeout=3600, retries=0)
                        # test -f guard makes the mv idempotent so the retry
                        # can't fail on "mv already happened but ssh wedged".
                        self._ssh_run(
                            ["ssh", host,
                             f"(test -f {cache_path} || "
                             f"mv {cache_path}.part {cache_path}) && "
                             # keep the 3 newest cache entries, drop the rest
                             f"ls -t {cache_dir}/*.pt 2>/dev/null | tail -n +4 "
                             f"| xargs -r rm --"],
                            timeout=60)
                        dt = time.perf_counter() - t0
                        self.log_message.emit(
                            f"[remote] Checkpoint uploaded in {dt:.0f}s "
                            f"({sz_mb / max(dt, 1e-9):.1f} MB/s effective).")
                    self._ssh_run(
                        ["ssh", host,
                         f"test -f {run_dir}/checkpoint.pt || "
                         f"ln {cache_path} {run_dir}/checkpoint.pt "
                         f"|| cp {cache_path} {run_dir}/checkpoint.pt"],
                        timeout=120)
                else:
                    self.log_message.emit(
                        f"[remote] Resume requested but no local checkpoint at "
                        f"{local_ckpt} — remote will start from initialized weights.")

            self.log_message.emit(f"[remote] Starting training on {host} (run {run_id}).")
            # Launch DETACHED (setsid + nohup, output redirected to a file on
            # the remote): the training process must NOT be a child of this
            # SSH session. Previously it was `ssh host "python3 remote_train.py"`
            # run in the foreground — a dropped network connection (WiFi blip,
            # router hiccup) tears down the SSH session, which sends SIGHUP to
            # the remote shell's child, killing training instantly with no
            # traceback (confirmed: server GPU/RAM/disk all healthy, process
            # simply gone, no error ever emitted). Detaching means only the
            # log STREAM below needs to reconnect after a drop; the actual
            # training is unaffected. `< /dev/null` + trailing `&` backgrounds
            # the job so this ssh call returns immediately instead of blocking
            # for the whole run.
            # (cmd & disown); exit 0 — explicit disown + exit measured as the
            # fastest, most reliable detach idiom of several tried (0s launch
            # return vs 1-3s for setsid-only / ssh -f variants; a bare
            # "setsid nohup ... &" was ALSO seen to block for a job's full
            # duration once, of uncertain cause — this form is the belt-and-
            # suspenders choice).
            # timeout=30: three separate incidents (2026-07-17 x2, 2026-07-18)
            # of this ssh.exe hanging client-side for 15+ minutes AFTER the
            # remote job had already detached and started (once even after it
            # had finished) — blocking this thread here means the log-tail
            # stage never starts and the GUI looks frozen right after
            # "Starting training". The remote detach is instant, so timeout
            # expiry means launch SUCCEEDED but the ssh connection wedged:
            # kill it and proceed to the stream, which independently verifies
            # the run via output.log + pgrep anyway.
            try:
                subprocess.run(
                    ["ssh", host,
                     f"cd {code_dir} && "
                     f"(setsid nohup python3 -u remote_train.py "
                     f"--config {run_dir}/config.json --run-dir {run_dir} "
                     f"> {run_dir}/output.log 2>&1 < /dev/null & disown); exit 0"],
                    check=True, capture_output=True, creationflags=_NO_WINDOW,
                    stdin=subprocess.DEVNULL, timeout=30)
            except subprocess.TimeoutExpired:
                self.log_message.emit(
                    "[remote] Launch ssh didn't return within 30s (known "
                    "client-side hang) — assuming the job detached and "
                    "verifying via the log stream...")

            final_status = self._consume_remote_events(host, run_dir)
            if final_status == "failed" and self.stop_event.is_set():
                final_status = "stopped"
            # Make sure the very last checkpoint is on disk locally before we
            # report the run as done (pull thread may still be busy).
            self._wait_for_checkpoint_pulls(timeout=600)
        except Exception as exc:
            self.log_message.emit(f"[remote] Training error: {exc}")
        finally:
            self._remote_run_dir = None
            self._remote_proc = None
        self._save_train_history(final_status, _train_start)
        self.training_finished.emit(final_status)

    def _consume_remote_events(self, host, run_dir) -> str:
        """Stream + dispatch JSON-line events from a remote run_dir's
        output.log until a 'finished' event arrives. Shared by a fresh launch
        (remote_train_loop) and reattach_remote_run (which skips sync/upload/
        launch entirely and just watches an already-running detached job) —
        factored out specifically so reattach doesn't duplicate this dispatch
        logic. Returns the final status string ('finished'/'stopped'/'failed',
        'failed' being the default if the stream ends without ever seeing a
        'finished' event)."""
        final_status = "failed"
        log_gen = self._stream_remote_output(host, run_dir)
        try:
            for raw in log_gen:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("{"):
                    self.log_message.emit(f"[remote] {line}")
                    continue
                try:
                    evt = json.loads(line)
                except ValueError:
                    self.log_message.emit(f"[remote] {line}")
                    continue
                kind = evt.get("event")
                if kind == "status":
                    self.log_message.emit(f"[remote] {evt.get('message', '')}")
                elif kind == "loss":
                    self.loss_recorded.emit(
                        evt.get("metrics", {}), evt.get("step", 0),
                        evt.get("epoch", 0), evt.get("batch_step", 0))
                elif kind == "cluster":
                    stats = evt.get("stats") or {}
                    if "target_clusters" in stats:
                        stats.setdefault("dead_sample_indices", [])
                        self.clusters_updated.emit(stats, evt.get("epoch", 0))
                elif kind == "model_created":
                    self.log_message.emit(
                        f"[remote] Model created ({evt.get('model_size', '?')} params).")
                elif kind == "checkpoint":
                    self._pull_remote_checkpoint_async(
                        host, evt.get("path"), evt.get("epoch"))
                elif kind == "cluster_sampling":
                    self._pull_remote_cluster_sampling_async(host, evt.get("path"))
                elif kind == "finished":
                    final_status = evt.get("status", "failed")
                    if evt.get("error"):
                        self.log_message.emit(
                            f"[remote] ERROR: {evt['error']}\n"
                            f"{evt.get('traceback', '')}")
                    break   # stop consuming; log_gen.close() below tidies up
        finally:
            log_gen.close()
        return final_status

    def reattach_remote_run(self, host, run_dir):
        """Resume GUI visibility into an ALREADY-RUNNING detached remote job
        without touching it — no code sync, no checkpoint upload, no launch.
        For exactly the situation this session hit: the local launch/monitor
        connection can itself hang or die (confirmed: a real remote_train.py
        invocation left its launch ssh.exe stuck for 15+ minutes on Windows,
        even though detach+reconnect for the log stream was separately
        verified working) while the remote training keeps running fine,
        unmonitored. Killing the stuck local process to recover makes the GUI
        misreport 'failed' even though nothing remote actually failed — this
        lets you get real visibility (and checkpoint pulls) back safely
        instead. Call with the host/run_dir of the run you want to reattach
        to (defaults to the last one this session launched). Runs on a
        background thread (spawned by start_reattach_remote, which — like
        start_training — owns all Qt widget-state changes on the main thread;
        this method itself only ever touches widgets via the log_message
        signal)."""
        _train_start = time.perf_counter()
        self._remote_host = host
        self._remote_run_dir = run_dir
        self.log_message.emit(f"[remote] Reattaching to existing run at {host}:{run_dir} "
                              f"(job keeps running regardless of this GUI's state)...")
        final_status = "failed"
        try:
            final_status = self._consume_remote_events(host, run_dir)
            if final_status == "failed" and self.stop_event.is_set():
                final_status = "stopped"
            self._wait_for_checkpoint_pulls(timeout=600)
        except Exception as exc:
            self.log_message.emit(f"[remote] Reattach error: {exc}")
        finally:
            self._remote_run_dir = None
            self._remote_proc = None
        self._save_train_history(final_status, _train_start)
        self.training_finished.emit(final_status)

    def _pull_remote_checkpoint_async(self, host, remote_path, epoch=None):
        """Download the remote checkpoint to the local checkpoint path.

        Runs in a worker thread so the event stream keeps flowing; if a pull
        is already in flight, remember the request and re-pull when it ends
        (only the newest checkpoint matters)."""
        if not remote_path:
            return
        if self.pause_event.is_set():
            # User paused: don't burn the link on a ~1GB pull. Only the newest
            # checkpoint matters, so just remember the latest request; it's
            # flushed by toggle_pause on resume.
            self._paused_pull_pending = (host, remote_path, epoch)
            self.log_message.emit(
                f"[remote] Paused — checkpoint download deferred"
                f"{f' (epoch {epoch})' if epoch is not None else ''}.")
            return
        if not hasattr(self, "_ckpt_pull_queue"):
            import queue
            self._ckpt_pull_queue = queue.Queue()
            self._ckpt_pull_busy = False
            threading.Thread(target=self._ckpt_pull_worker, daemon=True).start()
        self._ckpt_pull_queue.put((host, remote_path, epoch))

    def _ckpt_pull_worker(self):
        q = self._ckpt_pull_queue
        while True:
            item = q.get()
            self._ckpt_pull_busy = True
            # Only the newest checkpoint matters — drain any backlog.
            while not q.empty():
                try:
                    item = q.get_nowait()
                except Exception:
                    break
            p_host, p_remote, p_epoch = item
            try:
                local = self.checkpoint_path_for()
                local.parent.mkdir(parents=True, exist_ok=True)
                tmp = local.with_suffix(".remote_tmp")
                # -C: ~8% transfer saving measured on real fp32 checkpoints,
                # free at this link speed (see upload note in remote_train_loop).
                subprocess.run(["scp", "-O", "-q", "-C", f"{p_host}:{p_remote}", str(tmp)],
                               check=True, capture_output=True,
                               creationflags=_NO_WINDOW)
                if local.exists():
                    backup_dir = local.parent / "olds"
                    backup_dir.mkdir(exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    shutil.copy2(local, backup_dir / f"{local.stem}_{ts}{local.suffix}")
                    self._prune_old_backups(backup_dir, local.stem, local.suffix)
                tmp.replace(local)
                label = f" (epoch {p_epoch})" if p_epoch is not None else ""
                self.log_message.emit(f"[remote] Checkpoint pulled to {local}{label}.")
            except Exception as exc:
                self.log_message.emit(f"[remote] Checkpoint pull failed: {exc}")
            finally:
                self._ckpt_pull_busy = False

    def _wait_for_checkpoint_pulls(self, timeout=600):
        q = getattr(self, "_ckpt_pull_queue", None)
        if q is None:
            return
        if not q.empty() or getattr(self, "_ckpt_pull_busy", False):
            self.log_message.emit("[remote] Waiting for final checkpoint download...")
        deadline = time.monotonic() + timeout
        while (not q.empty() or getattr(self, "_ckpt_pull_busy", False)):
            if time.monotonic() > deadline:
                self.log_message.emit("[remote] Checkpoint download still running in background.")
                return
            time.sleep(0.5)

    def _pull_remote_cluster_sampling_async(self, host, remote_path):
        """Download + load the full remote cluster data (see
        remote_train.py's on_cluster_sampling) so the hard-negative /
        distinctiveness preview works for remote sessions automatically,
        the same way it already does for local training — instead of
        requiring a manual 'Build Cluster Data' click that recomputes
        everything from scratch locally. Same drain-to-newest queue pattern
        as checkpoint pulls (only the latest epoch's data matters)."""
        if not remote_path:
            return
        if not hasattr(self, "_cs_pull_queue"):
            import queue
            self._cs_pull_queue = queue.Queue()
            threading.Thread(target=self._cs_pull_worker, daemon=True).start()
        self._cs_pull_queue.put((host, remote_path))

    def _cs_pull_worker(self):
        q = self._cs_pull_queue
        while True:
            item = q.get()
            while not q.empty():
                try:
                    item = q.get_nowait()
                except Exception:
                    break
            host, remote_path = item
            try:
                import tempfile
                tmp = Path(tempfile.gettempdir()) / "uav_remote_cluster_sampling.pt"
                subprocess.run(
                    ["scp", "-O", "-q", "-C", f"{host}:{remote_path}", str(tmp)],
                    check=True, capture_output=True, creationflags=_NO_WINDOW)
                cs = torch.load(tmp, map_location="cpu", weights_only=False)
                self.cluster_sampling_ready.emit(cs)
                self.log_message.emit(
                    f"[remote] Cluster data synced from server "
                    f"({len(cs.get('cluster_members', {}))} clusters, "
                    f"{len(cs.get('all_indices', []))} samples).")
            except Exception as exc:
                self.log_message.emit(f"[remote] Cluster data sync failed: {exc}")

    def log_append(self, message):
        self.log.append(message)

    def add_loss(self, metrics, step, epoch, batch_step):
        loss = metrics["total_loss"]
        record = {
            "step": step,
            "epoch": epoch,
            "batch": batch_step,
            **metrics,
        }
        self.losses.append(loss)
        self.loss_records.append(record)
        self.log.append(f"Epoch {epoch} batch {batch_step} loss: {loss:.6f}")
        self.log.append(
            "  "
            f"pos_sim={metrics['positive_similarity']:.4f}, "
            f"neg_sim={metrics['negative_similarity']:.4f}, "
            f"pos_prob={metrics['positive_probability']:.4f}, "
            f"pos_score={metrics['positive_score']:.4f}, "
            f"neg_score={metrics['negative_score']:.4f}"
        )
        self.update_metrics_table(metrics)
        self.update_plot()

    def _on_ds_remote_root_swap(self):
        """Swap the remote-data-root field to the newly-selected dataset type's
        remembered path, stashing the outgoing type's current text first.
        Dataset types without remote support (crop_pairs, game4loc) leave the
        field untouched."""
        ds = self.dataset_type_input.currentData()
        prev = self._rr_prev_ds
        if prev is not None and prev != ds and prev in self._remote_roots_by_ds:
            txt = self.remote_data_root_input.text().strip()
            if txt:
                self._remote_roots_by_ds[prev] = txt
        if ds in self._remote_roots_by_ds:
            self.remote_data_root_input.setText(self._remote_roots_by_ds[ds])
        self._rr_prev_ds = ds

    def _save_ui_settings(self, path_override: Path = None):
        settings = {
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_size_input.value(),
            "microbatch_size": self.microbatch_size_input.value(),
            "learning_rate": self.lr_input.value(),
            "weight_decay": self.weight_decay_input.value(),
            "num_workers": self.num_workers_input.value(),
            "use_amp": self.amp_input.isChecked(),
            "backbone": self.backbone_input.currentData(),
            "pretrained": self.pretrained_input.isChecked(),
            "unfreeze_backbone_layers": self.unfreeze_backbone_layers_input.value(),
            "resume_checkpoint": self.resume_checkpoint_input.isChecked(),
            "checkpoint_path_override": self.checkpoint_path_input.text(),
            "shuffle": self.shuffle_input.isChecked(),
            "augment": self.augment_input.isChecked(),
            "hard_mining": self.hard_mining_input.isChecked(),
            "cluster_model_ckpt": self.cluster_model_input.text().strip(),
            "cluster_count": self.cluster_count_input.value(),
            "auto_cluster_k": self.auto_cluster_k_input.isChecked(),
            "cluster_every": self.cluster_every_input.value(),
            "enable_clustering": self.enable_clustering_input.isChecked(),
            "cluster_consistency_weight": self.cluster_consistency_weight_input.value(),
            "negative_weight": self.negative_weight_input.value(),
            "label_smoothing": self.label_smoothing_input.value(),
            "group_size": self.group_size_input.value(),
            "dataset_root": self.dataset_root_input.text(),
            "dataset_type": self.dataset_type_input.currentData(),
            "gta_data_root": self.gta_data_root_input.text(),
            "gta_json": self.gta_json_input.currentText(),
            "gta_mode": self.gta_mode_input.currentData(),
            "gta_augment_pos": self.gta_augment_pos_input.isChecked(),
            "filter_positives": self.filter_pos_input.isChecked(),
            "filter_distinctive_min": self.filter_pos_thresh_input.value(),
            "filter_distinctive_max": self.filter_pos_max_input.value(),
            "gta_test_json": self.gta_test_json_input.currentText().strip(),
            "u1652_train_root": self.u1652_root_input.text(),
            "denseuav_train_root": self.denseuav_root_input.text(),
            "sues200_train_root": self.sues200_root_input.text(),
            "denseuav_cross_altitude": self.denseuav_cross_alt_input.isChecked(),
            "denseuav_altitude_weight_tau": self.denseuav_alt_tau_input.value(),
            "denseuav_altitude_full_strength": self.denseuav_alt_full_strength_input.isChecked(),
            "training_label": self.training_label_input.text().strip(),
            "remote_train": self.remote_train_input.isChecked(),
            "remote_host": self.remote_host_input.text().strip(),
            "remote_code_dir": self.remote_code_dir_input.text().strip(),
            "remote_data_root": self.remote_data_root_input.text().strip(),
            "remote_data_roots": {
                **self._remote_roots_by_ds,
                **({self.dataset_type_input.currentData():
                    self.remote_data_root_input.text().strip()}
                   if self.dataset_type_input.currentData() in self._remote_roots_by_ds
                   and self.remote_data_root_input.text().strip() else {}),
            },
            "auto_eval": self.auto_eval_input.isChecked(),
            "epoch_eval": self.epoch_eval_input.isChecked(),
            "quick_eval_denseuav": self.quick_eval_denseuav_input.isChecked(),
            "quick_eval_every_n": self.quick_eval_every_n_input.value(),
            "auto_train_scale_head": self.auto_sh_input.isChecked(),
            # Scale head tab
            "sh_checkpoint": self.sh_checkpoint_input.text(),
            "sh_data_root":      self.sh_data_root_input.text(),
            "sh_dataset_type":   self.sh_dataset_type_input.currentData(),
            "sh_epochs":     self.sh_epochs_input.value(),
            "sh_batch_size": self.sh_batch_size_input.value(),
            "sh_lr":         self.sh_lr_input.value(),
            "sh_weight_decay": self.sh_wd_input.value(),
            "sh_amp":        self.sh_amp_input.isChecked(),
            "sh_autosave":   self.sh_autosave_input.isChecked(),
        }
        try:
            path = path_override or self._current_settings_path()
            path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _current_settings_path(self) -> Path:
        return UI_SETTINGS_PATH_REMOTE if self.remote_train_input.isChecked() else UI_SETTINGS_PATH

    def _on_remote_toggle(self, checked):
        """Auto-switch config profiles when "Train on remote (SSH)" flips.
        `checked` already reflects the NEW state (Qt fires toggled after the
        internal state changes), so _current_settings_path() now resolves to
        the mode we're ENTERING — meaning whatever's on screen must be saved
        to the OTHER file (the mode we're LEAVING) explicitly, before loading
        the newly-selected mode's own saved profile."""
        leaving_path = UI_SETTINGS_PATH if checked else UI_SETTINGS_PATH_REMOTE
        self._save_ui_settings(path_override=leaving_path)
        # Defensive: _load_ui_settings() below will itself call setChecked()
        # with whatever "remote_train" the target file has saved — which
        # should already match `checked` by construction (each file is only
        # ever saved while in its own mode), but block signals anyway so a
        # mismatched/hand-edited file can't cause a reentrant toggle loop.
        self.remote_train_input.blockSignals(True)
        self._load_ui_settings()
        self.remote_train_input.blockSignals(False)
        self.log.append(
            f"Switched to {'remote' if checked else 'local'} training profile "
            f"({self._current_settings_path().name}).")

    def _load_ui_settings(self):
        settings_path = self._current_settings_path()
        if not settings_path.exists():
            return
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            return

        if "epochs" in settings:
            self.epochs_input.setValue(settings["epochs"])
        if "batch_size" in settings:
            self.batch_size_input.setValue(settings["batch_size"])
        if "microbatch_size" in settings:
            self.microbatch_size_input.setValue(settings["microbatch_size"])
        if "learning_rate" in settings:
            self.lr_input.setValue(settings["learning_rate"])
        if "weight_decay" in settings:
            self.weight_decay_input.setValue(settings["weight_decay"])
        if "num_workers" in settings:
            self.num_workers_input.setValue(settings["num_workers"])
        if "use_amp" in settings:
            self.amp_input.setChecked(settings["use_amp"])
        if "backbone" in settings:
            idx = self.backbone_input.findData(settings["backbone"])
            if idx >= 0:
                self.backbone_input.setCurrentIndex(idx)
        if "pretrained" in settings:
            self.pretrained_input.setChecked(settings["pretrained"])
        if "unfreeze_backbone_layers" in settings:
            self.unfreeze_backbone_layers_input.setValue(settings["unfreeze_backbone_layers"])
        if "resume_checkpoint" in settings:
            self.resume_checkpoint_input.setChecked(settings["resume_checkpoint"])
        if "checkpoint_path_override" in settings:
            self.checkpoint_path_input.setText(settings["checkpoint_path_override"])
        if "shuffle" in settings:
            self.shuffle_input.setChecked(settings["shuffle"])
        if "augment" in settings:
            self.augment_input.setChecked(settings["augment"])
        if "hard_mining" in settings:
            self.hard_mining_input.setChecked(settings["hard_mining"])
        if "cluster_model_ckpt" in settings:
            self.cluster_model_input.setText(settings["cluster_model_ckpt"])
        if "cluster_count" in settings:
            self.cluster_count_input.setValue(settings["cluster_count"])
        if "auto_cluster_k" in settings:
            self.auto_cluster_k_input.setChecked(settings["auto_cluster_k"])
        if "cluster_every" in settings:
            self.cluster_every_input.setValue(settings["cluster_every"])
        if "enable_clustering" in settings:
            self.enable_clustering_input.setChecked(settings["enable_clustering"])
        if "cluster_consistency_weight" in settings:
            self.cluster_consistency_weight_input.setValue(settings["cluster_consistency_weight"])
        if "negative_weight" in settings:
            self.negative_weight_input.setValue(settings["negative_weight"])
        if "label_smoothing" in settings:
            self.label_smoothing_input.setValue(settings["label_smoothing"])

        if "group_size" in settings:
            self.group_size_input.setValue(settings["group_size"])
        if "dataset_root" in settings:
            self.dataset_root_input.setText(settings["dataset_root"])
        if "dataset_type" in settings:
            idx = self.dataset_type_input.findData(settings["dataset_type"])
            if idx >= 0:
                self.dataset_type_input.setCurrentIndex(idx)
        if "gta_data_root" in settings:
            self.gta_data_root_input.setText(settings["gta_data_root"])
        if "gta_json" in settings:
            self.gta_json_input.setCurrentText(settings["gta_json"])
        if "gta_mode" in settings:
            idx = self.gta_mode_input.findData(settings["gta_mode"])
            if idx >= 0:
                self.gta_mode_input.setCurrentIndex(idx)
        if "gta_augment_pos" in settings:
            self.gta_augment_pos_input.setChecked(settings["gta_augment_pos"])
        if "u1652_train_root" in settings:
            self.u1652_root_input.setText(settings["u1652_train_root"])
        if "denseuav_train_root" in settings:
            self.denseuav_root_input.setText(settings["denseuav_train_root"])
        if "sues200_train_root" in settings:
            self.sues200_root_input.setText(settings["sues200_train_root"])
        if "denseuav_cross_altitude" in settings:
            self.denseuav_cross_alt_input.setChecked(settings["denseuav_cross_altitude"])
        if "denseuav_altitude_weight_tau" in settings:
            self.denseuav_alt_tau_input.setValue(settings["denseuav_altitude_weight_tau"])
        if "denseuav_altitude_full_strength" in settings:
            self.denseuav_alt_full_strength_input.setChecked(settings["denseuav_altitude_full_strength"])
        if "filter_positives" in settings:
            self.filter_pos_input.setChecked(settings["filter_positives"])
        if "filter_distinctive_min" in settings:
            self.filter_pos_thresh_input.setValue(settings["filter_distinctive_min"])
        if "filter_distinctive_max" in settings:
            self.filter_pos_max_input.setValue(settings["filter_distinctive_max"])
        if "gta_test_json" in settings:
            self.gta_test_json_input.setCurrentText(settings["gta_test_json"])
        if "training_label" in settings:
            self.training_label_input.setText(settings["training_label"])
        if "remote_host" in settings and settings["remote_host"]:
            self.remote_host_input.setText(settings["remote_host"])
        if "remote_code_dir" in settings and settings["remote_code_dir"]:
            self.remote_code_dir_input.setText(settings["remote_code_dir"])
        if "remote_data_root" in settings and settings["remote_data_root"]:
            self.remote_data_root_input.setText(settings["remote_data_root"])
        if isinstance(settings.get("remote_data_roots"), dict):
            self._remote_roots_by_ds.update(
                {k: v for k, v in settings["remote_data_roots"].items()
                 if isinstance(v, str) and v.strip()})
        # NOTE: no migration from the legacy single "remote_data_root" value
        # into the per-ds dict — the old single-field design kept whatever was
        # last typed across dataset-type switches, so the saved (dataset_type,
        # remote_data_root) pairing is NOT trustworthy. Migrating it poisoned
        # the dict with a stale cross-dataset path (runs #224-#226). Known
        # dataset types start from the defaults instead, which point at the
        # actual upload locations on the server.
        # The per-ds dict wins over the legacy single value for known types —
        # the field must match the CURRENT dataset type, not whichever type
        # happened to be selected when the file was last saved.
        _ds_now = self.dataset_type_input.currentData()
        if _ds_now in self._remote_roots_by_ds:
            self.remote_data_root_input.setText(self._remote_roots_by_ds[_ds_now])
        self._rr_prev_ds = _ds_now
        if "auto_eval" in settings:
            self.auto_eval_input.setChecked(settings["auto_eval"])
        if "epoch_eval" in settings:
            self.epoch_eval_input.setChecked(settings["epoch_eval"])
        if "quick_eval_denseuav" in settings:
            self.quick_eval_denseuav_input.setChecked(settings["quick_eval_denseuav"])
        if "quick_eval_every_n" in settings:
            self.quick_eval_every_n_input.setValue(settings["quick_eval_every_n"])
        if "auto_train_scale_head" in settings:
            self.auto_sh_input.setChecked(settings["auto_train_scale_head"])
        # Scale head tab
        if "sh_checkpoint" in settings:
            self.sh_checkpoint_input.setText(settings["sh_checkpoint"])
        if "sh_data_root" in settings:
            self.sh_data_root_input.setText(settings["sh_data_root"])
        if "sh_dataset_type" in settings:
            for i in range(self.sh_dataset_type_input.count()):
                if self.sh_dataset_type_input.itemData(i) == settings["sh_dataset_type"]:
                    self.sh_dataset_type_input.setCurrentIndex(i)
                    break
        if "sh_epochs" in settings:
            self.sh_epochs_input.setValue(settings["sh_epochs"])
        if "sh_batch_size" in settings:
            self.sh_batch_size_input.setValue(settings["sh_batch_size"])
        if "sh_lr" in settings:
            self.sh_lr_input.setValue(settings["sh_lr"])
        if "sh_weight_decay" in settings:
            self.sh_wd_input.setValue(settings["sh_weight_decay"])
        if "sh_amp" in settings:
            self.sh_amp_input.setChecked(settings["sh_amp"])
        if "sh_autosave" in settings:
            self.sh_autosave_input.setChecked(settings["sh_autosave"])

    def read_training_params(self):
        return {
            "epochs": self.epochs_input.value(),
            "batch_size": self.batch_size_input.value(),
            "microbatch_size": self.microbatch_size_input.value(),
            "learning_rate": self.lr_input.value(),
            "weight_decay": self.weight_decay_input.value(),
            # Which descriptor pathway is optimized. Derived from the backbone
            # CONFIG's no_head flag, not the key name — "convnext_b"/"convnext_t"
            # are no-head backbones without "nohead" in their key, and the old
            # name-based check mislabeled them cluster_head_512 (a head that
            # doesn't exist on these models).
            "training_mode": ("backbone_raw_nohead"
                              if BACKBONE_CONFIGS.get(
                                  self.backbone_input.currentData() or "", {}
                              ).get("no_head")
                              else "cluster_head_512"),
            "embed_dim": CLUSTER_DESCRIPTOR_DIM,
            "global_dim": GLOBAL_DESCRIPTOR_DIM,
            "device": self.device_input.currentData(),
            "num_workers": self.num_workers_input.value(),
            "use_amp": self.amp_input.isChecked(),
            "backbone": self.backbone_input.currentData(),
            "pretrained": self.pretrained_input.isChecked(),
            "freeze_backbone": True,
            "unfreeze_backbone_layers": self.unfreeze_backbone_layers_input.value(),
            "resume_checkpoint": self.resume_checkpoint_input.isChecked(),
            "checkpoint_path_override": self.checkpoint_path_input.text(),
            "shuffle": self.shuffle_input.isChecked(),
            "augment": self.augment_input.isChecked(),
            "hard_mining": self.hard_mining_input.isChecked(),
            "cluster_model_ckpt": self.cluster_model_input.text().strip(),
            "cluster_count": self.cluster_count_input.value(),
            "auto_cluster_k": self.auto_cluster_k_input.isChecked(),
            "cluster_every": self.cluster_every_input.value(),
            "enable_clustering": self.enable_clustering_input.isChecked(),
            "cluster_consistency_weight": self.cluster_consistency_weight_input.value(),
            "negative_weight": self.negative_weight_input.value(),
            "label_smoothing": self.label_smoothing_input.value(),
            "group_size": self.group_size_input.value(),
            "dataset_type": self.dataset_type_input.currentData(),
            "dataset_root": self.dataset_root_input.text() or (DATASET_ROOT if self.dataset_type_input.currentData() == "crop_pairs" else ""),
            "gta_data_root": self.gta_data_root_input.text(),
            "gta_json": self.gta_json_input.currentText(),
            "gta_mode": self.gta_mode_input.currentData() or "pos_semipos",
            "gta_augment_pos": self.gta_augment_pos_input.isChecked(),
            "filter_positives": self.filter_pos_input.isChecked(),
            "filter_distinctive_min": self.filter_pos_thresh_input.value(),
            "filter_distinctive_max": self.filter_pos_max_input.value(),
            "gta_test_json": self.gta_test_json_input.currentText().strip(),
            "u1652_train_root": self.u1652_root_input.text(),
            "denseuav_train_root": self.denseuav_root_input.text(),
            "sues200_train_root": self.sues200_root_input.text(),
            "denseuav_cross_altitude": self.denseuav_cross_alt_input.isChecked(),
            "denseuav_altitude_weight_tau": self.denseuav_alt_tau_input.value(),
            "denseuav_altitude_full_strength": self.denseuav_alt_full_strength_input.isChecked(),
            "epoch_eval": self.epoch_eval_input.isChecked(),
            "quick_eval_denseuav": self.quick_eval_denseuav_input.isChecked(),
            "quick_eval_every_n": self.quick_eval_every_n_input.value(),
            "training_label": self.training_label_input.text().strip(),
            "remote_train": self.remote_train_input.isChecked(),
            "remote_host": self.remote_host_input.text().strip(),
            "remote_code_dir": self.remote_code_dir_input.text().strip(),
            "remote_data_root": self.remote_data_root_input.text().strip(),
            "auto_train_scale_head": self.auto_sh_input.isChecked(),
            "sh_data_root":    self.sh_data_root_input.text().strip(),
            "sh_dataset_type": self.sh_dataset_type_input.currentData(),
            "sh_epochs":       self.sh_epochs_input.value(),
            "sh_batch_size":   self.sh_batch_size_input.value(),
            "sh_lr":           self.sh_lr_input.value(),
            "sh_weight_decay": self.sh_wd_input.value(),
            "sh_amp":          self.sh_amp_input.isChecked(),
        }

    def set_parameter_inputs_enabled(self, enabled):
        for widget in (
            self.epochs_input,
            self.batch_size_input,
            self.microbatch_size_input,
            self.lr_input,
            self.backbone_input,
            self.pretrained_input,
            self.resume_checkpoint_input,
            self.augment_input,
            self.cluster_count_input,
            self.auto_cluster_k_input,
            self.cluster_every_input,
            self.negative_weight_input,
            self.label_smoothing_input,
            self.group_size_input,
            self.dataset_root_input,
            self.dataset_type_input,
            self.gta_data_root_input,
            self.gta_json_input,
            self.gta_mode_input,
            self.gta_augment_pos_input,
            self.gta_test_json_input,
            self.training_label_input,
        ):
            widget.setEnabled(enabled)

    def update_cluster_status(self, stats, epoch):
        self.cluster_epoch_label.setText(f"Epoch {epoch}")
        self.cluster_target_label.setText(
            f"{stats['target_clusters']} requested, {stats['actual_clusters']} used"
        )
        self.cluster_non_empty_label.setText(str(stats["non_empty_clusters"]))
        self.cluster_samples_label.setText(
            f"{stats['sample_count']} x {stats['embedding_dim']}D"
        )
        same_cluster_probability = stats.get("positive_same_cluster_probability")
        if same_cluster_probability is not None:
            self.cluster_pair_probability_label.setText(f"{same_cluster_probability:.4f}")
        dead_count = stats.get("dead_cluster_count")
        if dead_count is not None:
            self.dead_cluster_count_label.setText(str(dead_count))
            self.dead_sample_count_label.setText(str(len(stats.get("dead_sample_indices", []))))
        self.log.append(
            "Embedding clusters: "
            f"{stats['non_empty_clusters']} non-empty / {stats['actual_clusters']} used "
            f"from {stats['sample_count']} embeddings at epoch {epoch}. "
            f"Anchor-positive same-cluster={same_cluster_probability:.4f}."
        )

    def reset_cluster_status(self):
        self.cluster_epoch_label.setText("Not generated")
        self.cluster_target_label.setText("-")
        self.cluster_non_empty_label.setText("-")
        self.cluster_samples_label.setText("-")
        self.cluster_pair_probability_label.setText("-")
        self.dead_cluster_count_label.setText("-")
        self.dead_sample_count_label.setText("-")

    def estimate_model_size(self):
        backbone_params = 27519354
        backbone_features = 768
        cluster_count = self.cluster_count_input.value() if hasattr(self, "cluster_count_input") else NUM_CLUSTERS
        cluster_head_params = backbone_features * GLOBAL_DESCRIPTOR_DIM + GLOBAL_DESCRIPTOR_DIM
        retrieval_head_params = (backbone_features * GLOBAL_DESCRIPTOR_DIM + GLOBAL_DESCRIPTOR_DIM) * cluster_count
        head_params = cluster_head_params + retrieval_head_params
        total_params = backbone_params + head_params
        return {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "total_params": total_params,
            "fp32_mb": total_params * 4 / (1024 ** 2),
        }

    def model_size_from_model(self, model):
        backbone_params = sum(param.numel() for param in model.backbone.parameters())
        has_head = getattr(model, "_use_head", True) and hasattr(model, "cluster_head")
        head_params = sum(param.numel() for param in model.cluster_head.parameters()) if has_head else 0
        total_params = backbone_params + head_params
        return {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "total_params": total_params,
            "fp32_mb": total_params * 4 / (1024 ** 2),
        }

    def format_params(self, count):
        return f"{count:,} ({count / 1_000_000:.2f}M)"

    def update_model_size_preview(self):
        size = self.estimate_model_size()
        self.set_model_size_labels(size)

    def set_model_size_labels(self, size):
        self.backbone_size_label.setText(self.format_params(size["backbone_params"]))
        self.head_size_label.setText(self.format_params(size["head_params"]))
        self.total_size_label.setText(self.format_params(size["total_params"]))
        self.model_memory_label.setText(f"{size['fp32_mb']:.1f} MB")

    def update_plot(self):
        self.total_loss_ax.clear()
        self.total_loss_ax.set_title("Total Loss")
        self.total_loss_ax.set_xlabel("Step")
        self.total_loss_ax.set_ylabel("Loss")
        self.total_loss_ax.grid(True)
        if self.loss_records:
            steps = [record["step"] for record in self.loss_records]
            self.total_loss_ax.plot(
                steps,
                [record["total_loss"] for record in self.loss_records],
                label="Total loss",
                color="#1f77b4",
            )
            self.total_loss_ax.legend(loc="best")
        self.total_loss_figure.tight_layout()
        self.total_loss_canvas.draw_idle()

        self.ax.clear()
        self.ax.set_title("Loss Elements")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Loss")
        self.ax.grid(True)
        if self.loss_records:
            steps = [record["step"] for record in self.loss_records]
            self.ax.plot(
                steps,
                [record.get("alignment_loss", 0.0)
                 * record.get("alignment_weight", 1.0)
                 for record in self.loss_records],
                label="Alignment × w",
                color="#1f77b4",
            )
            self.ax.plot(
                steps,
                [record.get("variance_loss", 0.0)
                 * record.get("variance_weight", 1.0)
                 for record in self.loss_records],
                label="Variance × w",
                color="#2ca02c",
            )
            self.ax.plot(
                steps,
                [record.get("negative_loss", 0.0)
                 * record.get("negative_weight", 1.0)
                 for record in self.loss_records],
                label="Negative × w",
                color="#d62728",
            )
            self.ax.plot(
                steps,
                [record.get("cluster_consistency_loss", 0.0)
                 * record.get("cluster_consistency_weight", 1.0)
                 for record in self.loss_records],
                label="Consistency × w",
                color="#ff7f0e",
            )
            self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_metrics_table(self, metrics):
        rows = [
            ("Total loss", "total_loss"),
            ("Positive similarity", "positive_similarity"),
            ("Negative similarity", "negative_similarity"),
            ("Positive score", "positive_score"),
            ("Negative score", "negative_score"),
            ("Positive probability", "positive_probability"),
            ("Descriptor dim", "descriptor_dim"),
            ("Alignment loss", "alignment_loss"),
            ("Alignment weight", "alignment_weight"),
            ("Variance loss", "variance_loss"),
            ("Variance weight", "variance_weight"),
            ("Negative loss", "negative_loss"),
            ("Negative weight", "negative_weight"),
            ("Negative margin", "negative_margin"),
            ("Cluster consistency loss", "cluster_consistency_loss"),
            ("Cluster consistency weight", "cluster_consistency_weight"),
            ("Descriptor std", "descriptor_std"),
            ("Anchor-positive same cluster", "anchor_positive_same_cluster"),
        ]
        self.metrics_table.setRowCount(len(rows))
        for row, (label, key) in enumerate(rows):
            self.metrics_table.setItem(row, 0, QTableWidgetItem(label))
            value = metrics.get(key)
            text = "" if value is None else f"{value:.6f}"
            self.metrics_table.setItem(row, 1, QTableWidgetItem(text))

    def on_model_created(self, model_size):
        self.save_btn.setEnabled(True)
        self.set_model_size_labels(model_size)
        self.log.append(
            "Model size: "
            f"backbone={self.format_params(model_size['backbone_params'])}, "
            f"head={self.format_params(model_size['head_params'])}, "
            f"total={self.format_params(model_size['total_params'])}, "
            f"fp32={model_size['fp32_mb']:.1f} MB."
        )

    def on_training_finished(self, status):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.blockSignals(True)
        self.pause_btn.setChecked(False)
        self.pause_btn.setText("Pause")
        self.pause_btn.blockSignals(False)
        self.pause_btn.setEnabled(False)
        self.pause_event.clear()
        pending = getattr(self, "_paused_pull_pending", None)
        self._paused_pull_pending = None
        if pending:
            self.log_message.emit(
                f"[remote] Final checkpoint was NOT downloaded (paused): "
                f"{pending[0]}:{pending[1]} — pull manually with scp if needed.")
        self.set_parameter_inputs_enabled(True)
        if status == "finished":
            self.log.append("Training finished!")
        elif status == "stopped":
            self.log.append("Training stopped.")
        else:
            self.log.append("Training failed.")
        if status != "failed" and self.auto_eval_input.isChecked():
            run_num = getattr(self, "_last_train_run_number", None)
            if run_num is not None:
                self._launch_post_eval(run_num)

    def _launch_post_eval(self, train_run_number: int):
        import subprocess, sys as _sys
        script = Path(__file__).parent / "post_training_eval.py"
        python = _sys.executable
        self.log.append(f"Auto-eval: starting post-training eval for run #{train_run_number}...")
        try:
            proc = subprocess.Popen(
                [python, str(script), "--train_run", str(train_run_number)],
                cwd=str(Path(__file__).parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            self.log.append(f"Auto-eval: failed to launch subprocess — {e}")
            return

        def _monitor():
            for line in proc.stdout:
                self.log_message.emit(line.rstrip())
            proc.wait()
            if proc.returncode == 0:
                self.log_message.emit(f"Auto-eval: run #{train_run_number} complete — train_history.html updated.")
            else:
                self.log_message.emit(f"Auto-eval: subprocess exited with code {proc.returncode}.")

        threading.Thread(target=_monitor, daemon=True).start()

    def _ensure_local_model_loaded(self) -> bool:
        """Build+load self.model from the current checkpoint FILE if it isn't
        already resident in memory. Needed for remote training: the model
        itself only ever exists on the remote server — remote_train_loop
        never populates self.model locally, only pulls the checkpoint FILE
        after each epoch — so without this, Build Cluster Data / hard-negative
        preview / distinctiveness preview are permanently unusable whenever
        training runs remotely, even though a perfectly good checkpoint is
        sitting on disk. Loaded in eval() mode: this path is analysis/preview
        only, never trained against."""
        if self.model is not None:
            return True
        ckpt_path = self.checkpoint_path_for()
        if not ckpt_path.exists():
            return False
        try:
            from model import SwinEmbedding
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            tp = ckpt.get("training_params") or {}
            backbone = tp.get("backbone", self.backbone_input.currentData())
            cluster_count = tp.get("cluster_count", self.cluster_count_input.value())
            device = torch.device(self.device_input.currentData())
            model = SwinEmbedding(embed_dim=CLUSTER_DESCRIPTOR_DIM, pretrained=False,
                                  num_clusters=cluster_count, backbone=backbone).to(device)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
            self.model = model
            if ckpt.get("kmeans_centroids") is not None:
                self.kmeans_centroids = ckpt["kmeans_centroids"]
            self.log_message.emit(
                f"Loaded model from checkpoint for local preview/analysis "
                f"({ckpt_path.name}, backbone={backbone}).")
            return True
        except Exception as exc:
            self.log_message.emit(f"Could not load checkpoint for preview: {exc}")
            return False

    def start_build_cluster_data(self):
        if not self._ensure_local_model_loaded():
            self.cluster_excl_status.setText(
                "No model loaded and no checkpoint file found. Run or resume training first.")
            return
        if self.training_thread is not None and self.training_thread.is_alive():
            self.cluster_excl_status.setText("Main training is running — stop it first.")
            return
        self.build_cluster_data_btn.setEnabled(False)
        self.cluster_excl_status.setText("Building cluster assignments...")
        t = threading.Thread(target=self._build_cluster_data_loop, daemon=True)
        t.start()

    def _build_cluster_data_loop(self):
        try:
            from trainer import compute_embedding_cluster_stats, even_batch_size
            from torch.utils.data import DataLoader
            import torch

            params = self.training_params or {}
            device = torch.device(params.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
            cluster_count = params.get("cluster_count", self.cluster_count_input.value())

            # Dispatches by dataset_type (game4loc/university1652/sues200/
            # denseuav/crop_pairs) — previously hardcoded to SatCropDataset
            # regardless of dataset_type, so Build Cluster Data silently built
            # cluster assignments from the WRONG dataset for anything other
            # than plain crop pairs.
            dataset = self._load_dataset_for_preview()
            loader = DataLoader(dataset,
                                batch_size=even_batch_size(len(dataset), params.get("batch_size", 64)),
                                shuffle=False, num_workers=0,
                                pin_memory=device.type == "cuda")

            with self.model_lock:
                model = self.model

            prev_centroids = self.kmeans_centroids.to(device) if self.kmeans_centroids is not None else None
            use_fixed = prev_centroids is not None

            self.log_message.emit(
                f"Building cluster data: K={cluster_count}, "
                f"{'fixed centroids' if use_fixed else 'running K-means from scratch'}."
            )

            *_, cluster_sampling = compute_embedding_cluster_stats(
                model, loader, device, cluster_count,
                amp_enabled=params.get("use_amp", torch.cuda.is_available()),
                stop_event=None,
                model_lock=self.model_lock,
                return_assignments=True,
                prev_centroids=prev_centroids,
                use_fixed_centroids=use_fixed,
            )
            if cluster_sampling is None:
                self.cluster_data_build_finished.emit("failed")
                return
            self.cluster_sampling_ready.emit(cluster_sampling)
            self.cluster_data_build_finished.emit("ok")
        except Exception as exc:
            self.log_message.emit(f"Build cluster data error: {exc}")
            self.cluster_data_build_finished.emit("failed")

    def on_cluster_data_build_finished(self, status):
        self.build_cluster_data_btn.setEnabled(True)
        if status == "ok":
            self.cluster_excl_status.setText("Cluster data built — click a row to browse samples.")
        else:
            self.cluster_excl_status.setText("Cluster data build failed — see log.")

    def on_auto_k_found(self, k):
        self.cluster_count_input.setValue(k)
        self.training_params["cluster_count"] = k

    def on_cluster_sampling_ready(self, cluster_sampling):
        self.last_cluster_sampling = cluster_sampling
        self._rebuild_cluster_excl_table()

    # ── Cluster exclusion table ───────────────────────────────────────────────

    def _rebuild_cluster_excl_table(self):
        cs = self.last_cluster_sampling
        if cs is None:
            return
        members_map = cs.get("cluster_members", {})
        cluster_loss = self._cluster_table_data.get("cluster_loss", {})
        cluster_var  = self._cluster_table_data.get("cluster_variances", {})

        self.cluster_excl_table.blockSignals(True)
        self.cluster_excl_table.setRowCount(0)
        for cid in sorted(members_map.keys()):
            members = members_map[cid]
            row = self.cluster_excl_table.rowCount()
            self.cluster_excl_table.insertRow(row)

            # checkbox cell
            chk = QCheckBox()
            excluded_count = sum(1 for m in members if m in self.excluded_pair_orig_idx)
            chk.setChecked(excluded_count == 0)   # checked = included in training
            chk.stateChanged.connect(lambda state, c=cid: self._on_cluster_include_toggled(c, state))
            cell_w = QWidget()
            cell_l = QHBoxLayout(cell_w)
            cell_l.setContentsMargins(4, 0, 4, 0)
            cell_l.addWidget(chk)
            self.cluster_excl_table.setCellWidget(row, 0, cell_w)

            self.cluster_excl_table.setItem(row, 1, QTableWidgetItem(str(cid)))
            self.cluster_excl_table.setItem(row, 2, QTableWidgetItem(str(len(members))))
            loss_val = cluster_loss.get(cid, cluster_loss.get(str(cid), ""))
            self.cluster_excl_table.setItem(
                row, 3, QTableWidgetItem(f"{loss_val:.4f}" if isinstance(loss_val, float) else ""))
            var_val = cluster_var.get(cid, cluster_var.get(str(cid), ""))
            self.cluster_excl_table.setItem(
                row, 4, QTableWidgetItem(f"{var_val:.2e}" if isinstance(var_val, float) else ""))

            if excluded_count > 0:
                for col in range(1, 5):
                    item = self.cluster_excl_table.item(row, col)
                    if item:
                        item.setForeground(self.palette().color(self.palette().Disabled, self.palette().Text))

        self.cluster_excl_table.blockSignals(False)
        self._update_excl_status()

    def _on_cluster_include_toggled(self, cluster_id, state):
        cs = self.last_cluster_sampling
        if cs is None:
            return
        members = cs.get("cluster_members", {}).get(cluster_id, [])
        if state == Qt.Checked:          # include → remove from excluded set
            self.excluded_pair_orig_idx.difference_update(members)
        else:                            # exclude → add all members
            self.excluded_pair_orig_idx.update(members)
        self._update_excl_status()
        # Refresh the currently shown sample grid if it's this cluster
        self._render_cluster_samples_in_grid(cluster_id)

    def _on_cluster_table_selection_changed(self):
        rows = self.cluster_excl_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        cid_item = self.cluster_excl_table.item(row, 1)
        if cid_item is None:
            return
        try:
            cid = int(cid_item.text())
        except ValueError:
            return
        self._render_cluster_samples_in_grid(cid)

    def _render_cluster_samples_in_grid(self, cluster_id):
        cs = self.last_cluster_sampling
        if cs is None:
            return
        members = cs.get("cluster_members", {}).get(cluster_id, [])
        dataset = getattr(self, "_cached_dataset_for_excl", None)
        if dataset is None:
            try:
                dataset = self._load_dataset_for_preview()
                self._cached_dataset_for_excl = dataset
            except Exception:
                return

        while self.dead_cluster_grid.count():
            item = self.dead_cluster_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        cols = 4
        shown = 0
        for orig_idx in members[:32]:
            if orig_idx >= len(dataset.pairs):
                continue
            anchor_path = Path(dataset.pairs[orig_idx][0])
            is_excluded = orig_idx in self.excluded_pair_orig_idx

            container = QWidget()
            cl = QVBoxLayout(container)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(2)

            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            px = QPixmap(str(anchor_path))
            if not px.isNull():
                img_label.setPixmap(px.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                img_label.setText("(err)")
            if is_excluded:
                img_label.setStyleSheet("opacity:0.4; border:2px solid #d62728;")

            name_lbl = QLabel(anchor_path.name)
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setWordWrap(True)
            name_lbl.setStyleSheet("font-size:9px;")

            excl_btn = QPushButton("✕ Exclude" if not is_excluded else "↩ Include")
            excl_btn.setFixedHeight(20)
            excl_btn.setStyleSheet(
                "background:#d62728;color:white;font-size:9px;" if not is_excluded
                else "background:#2ca02c;color:white;font-size:9px;"
            )
            excl_btn.clicked.connect(
                lambda _, idx=orig_idx, cid=cluster_id: self._toggle_pair_exclusion(idx, cid))

            cl.addWidget(img_label)
            cl.addWidget(name_lbl)
            cl.addWidget(excl_btn)
            self.dead_cluster_grid.addWidget(container, shown // cols, shown % cols)
            shown += 1

        excl_in_cluster = sum(1 for m in members if m in self.excluded_pair_orig_idx)
        self.cluster_excl_status.setText(
            f"Cluster {cluster_id}: {len(members)} pairs, "
            f"{excl_in_cluster} excluded — showing {shown}."
        )
        self.cluster_excl_status.setStyleSheet("")

    def _toggle_pair_exclusion(self, orig_idx, cluster_id):
        if orig_idx in self.excluded_pair_orig_idx:
            self.excluded_pair_orig_idx.discard(orig_idx)
        else:
            self.excluded_pair_orig_idx.add(orig_idx)
        self._update_excl_status()
        self._render_cluster_samples_in_grid(cluster_id)
        self._refresh_table_row_for_cluster(cluster_id)

    def _refresh_table_row_for_cluster(self, cluster_id):
        cs = self.last_cluster_sampling
        if cs is None:
            return
        members = cs.get("cluster_members", {}).get(cluster_id, [])
        excluded_count = sum(1 for m in members if m in self.excluded_pair_orig_idx)
        for row in range(self.cluster_excl_table.rowCount()):
            item = self.cluster_excl_table.item(row, 1)
            if item and int(item.text()) == cluster_id:
                chk_w = self.cluster_excl_table.cellWidget(row, 0)
                if chk_w:
                    chk = chk_w.findChild(QCheckBox)
                    if chk:
                        chk.blockSignals(True)
                        chk.setChecked(excluded_count == 0)
                        chk.blockSignals(False)
                break

    def _update_excl_status(self):
        n = len(self.excluded_pair_orig_idx)
        if n == 0:
            self.cluster_excl_status.setText("No exclusions — all pairs included in training.")
            self.cluster_excl_status.setStyleSheet("color:#555;font-style:italic;")
        else:
            self.cluster_excl_status.setText(
                f"{n} pair(s) excluded from training. Exclusions take effect on the next training start.")
            self.cluster_excl_status.setStyleSheet("color:#d62728;font-weight:bold;")

    def _clear_all_exclusions(self):
        self.excluded_pair_orig_idx.clear()
        self._cached_dataset_for_excl = None
        self._rebuild_cluster_excl_table()
        while self.dead_cluster_grid.count():
            item = self.dead_cluster_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _load_dataset_for_preview(self):
        params = self.training_params or {}
        ds_type = params.get("dataset_type", "crop_pairs")
        from model import backbone_img_size
        backbone = params.get("backbone", "swin_t")
        if ds_type == "game4loc":
            from dataset import GtaUavDataset
            gta_json = params.get("gta_json", "same-area-drone2sate-train.json")
            return GtaUavDataset(
                data_root=params.get("gta_data_root", ""),
                pairs_meta_file=gta_json,
                mode=params.get("gta_mode", "pos"),
                img_size=backbone_img_size(backbone),
                augment=False,
            )
        if ds_type == "university1652":
            from dataset import University1652Dataset
            return University1652Dataset(
                train_root=params.get("u1652_train_root",
                                      r"D:\UAV_DATASET\university-1652\University-Release\train"),
                img_size=backbone_img_size(backbone),
                augment=False,
            )
        if ds_type == "sues200":
            from dataset import Sues200Dataset
            return Sues200Dataset(
                train_root=params.get("sues200_train_root",
                                      r"D:\UAV_DATASET\SUES-200-split\train"),
                img_size=backbone_img_size(backbone),
                augment=False,
            )
        if ds_type == "denseuav":
            from dataset import DenseUAVDataset
            return DenseUAVDataset(
                train_root=params.get("denseuav_train_root",
                                      r"D:\UAV_DATASET\DenseUAV\DenseUAV\train"),
                img_size=backbone_img_size(backbone),
                augment=False,
                cross_altitude=params.get("denseuav_cross_altitude", True),
                altitude_weight_tau=(None if params.get("denseuav_altitude_full_strength", False)
                                    else params.get("denseuav_altitude_weight_tau", 20.0)),
            )
        from dataset import SatCropDataset
        return SatCropDataset(params.get("dataset_root", DATASET_ROOT), augment=False)

    # =========================================================================
    # Scale Head tab
    # =========================================================================

    def _build_scale_head_tab(self):
        tab = QWidget()
        root = QHBoxLayout(tab)

        # ── Left: params ──────────────────────────────────────────────────────
        left = QWidget()
        left.setMaximumWidth(360)
        left_layout = QVBoxLayout(left)

        paths_grp = QGroupBox("Checkpoint & Data")
        pform = QFormLayout(paths_grp)

        # ── Model source ──────────────────────────────────────────────────
        from PyQt5.QtWidgets import QComboBox as _QCB2
        self.sh_source_combo = _QCB2()
        self.sh_source_combo.addItem("Load existing checkpoint", "checkpoint")
        self.sh_source_combo.addItem("Pretrained backbone (new model)", "backbone")
        pform.addRow("Model source", self.sh_source_combo)

        # ── Checkpoint row (visible when source=checkpoint) ───────────────
        self.sh_checkpoint_input = QLineEdit()
        self.sh_checkpoint_input.setPlaceholderText("checkpoints/latest_swin_b.pt")
        ckpt_btn = QPushButton("…")
        ckpt_btn.setFixedWidth(28)
        ckpt_btn.clicked.connect(self._sh_browse_checkpoint)
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(self.sh_checkpoint_input)
        ckpt_row.addWidget(ckpt_btn)
        self._sh_ckpt_label = QLabel("Checkpoint")
        pform.addRow(self._sh_ckpt_label, ckpt_row)

        # ── Backbone row (visible when source=backbone) ────────────────────
        self.sh_new_backbone_combo = _QCB2()
        self.sh_new_backbone_combo.addItem("Swin-T  (28 M, 224×224)",          "swin_t")
        self.sh_new_backbone_combo.addItem("Swin-B  (88 M, 384×384)",          "swin_b")
        self.sh_new_backbone_combo.addItem("SwinV2-B  (88 M, 384×384)",        "swinv2_b")
        self.sh_new_backbone_combo.addItem("ConvNeXt-B  (89 M, 384×384, 1024-D)", "convnext_b")
        self.sh_new_backbone_combo.addItem("ConvNeXt-T  (28 M, 384×384, 768-D)", "convnext_t")
        self.sh_new_backbone_combo.addItem("ViT-B/16  (86 M, 224×224)",        "vit_b")
        self.sh_new_backbone_combo.addItem("DINOv2-B/14  (86 M, 224×224)",     "dinov2_b")
        self._sh_backbone_label = QLabel("Backbone")
        pform.addRow(self._sh_backbone_label, self.sh_new_backbone_combo)
        self._sh_backbone_label.setVisible(False)
        self.sh_new_backbone_combo.setVisible(False)

        def _on_sh_source(idx):
            is_ckpt = (self.sh_source_combo.itemData(idx) == "checkpoint")
            self._sh_ckpt_label.setVisible(is_ckpt)
            self.sh_checkpoint_input.setVisible(is_ckpt)
            ckpt_btn.setVisible(is_ckpt)
            self._sh_backbone_label.setVisible(not is_ckpt)
            self.sh_new_backbone_combo.setVisible(not is_ckpt)

        self.sh_source_combo.currentIndexChanged.connect(_on_sh_source)

        from PyQt5.QtWidgets import QComboBox as _QCB
        self.sh_dataset_type_input = _QCB()
        self.sh_dataset_type_input.addItem("VisLoc altitude CSV", "visloc")
        self.sh_dataset_type_input.addItem("Scale crop dataset (scale_gt.csv)", "scale_crop")
        self.sh_dataset_type_input.addItem("SUES-200  (drone_view_512/NNNN/ALT/)", "sues200")
        self.sh_dataset_type_input.addItem("DenseUAV (multi-altitude drone/satellite)", "denseuav")
        pform.addRow("Dataset type", self.sh_dataset_type_input)

        self.sh_data_root_input = QLineEdit("D:/UAV_DATASET/VisLoc")
        data_btn = QPushButton("…")
        data_btn.setFixedWidth(28)
        data_btn.clicked.connect(self._sh_browse_data_root)
        data_row = QHBoxLayout()
        data_row.addWidget(self.sh_data_root_input)
        data_row.addWidget(data_btn)
        self._sh_data_root_label = pform.addRow("Dataset root", data_row)

        _PLACEHOLDERS = {
            "visloc":     "D:/UAV_DATASET/VisLoc",
            "scale_crop": "path/to/scale_dataset",
            "sues200":    "D:/UAV_DATASET/SUES-200-512x512",
            "denseuav":   "D:/UAV_DATASET/DenseUAV/DenseUAV",
        }

        def _on_sh_dataset_type(idx):
            dtype = self.sh_dataset_type_input.itemData(idx)
            self.sh_data_root_input.setPlaceholderText(
                _PLACEHOLDERS.get(dtype, "path/to/dataset"))

        self.sh_dataset_type_input.currentIndexChanged.connect(_on_sh_dataset_type)
        _on_sh_dataset_type(0)

        left_layout.addWidget(paths_grp)

        hp_grp = QGroupBox("Hyper-parameters")
        hform = QFormLayout(hp_grp)

        self.sh_epochs_input = QSpinBox()
        self.sh_epochs_input.setRange(1, 500)
        self.sh_epochs_input.setValue(20)
        hform.addRow("Epochs", self.sh_epochs_input)

        self.sh_batch_size_input = QSpinBox()
        self.sh_batch_size_input.setRange(1, 256)
        self.sh_batch_size_input.setValue(32)
        hform.addRow("Batch size", self.sh_batch_size_input)

        self.sh_lr_input = QDoubleSpinBox()
        self.sh_lr_input.setDecimals(6)
        self.sh_lr_input.setRange(1e-6, 1.0)
        self.sh_lr_input.setSingleStep(1e-4)
        self.sh_lr_input.setValue(1e-3)
        hform.addRow("Learning rate", self.sh_lr_input)

        self.sh_wd_input = QDoubleSpinBox()
        self.sh_wd_input.setDecimals(5)
        self.sh_wd_input.setRange(0.0, 1.0)
        self.sh_wd_input.setValue(1e-4)
        hform.addRow("Weight decay", self.sh_wd_input)

        self.sh_amp_input = QCheckBox()
        self.sh_amp_input.setChecked(torch.cuda.is_available())
        hform.addRow("AMP", self.sh_amp_input)

        self.sh_autosave_input = QCheckBox()
        self.sh_autosave_input.setChecked(True)
        self.sh_autosave_input.setToolTip(
            "Merge updated scale_head weights back into the checkpoint file after training.")
        hform.addRow("Save checkpoint after training", self.sh_autosave_input)

        sh_btn_row = QHBoxLayout()
        self.sh_start_btn = QPushButton("Start Scale Head Training")
        self.sh_start_btn.clicked.connect(self._sh_start)
        self.sh_stop_btn = QPushButton("Stop")
        self.sh_stop_btn.setEnabled(False)
        self.sh_stop_btn.clicked.connect(self._sh_stop)
        sh_btn_row.addWidget(self.sh_start_btn)
        sh_btn_row.addWidget(self.sh_stop_btn)
        left_layout.insertLayout(0, sh_btn_row)

        left_layout.addWidget(hp_grp)
        left_layout.addStretch()

        root.addWidget(left)

        # ── Right: monitor ────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)

        self.sh_fig = Figure(figsize=(5, 3), tight_layout=True)
        self.sh_ax  = self.sh_fig.add_subplot(111)
        self.sh_ax.set_xlabel("Step")
        self.sh_ax.set_ylabel("Huber loss")
        self.sh_ax.set_title("Scale head training loss")
        self.sh_canvas = FigureCanvas(self.sh_fig)
        right_layout.addWidget(self.sh_canvas, stretch=2)

        self.sh_epoch_label = QLabel("—")
        self.sh_epoch_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.sh_epoch_label)

        self.sh_log = QTextEdit()
        self.sh_log.setReadOnly(True)
        right_layout.addWidget(self.sh_log, stretch=1)

        root.addWidget(right, stretch=1)
        return tab

    # ── Scale head slots ──────────────────────────────────────────────────────

    def _sh_on_log(self, msg: str):
        self.sh_log.append(msg)

    def _sh_on_loss(self, epoch: int, val: float):
        if epoch != self.sh_current_epoch:
            if self.sh_losses:
                self.sh_epoch_boundaries.append(len(self.sh_losses))
            self.sh_current_epoch = epoch
            self.sh_epoch_label.setText(
                f"Epoch {epoch} / {self.sh_epochs_input.value()}")
        self.sh_losses.append(val)
        self._sh_redraw_plot()

    def _sh_on_done(self, status: str):
        self.sh_start_btn.setEnabled(True)
        self.sh_stop_btn.setEnabled(False)
        self.sh_log.append(f"\nScale head training {status}.")
        self._save_ui_settings()

    def _sh_redraw_plot(self):
        ax = self.sh_ax
        ax.clear()
        ax.set_xlabel("Step")
        ax.set_ylabel("Huber loss")
        ax.plot(self.sh_losses, linewidth=0.8, color="#4C8AF7", label="raw")
        for b in self.sh_epoch_boundaries:
            ax.axvline(b, color="gray", linewidth=0.5, linestyle="--")
        if len(self.sh_losses) > 20:
            w = max(10, len(self.sh_losses) // 20)
            smooth = [
                sum(self.sh_losses[max(0, i - w):i + 1]) / min(i + 1, w + 1)
                for i in range(len(self.sh_losses))
            ]
            ax.plot(smooth, linewidth=1.5, color="#E85858", label="smoothed")
        ax.legend(fontsize=8)
        self.sh_canvas.draw_idle()

    # ── Scale head actions ────────────────────────────────────────────────────

    def _sh_start(self):
        if self.sh_thread is not None and self.sh_thread.is_alive():
            self.sh_log.append("Already running.")
            return
        self.sh_losses.clear()
        self.sh_epoch_boundaries.clear()
        self.sh_current_epoch = 0
        self.sh_ax.clear()
        self.sh_canvas.draw()
        self.sh_log.clear()
        self.sh_stop_event.clear()
        self.sh_start_btn.setEnabled(False)
        self.sh_stop_btn.setEnabled(True)
        self._save_ui_settings()
        self.sh_thread = threading.Thread(target=self._sh_worker, daemon=True)
        self.sh_thread.start()

    def _sh_stop(self):
        self.sh_stop_event.set()
        self.sh_stop_btn.setEnabled(False)
        self.sh_log.append("Stop requested…")

    def _sh_worker(self):
        try:
            from scale_head_trainer import train_scale_head
            device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            source    = self.sh_source_combo.currentData()
            is_new    = (source == "backbone")
            autosave  = self.sh_autosave_input.isChecked()

            if is_new:
                from model import SwinEmbedding
                backbone_key = self.sh_new_backbone_combo.currentData()
                self.sh_log_signal.emit(
                    f"Initialising {backbone_key} with pretrained weights…")
                model     = SwinEmbedding(backbone=backbone_key, pretrained=True).to(device)
                ckpt_path = Path("checkpoints") / f"scale_{backbone_key}.pt"
                ckpt_path.parent.mkdir(exist_ok=True)
                self.sh_log_signal.emit(f"Will save to: {ckpt_path}")
            else:
                from visloc_eval import _load_model
                ckpt_path = Path(self.sh_checkpoint_input.text().strip())
                self.sh_log_signal.emit(f"Loading checkpoint: {ckpt_path.name}")
                model = _load_model(ckpt_path, device)

            if not hasattr(model, "scale_head"):
                self.sh_log_signal.emit(
                    "ERROR: model has no scale_head — try reloading the checkpoint.")
                self.sh_done_signal.emit("failed")
                return

            n_params = sum(p.numel() for p in model.scale_head.parameters())
            self.sh_log_signal.emit(
                f"Scale head: {n_params:,} params.  Backbone + retrieval head frozen.")

            def _epoch_done(ep):
                if autosave:
                    self._sh_save_checkpoint(model, ckpt_path, device, is_new=is_new)
                    self.sh_log_signal.emit(f"  [autosave] epoch {ep} → {ckpt_path.name}")

            ok = train_scale_head(
                model            = model,
                data_root        = self.sh_data_root_input.text().strip(),
                dataset_type     = self.sh_dataset_type_input.currentData(),
                device           = device,
                epochs           = self.sh_epochs_input.value(),
                batch_size       = self.sh_batch_size_input.value(),
                lr               = self.sh_lr_input.value(),
                weight_decay     = self.sh_wd_input.value(),
                amp_enabled      = self.sh_amp_input.isChecked(),
                status_callback  = lambda m: self.sh_log_signal.emit(m),
                loss_callback    = lambda ep, _, v: self.sh_loss_signal.emit(ep, v),
                epoch_callback   = _epoch_done,
                stop_event       = self.sh_stop_event,
            )
            if not ok:
                self.sh_done_signal.emit("failed")
                return

            if self.sh_stop_event.is_set() and autosave:
                self._sh_save_checkpoint(model, ckpt_path, device, is_new=is_new)
                self.sh_log_signal.emit("  [autosave] partial progress saved on stop.")

            self.sh_done_signal.emit("finished")

        except Exception as exc:
            import traceback
            self.sh_log_signal.emit(f"ERROR: {exc}\n{traceback.format_exc()}")
            self.sh_done_signal.emit("failed")

    def _sh_save_checkpoint(self, model, ckpt_path: Path, device, is_new: bool = False):
        try:
            if is_new or not ckpt_path.exists():
                # Fresh model — save full state dict with backbone metadata
                ckpt = {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "training_params":  {"backbone": getattr(model, "backbone_name", "swin_t")},
                }
            else:
                # Existing checkpoint — replace scale_head weights entirely.
                # Remove ALL old scale_head keys first (may be from a different architecture),
                # then add the current model's scale_head keys.  This prevents stale keys
                # (e.g. scale_head.0.weight from an older flat architecture) from persisting
                # alongside current keys and triggering the architecture-change warning on load.
                ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)
                state     = ckpt.get("model_state_dict", ckpt)
                new_state = model.state_dict()
                state = {k: v for k, v in state.items() if not k.startswith("scale_head.")}
                for k, v in new_state.items():
                    if k.startswith("scale_head."):
                        state[k] = v.cpu()
                if "model_state_dict" in ckpt:
                    ckpt["model_state_dict"] = state
                else:
                    ckpt = state
            torch.save(ckpt, ckpt_path)
            self.sh_log_signal.emit(f"Saved → {ckpt_path.name}")
        except Exception as exc:
            self.sh_log_signal.emit(f"Save failed: {exc}")

    def _sh_browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select checkpoint", "checkpoints", "PyTorch (*.pt *.pth)")
        if path:
            self.sh_checkpoint_input.setText(path)

    def _sh_browse_data_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select VisLoc root", self.sh_data_root_input.text())
        if path:
            self.sh_data_root_input.setText(path)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec_())
