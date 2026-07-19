import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# Token-layout helpers (Swin vs ViT)
# ---------------------------------------------------------------------------

def _to_spatial_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Normalise backbone output to [B, H, W, C] for the scale head.

    Swin  → already [B, H, W, C], returned as-is.
    ViT   → [B, N+1, C] (CLS at index 0); strip CLS, reshape to square grid.
    """
    if tokens.dim() == 4:
        return tokens
    B, N1, C = tokens.shape
    N = N1 - 1
    H = W = int(N ** 0.5)
    return tokens[:, 1:].reshape(B, H, W, C)


def _global_features(tokens: torch.Tensor) -> torch.Tensor:
    """Extract global feature vector from backbone token output.

    Swin  → mean over spatial dims [B, H, W, C] → [B, C].
    ViT   → CLS token [B, N+1, C] → [B, C].
    Equivalent to backbone(x) with num_classes=0 for both architectures.
    """
    if tokens.dim() == 4:
        return tokens.mean(dim=(1, 2))
    return tokens[:, 0]


# ---------------------------------------------------------------------------
# Multi-scale spatial pooling scale head
# ---------------------------------------------------------------------------

class _EarlyStageScaleHead(nn.Module):
    """Predicts log(geographic_width_metres) from early backbone stage features.

    Uses features from the first N-1 stages of Swin (before the final, most
    scale-invariant stage) or from intermediate blocks of ViT/DINOv2.  Early
    features retain spatial-frequency and texture cues that encode the apparent
    size of ground objects — information that is progressively discarded by the
    deeper, semantics-focused layers.

    Each stage's feature map is globally average-pooled, projected to PROJ_DIM,
    then the projections are concatenated and passed through an MLP → scalar.

    stage_feats: list of [B, H_i, W_i, C_i] tensors (Swin) or [B, N+1, C_i]
    tensors (ViT) — _to_spatial_tokens normalises both before pooling.
    """

    PROJ_DIM = 128

    def __init__(self, stage_dims: list):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Linear(d, self.PROJ_DIM) for d in stage_dims
        ])
        concat_dim = self.PROJ_DIM * len(stage_dims)
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(concat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, stage_feats: list) -> torch.Tensor:
        """stage_feats: list of [B, H_i, W_i, C_i] or [B, N+1, C_i] tensors."""
        parts = []
        for feat, proj in zip(stage_feats, self.projs):
            spatial = _to_spatial_tokens(feat)      # [B, H_i, W_i, C_i]
            pooled  = spatial.mean(dim=(1, 2))      # [B, C_i]  global avg pool
            parts.append(proj(pooled))              # [B, PROJ_DIM]
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)  # [B]


NUM_CLUSTERS = 16
GLOBAL_DESCRIPTOR_DIM = 512
CLUSTER_DESCRIPTOR_DIM = 512

# Supported backbones: timm model name + native input resolution
BACKBONE_CONFIGS = {
    "swin_t": {
        "timm_name": "swin_tiny_patch4_window7_224",
        "img_size":  224,
    },
    "swin_b": {
        # Swin-B pretrained on ImageNet-22K, fine-tuned on IN-1K — same as Game4Loc
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "img_size":  384,
    },
    "swin_b_nohead": {
        # Swin-B with NO projection head: raw 1024-D L2-normalised backbone output.
        # Exactly matches Game4Loc's architecture (no Linear→GELU layer).
        "timm_name": "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
        "img_size":  384,
        "no_head":   True,
    },
    "swinv2_b": {
        # Swin Transformer V2 Base — scaled cosine attention, log-spaced pos bias,
        # post-norm.  Pre-trained IN-22K at 192px, fine-tuned IN-1K at 384px
        # (window 12→24).  Mentioned in Game4Loc paper as comparison backbone.
        # forward_features returns [B, 12, 12, 1024] same as swin_b.
        "timm_name": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "img_size":  384,
    },
    "convnext_b": {
        # ConvNeXt-Base — Facebook IN-22K pretrained, fine-tuned on IN-1K at 384px.
        # Matches Game4Loc's exact variant (convnext_base.fb_in22k_ft_in1k_384).
        # forward_features returns [B, 1024, H, W] (channels-first, unlike Swin).
        # backbone.stages gives 4 stages with dims [128, 256, 512, 1024].
        # no_img_size_arg: ConvNeXt's create_model does not accept img_size.
        # no_head: raw 1024-D L2-norm output, matching Game4Loc's architecture.
        "timm_name": "convnext_base.fb_in22k_ft_in1k_384",
        "img_size":  384,
        "no_img_size_arg": True,
        "no_head":   True,
    },
    "convnext_t": {
        # ConvNeXt-Tiny — same IN-22K→IN-1K@384 pretraining lineage as convnext_b
        # but 28 M params (~3x smaller). Same methodology: raw L2-norm backbone
        # output, no projection head. forward_features returns [B, 768, 12, 12]
        # (channels-first); stage 2 gives 24x24x384 — both grids mirror Swin-B's
        # 12x12/24x24 layout, so VQ re-rank and geo verification carry over as-is.
        "timm_name": "convnext_tiny.fb_in22k_ft_in1k_384",
        "img_size":  384,
        "no_img_size_arg": True,
        "no_head":   True,
    },
    "vit_b": {
        # ViT-Base/16 — ImageNet-21K pretrained, fine-tuned on IN-1K.
        # forward_features returns [B, 197, 768] (CLS + 14×14 patch tokens).
        "timm_name": "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        "img_size":  224,
    },
    "dinov2_b": {
        # DINOv2 ViT-Base/14 — self-supervised on LVD-142M, strong dense features.
        # forward_features returns [B, 257, 768] (CLS + 16×16 patch tokens at 224px).
        "timm_name": "vit_base_patch14_dinov2.lvd142m",
        "img_size":  224,
    },
}

def backbone_img_size(backbone: str = "swin_t") -> int:
    """Return the recommended input resolution for a backbone key."""
    return BACKBONE_CONFIGS[backbone]["img_size"]


def safe_normalize(x: torch.Tensor, dim: int = -1, min_norm: float = 1.0) -> torch.Tensor:
    """L2-normalize with a floor on the norm to prevent amplifying low-energy embeddings.

    Featureless tiles (uniform ocean, solid sky) produce near-zero backbone activations.
    Plain F.normalize divides by a tiny number, blowing the direction up to the unit sphere
    at an arbitrary angle — making such tiles spuriously similar to unrelated clusters.

    With min_norm, any vector whose norm < min_norm is NOT amplified to unit length; it
    stays at magnitude (actual_norm / min_norm) < 1.  Normal tiles (norm >> min_norm) are
    still normalized to ~unit sphere as before.  Tune min_norm against a norm histogram of
    your gallery embeddings; 1.0 works for typical Linear+GELU head outputs.
    """
    norm = x.norm(dim=dim, keepdim=True).clamp(min=min_norm)
    return x / norm


class SwinEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim=CLUSTER_DESCRIPTOR_DIM,
        pretrained=True,
        num_clusters=NUM_CLUSTERS,
        global_dim=GLOBAL_DESCRIPTOR_DIM,
        backbone: str = "swin_t",
    ):
        super().__init__()
        if embed_dim != CLUSTER_DESCRIPTOR_DIM:
            raise ValueError(f"Cluster descriptor dimension is fixed to {CLUSTER_DESCRIPTOR_DIM}.")
        if backbone not in BACKBONE_CONFIGS:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(BACKBONE_CONFIGS)}")

        self.backbone_name = backbone
        self.num_clusters = num_clusters  # informational only; no per-cluster weight layers
        self.global_dim = global_dim
        self.cluster_dim = embed_dim
        cfg = BACKBONE_CONFIGS[backbone]
        _create_kwargs = {"pretrained": pretrained, "num_classes": 0}
        if not cfg.get("no_img_size_arg"):
            _create_kwargs["img_size"] = cfg["img_size"]
        self.backbone = timm.create_model(cfg["timm_name"], **_create_kwargs)
        # _use_head controls the cluster projection head only ("nohead" = raw backbone for retrieval).
        # The scale head is separate and created for ALL backbone variants.
        self._use_head = not cfg.get("no_head", False)
        if self._use_head:
            self.cluster_head = nn.Sequential(
                nn.Linear(self.backbone.num_features, global_dim),
                nn.GELU(),
            )

        # ── Scale head: hooks on early backbone stages (all variants incl. nohead) ──
        # Swin / SwinV2: backbone.layers — channels-last [B, H, W, C] output.
        # ConvNeXt:       backbone.stages — channels-first [B, C, H, W] output.
        # ViT / DINOv2:  backbone.blocks — sequence [B, N+1, C] output.
        if hasattr(self.backbone, "layers") and not hasattr(self.backbone, "blocks"):
            all_stages = list(self.backbone.layers)
            self._scale_hook_modules = all_stages[:-1]
            base_dim = self.backbone.embed_dim
            scale_stage_dims = [base_dim * (2 ** i)
                                for i in range(len(self._scale_hook_modules))]
            self._backbone_channels_first = False
        elif hasattr(self.backbone, "stages"):
            # ConvNeXt: 4 stages, hook first N-1; dims inferred via a dummy forward.
            all_stages = list(self.backbone.stages)
            self._scale_hook_modules = all_stages[:-1]
            self._backbone_channels_first = True
            with torch.no_grad():
                _dummy = torch.zeros(1, 3, cfg["img_size"], cfg["img_size"])
                _caps: list = []
                def _cap_hook(mod, inp, out, _c=_caps):
                    _c.append(out)
                _hs = [m.register_forward_hook(_cap_hook) for m in self._scale_hook_modules]
                try:
                    self.backbone.forward_features(_dummy)
                finally:
                    for _h in _hs:
                        _h.remove()
            scale_stage_dims = [t.shape[1] for t in _caps]   # channels-first → dim 1
        else:
            all_blocks = list(self.backbone.blocks)
            n = len(all_blocks)
            idx_a = max(0, n // 3 - 1)
            idx_b = max(idx_a + 1, 2 * n // 3 - 1)
            self._scale_hook_modules = [all_blocks[idx_a], all_blocks[idx_b]]
            scale_stage_dims = [self.backbone.embed_dim] * len(self._scale_hook_modules)
            self._backbone_channels_first = False
        self.scale_head = _EarlyStageScaleHead(scale_stage_dims)

    def _collect_early_features(self, x: torch.Tensor):
        """Single forward pass that captures early stage activations via hooks.

        Returns (early_feats: list[Tensor], final_tokens: Tensor).
        early_feats[i] is the output of self._scale_hook_modules[i].
        final_tokens is the full backbone.forward_features output used for retrieval.
        Hooks are registered and removed within this call — safe for concurrent use.
        """
        captured = []

        def _hook(module, input, output):
            captured.append(output)

        handles = [m.register_forward_hook(_hook) for m in self._scale_hook_modules]
        try:
            final_tokens = self.backbone.forward_features(x)
        finally:
            for h in handles:
                h.remove()

        if self._backbone_channels_first:
            # ConvNeXt returns [B, C, H, W]; normalise to [B, H, W, C] so
            # _to_spatial_tokens and _global_features work identically to Swin.
            captured = [t.permute(0, 2, 3, 1).contiguous() for t in captured]
            final_tokens = final_tokens.permute(0, 2, 3, 1).contiguous()

        return captured, final_tokens

    def predict_log_scale(self, x: torch.Tensor):
        """Predict log(metres) from a raw image tensor [B,3,H,W]. Returns [B]."""
        early_feats, _ = self._collect_early_features(x)
        return self.scale_head(early_feats)

    def encode_spatial_features(self, x: torch.Tensor, stage: int = 2):
        """Retrieval feats [B,C] + per-location spatial tokens for geometric shift
        verification (fine matching: estimate query<->gallery transform, reject if
        implied real-world shift is too large). Shares one backbone pass with the
        scale head via the same early-stage hooks. Returns (feats [B,C],
        spatial [B,H,W,C_stage] channels-last, already normalised to [B,H,W,C]
        regardless of backbone via _collect_early_features's channels-first
        handling).

        stage: which spatial grid to return —
          2 (default): LAST hooked early stage (stage 2 for Swin/SwinV2: 512-d,
              24x24 grid at 384x384 input).
          3: the final backbone stage before global pooling, i.e. final_tokens
              itself (1024-d, 12x12 grid at 384x384 input for Swin-B) — coarser
              spatial resolution but higher-capacity per-location descriptor.
              Free: final_tokens is already computed for `feats` below, so
              stage=3 costs nothing extra beyond stage=2.
        """
        early_feats, final_tokens = self._collect_early_features(x)
        feats = _global_features(final_tokens)
        spatial = final_tokens if stage == 3 else early_feats[-1]
        return feats, spatial

    def encode_features_and_scale(self, x: torch.Tensor):
        """Single backbone pass → (retrieval_feats [B,C], log_scale [B]).

        Captures early-stage features for scale via hooks during the main forward pass,
        so both retrieval and scale share one backbone call.  Works for all backbone
        variants including nohead (retrieval_feats = raw 1024-D backbone output).
        """
        early_feats, final_tokens = self._collect_early_features(x)
        feats = _global_features(final_tokens)   # [B, C]
        scale = self.scale_head(early_feats)      # [B]
        return feats, scale

    def load_state_dict(self, state_dict, strict=True):
        """Load weights; scale_head keys are filtered by shape so stale architecture
        keys (wrong shape or not in current model) are silently dropped rather than
        causing the entire scale_head to be discarded."""
        model_state = self.state_dict()
        # Pass non-scale-head keys through unchanged; keep only shape-compatible scale_head keys.
        filtered = {}
        stale_count = 0
        for k, v in state_dict.items():
            if k.startswith("scale_head."):
                if k in model_state and model_state[k].shape == v.shape:
                    filtered[k] = v
                else:
                    stale_count += 1   # wrong shape or key not in current model — drop
            else:
                filtered[k] = v
        scale_missing = [k for k in model_state if k.startswith("scale_head.") and k not in filtered]
        if stale_count or scale_missing:
            print("WARNING: scale_head weights skipped (architecture changed). "
                  "Please retrain the scale head.", flush=True)
        # strict=False because some scale_head model keys may be absent from filtered;
        # non-scale-head keys are passed through unchanged so backbone/head loading is unaffected.
        return super().load_state_dict(filtered, strict=False)

    def encode_features(self, x):
        return self.backbone(x)

    def encode_global(self, x):
        return self.encode_cluster_head(x)

    def encode_cluster_head(self, x):
        features = self.encode_features(x)
        return self.encode_cluster_from_features(features)

    def encode_retrieval(self, x, centroids=None):
        features = self.encode_features(x)
        cluster_membership = self._cluster_membership(features, centroids)
        return self.encode_retrieval_from_features(features, cluster_membership)

    def encode_cluster_from_features(self, features):
        if self._use_head:
            return safe_normalize(self.cluster_head(features), dim=-1)
        return safe_normalize(features, dim=-1)

    def encode_retrieval_from_features(self, features, cluster_ids):
        """Retrieval embedding — delegates to cluster head (retrieval_heads removed)."""
        return self.encode_cluster_from_features(features)

    def encode_retrieval(self, x, centroids=None, cluster_ids=None):
        features = self.encode_features(x)
        if cluster_ids is None:
            cluster_desc = self.encode_cluster_from_features(features)
            if centroids is not None:
                cluster_ids = torch.matmul(cluster_desc, centroids.T).argmax(dim=-1)
            else:
                cluster_ids = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        return self.encode_retrieval_from_features(features, cluster_ids)

    def _cluster_membership(self, features, centroids):
        if centroids is None:
            return None
        cluster_desc = self.encode_cluster_from_features(features)
        return torch.matmul(cluster_desc, centroids.T)

    def encode_cluster_from_global(self, global_desc, cluster_ids):
        return F.normalize(global_desc, dim=-1)

    def forward(self, x, cluster_ids=None, return_global=False, centroids=None):
        features = self.encode_features(x)
        global_desc = self.encode_cluster_from_features(features)
        if return_global:
            return global_desc
        if cluster_ids is None:
            if centroids is not None:
                cluster_ids = torch.matmul(global_desc, centroids.T).argmax(dim=-1)
            else:
                cluster_ids = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        return self.encode_retrieval_from_features(features, cluster_ids)
