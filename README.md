# UAV_Visual_Clustering

**Cluster-Routed Descriptor Learning with Structured Hard-Negative Mining for UAV Cross-View Localization**

*Kim-Phuong Phung, Quang-Uy Nguyen — Le Quy Don Technical University*
*Paper currently in preparation.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository is the official code release for the paper above. It presents
a **cluster-routed descriptor pipeline** for UAV-to-satellite cross-view
localization. A pretrained Swin-B or ConvNeXt backbone's raw, L2-normalized
output — with **no projection head** — serves directly as the retrieval
descriptor for three roles at once: FAISS K-means routing over a large
satellite-tile gallery, candidate scoring, and training supervision.

The central idea is that the **same K-means partition used for routing at
evaluation time also organizes hard-negative mining during training**:
negatives are drawn from a query's own cluster and its nearest neighboring
clusters (ranked by descriptor similarity), with co-positive exclusion so
that two views of the same place are never pushed apart as false negatives.
This concentrates the contrastive signal on the visually repetitive
look-alike candidates — nearby rooftops, road grids, fields — where retrieval
actually fails, instead of on the mostly-uninformative random negatives a
flat in-batch InfoNCE objective would draw.

At evaluation time, a query is coarsely routed to its nearest map clusters
(cutting candidate scoring by roughly `K/K_c`), then optionally re-ranked
with a **vector-quantized semantic re-ranking** stage: each gallery tile's
intermediate spatial feature grid is quantized against a shared K-means
codebook into a tiny (144-byte) "semantic map," and top candidates are
re-scored by rotation-invariant cell-wise agreement with the query's own map
— recovering the spatial-arrangement information that global average pooling
discards. This stage is effective specifically on **densely tiled galleries**
(tiles sampled far more frequently than their own footprint, so top
candidates are near-duplicate overlapping views); it is a no-op or mild
regression on distinct-location galleries, and is reported only where it
helps.

Full method details, all losses, and every experimental protocol are in
[`paper/main.tex`](paper/main.tex).

## Key results

**DenseUAV** (drone→satellite, same-area, full 18,198-tile confusion gallery):

| Method | Backbone | Params | R@1 | R@5 | SDM@1 |
|---|---|---|---|---|---|
| DenseUAV baseline | ViT-S | ~22M | 83.01 | 95.58 | 86.50 |
| MCCG | ConvNeXt-T | ~28M | 83.14 | 93.39 | 85.94 |
| CEUSP | ConvNeXt-T | ~28M | 89.45 | 96.05 | **91.01** |
| **Ours** | Swin-B | ~88M | 86.62 | 96.83 | 89.50 |
| **Ours + VQ re-rank** | Swin-B | ~88M | 88.55 | 96.61 | — |
| **Ours** | ConvNeXt-T | ~28M | 82.41 | 94.59 | 85.83 |
| **Ours + VQ re-rank** | ConvNeXt-T | ~28M | **91.89** | **96.83** | — |
| **Ours** | ConvNeXt-B | ~89M | 84.56 | 95.02 | 87.45 |
| **Ours + VQ re-rank** | ConvNeXt-B | ~89M | 91.29 | 96.48 | — |

**SUES-200** (official 120/80 split, 80-tile gallery, R@1 by altitude):

| Method | Backbone | 150m | 200m | 250m | 300m |
|---|---|---|---|---|---|
| Sample4Geo | ConvNeXt-B | 94.75 | 96.75 | 97.25 | 97.20 |
| Game4Loc | ConvNeXt-B | 94.62 | 96.55 | 97.55 | 97.67 |
| **Ours** | Swin-B | **96.30** | **98.25** | **99.00** | 99.15 |
| **Ours** | ConvNeXt-T | 91.77 | 95.20 | 96.90 | 97.52 |
| **Ours** | ConvNeXt-B | 94.97 | 97.28 | 98.50 | **99.35** |

**University-1652** (R@1 / AP, official trapezoidal AP, junk excluded):

| Method | Backbone | D→S R@1 | D→S AP | S→D R@1 | S→D AP |
|---|---|---|---|---|---|
| Sample4Geo | ConvNeXt-B | 92.65 | 93.81 | 95.14 | 91.39 |
| **Ours** | Swin-B | **94.52** | **95.44** | **96.86** | **93.83** |
| **Ours** | ConvNeXt-T | 82.08 | 84.69 | 92.44 | 80.38 |
| **Ours** | ConvNeXt-B | 90.91 | 92.44 | 95.72 | 90.32 |

Full tables (all compared methods, both SUES-200 gallery conventions,
per-altitude breakdowns) are in the paper.

## Datasets & leaderboards

| Dataset | Official repository |
|---|---|
| DenseUAV | [Dmmm1997/DenseUAV](https://github.com/Dmmm1997/DenseUAV) |
| SUES-200 | [Reza-Zhu/SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark) |
| University-1652 | [layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline) (README includes the official leaderboard) |
| GTA-UAV / Game4Loc | [Yux1angJi/GTA-UAV](https://github.com/Yux1angJi/GTA-UAV) |
| UAV-VisLoc | [IntelliSensing/UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) |

## Repository layout

| File | Role |
|---|---|
| `model.py` | Backbone wrapper (`SwinEmbedding`): Swin-B / ConvNeXt-T / ConvNeXt-B, raw L2-normalized descriptor, no projection head |
| `trainer.py` | Training loop: paired InfoNCE, variance regularization, cluster-structured hard-negative mining, cluster-consistency loss |
| `clustering.py` | FAISS K-means wrapper (`ClusterIndex`) and fixed-centroid assignment used for routing and negative sampling |
| `loss.py` | Core contrastive losses (`weighted_info_nce`, `group_whole_slice_info_nce`) |
| `dataset.py` | Dataset classes for DenseUAV, SUES-200, University-1652, and Game4Loc/UAV-VisLoc-tiled training pairs |
| `train.py` | PyQt5 desktop app: local/remote training launch, live loss & cluster diagnostics, checkpoint management |
| `remote_train.py` | Headless training entry point (`--config config.json --run-dir <dir>`) used for remote/background runs |
| `general_eval_gui.py` | PyQt5 evaluation app: DenseUAV / SUES-200 / University-1652 / Game4Loc-VisLoc protocols, VQ re-rank, geometric verification |
| `geo_verify.py` | Vector-quantized semantic re-ranking: codebook fitting, spatial quantization, cell-wise agreement scoring |
| `prepare_visloc.py` | One-time conversion of raw UAV-VisLoc into the Game4Loc tiling/JSON format used for training |
| `monitor.py` | Lightweight continuous GPU/RAM/process health logger for long remote training runs |
| `paper/` | LaTeX source of the paper (`main.tex`, `references.bib`) |

This repo intentionally ships only the code path that produced the results
above. A larger, in-progress research workspace (ablations, alternative
architectures that were tried and set aside, GUI variants) exists alongside
this project but is not part of the reproducible release. `train.py` and
`general_eval_gui.py` are the full research GUIs and retain a few optional,
legacy tabs (e.g. an early per-cluster-head scale predictor) whose supporting
modules were intentionally left out of this trim since they are unrelated to
the paper's method — those specific buttons are inactive here, but every
training and evaluation path described in this README is fully self-contained.

## Installation

```bash
git clone https://github.com/kimphg/UAV_Visual_Clustering.git
cd UAV_Visual_Clustering
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training and evaluation; `requirements.txt`
pins a CUDA 12.6 PyTorch build (`--extra-index-url` line) — adjust the
`torch`/`torchvision` versions to match your local CUDA toolkit if needed.

## Data preparation

- **DenseUAV / SUES-200 / University-1652**: download from their official
  sources (see paper references) and point the relevant `*_train_root` /
  dataset-root fields at the extracted folders — no conversion needed.
- **UAV-VisLoc / Game4Loc-style training**: run the one-time tiling step
  first:
  ```bash
  python prepare_visloc.py --data-root /path/to/VisLoc
  ```
  This produces the `same-area-drone2sate-{train,test}.json` /
  `cross-area-drone2sate-{train,test}.json` pairing files and pre-tiled
  satellite images consumed by training and evaluation.

## Usage

**Training (GUI):**
```bash
python train.py
```
Configure backbone, dataset, cluster count, and hard-negative-mining
settings in the GUI; supports both local and remote (SSH) launches.

**Training (headless / scripted):**
```bash
python remote_train.py --config config.json --run-dir ./runs/my_run
```
See `train.py`'s remote-launch code path for the exact config schema (backbone,
learning rate, unfreeze depth, cluster count, dataset type/roots, etc.).

**Evaluation (GUI):**
```bash
python general_eval_gui.py
```
Select the dataset (DenseUAV / SUES-200 / University-1652 / Game4Loc-VisLoc),
point it at a checkpoint, and run. VQ re-rank and geometric verification are
available as optional post-processing stages on the coarse ranking.

## Checkpoints

Trained checkpoints are not committed to this repository (multi-GB
binaries). See the paper for the exact per-dataset fine-tuning recipe used to
produce every reported number; checkpoints can be reproduced from ImageNet-
or DenseUAV-pretrained weights following those recipes, or requested from the
authors.

## Citation

<!-- Venue/volume pending: paper is in preparation, update journal/year and
     add a DOI or arXiv eprint field once available. -->
```bibtex
@article{phung2026clusterrouted,
  title   = {Cluster-Routed Descriptor Learning with Structured Hard-Negative
             Mining for UAV Cross-View Localization},
  author  = {Phung, Kim-Phuong and Nguyen, Quang-Uy},
  journal = {TBD},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

This work builds on and compares against DenseUAV, SUES-200, University-1652,
Game4Loc/GTA-UAV, UAV-VisLoc, CEUSP, MCCG, CAMP, Sample4Geo, and other prior
cross-view localization methods cited in the paper.
