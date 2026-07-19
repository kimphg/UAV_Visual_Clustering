import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Game4Loc-style losses
# ---------------------------------------------------------------------------

def weighted_info_nce(query, positive, tau=0.07, label_smoothing=0.0, k_smooth=5.0,
                      positive_weights=None, return_metrics=False):
    """Symmetric InfoNCE with per-sample IoU label smoothing (Game4Loc WeightedInfoNCE).

    When positive_weights is provided (IoU scores in [0,1]):
        eps_i = 1 - (1 - label_smoothing) / (1 + exp(-k_smooth * w_i))
    High IoU → eps → 0 (hard positive). Low IoU → eps → label_smoothing (soft uniform).
    When positive_weights is None, eps = label_smoothing for all samples.
    label_smoothing=0 + no weights → identical to standard symmetric InfoNCE.
    """
    q = F.normalize(query, dim=-1)
    p = F.normalize(positive, dim=-1)
    logits = q @ p.T / tau
    n = logits.shape[0]

    if positive_weights is not None:
        eps = 1.0 - (1.0 - label_smoothing) / (
            1.0 + torch.exp(-k_smooth * positive_weights.to(dtype=q.dtype, device=q.device))
        )
    else:
        eps = logits.new_full((n,), label_smoothing)

    def _wloss(sim, eps_vec):
        lse = torch.logsumexp(sim, dim=1)        # (n,)
        diag = sim.diagonal()                    # (n,)
        mean_row = sim.mean(dim=1)               # (n,)
        per_sample = (1.0 - eps_vec) * (-diag + lse) + eps_vec * (-mean_row + lse)
        return per_sample.mean()

    loss = (_wloss(logits, eps) + _wloss(logits.T, eps)) / 2.0

    if not return_metrics:
        return loss

    with torch.no_grad():
        probs = torch.softmax(logits.float(), dim=1)
        metrics = {
            "total_loss": loss.detach().item(),
            "positive_similarity": logits.diagonal().mean().item() * tau,
            "positive_probability": probs.diagonal().mean().item(),
            "label_smoothing_mean": eps.mean().item(),
        }
    return loss, metrics


def group_whole_slice_info_nce(query, positive, group_len, label_smoothing=0.0, tau=0.07,
                                return_metrics=False):
    """Game4Loc GroupInfoNCE whole_slice variant.

    query / positive: [G*N, D] where G = number of groups, N = group_len.
    Group g occupies rows [g*N : (g+1)*N].  For each row i in group g, all N
    entries in group g of *positive* are treated as positives.

    Vectorised — no Python loops over individual samples.
    """
    q = F.normalize(query, dim=-1)
    p = F.normalize(positive, dim=-1)
    GN = q.shape[0]
    N = group_len
    G = GN // N

    logits = q @ p.T / tau  # [G*N, G*N]

    group_ids = torch.arange(GN, device=q.device) // N       # [G*N]
    pos_mask = group_ids.unsqueeze(1) == group_ids.unsqueeze(0)  # [G*N, G*N]

    def _loss_dir(sim):
        all_lse = torch.logsumexp(sim, dim=1)                         # [G*N]
        pos_sim = sim.masked_fill(~pos_mask, float("-inf"))
        pos_lse = torch.logsumexp(pos_sim, dim=1)                     # [G*N]
        # Hard term
        loss_hard = (1.0 - label_smoothing) * (-pos_lse + all_lse).mean()
        # Soft term: eps/G * sum_g (-logsumexp(sim[i,g*N:(g+1)*N]) + all_lse[i])
        sim_by_group = sim.view(GN, G, N)                             # [G*N, G, N]
        g_lse = torch.logsumexp(sim_by_group, dim=2)                  # [G*N, G]
        loss_soft = (label_smoothing / G) * (-g_lse + all_lse.unsqueeze(1)).sum(dim=1).mean()
        return loss_hard + loss_soft

    loss = (_loss_dir(logits) + _loss_dir(logits.T)) / 2.0

    if not return_metrics:
        return loss

    with torch.no_grad():
        metrics = {
            "total_loss": loss.detach().item(),
            "group_len": N,
            "n_groups": G,
        }
    return loss, metrics


# ---------------------------------------------------------------------------
# Game4Loc SDM (Soft Distance Metric) — vectorised, returns per-query scores
# ---------------------------------------------------------------------------

def compute_sdm_scores(distances_m, k, s=0.001):
    """Compute SDM@k for a single query given sorted top-K distances in metres.

    SDM@k = Σ_{i=0}^{k-1}  (k-i) * exp(-s*d_i)  /  Σ_{i=0}^{k-1} (k-i)

    Returns a scalar in (0, 1].  d_i should be already sorted ascending by rank,
    i.e. distances_m[0] is top-1, distances_m[1] is rank-2, etc.
    """
    k = min(k, len(distances_m))
    weights = [k - i for i in range(k)]
    nom = sum(w * ((-s * d) if isinstance(d, float) else float((-s * d))) for w, d in
              zip(weights, distances_m[:k]))
    # use exp manually
    import math
    nom = sum(w * math.exp(-s * float(d)) for w, d in zip(weights, distances_m[:k]))
    den = sum(weights)
    return nom / den if den > 0 else 0.0

