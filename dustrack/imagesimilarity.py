"""DINOv3-feature image-similarity index for DUSTrack.

The foundational layer of the general-model / labeling-consistency workflow:
embed ultrasound image patches into a feature space where *visual* similarity
is meaningful, then select, sort, and cluster frames by distance in that
space. Per Corazon's empirical result DINOv3 is that space for ultrasound,
which is why this replaces the ResNet18 / imagehash / SSIM methods.

The module is split so the feature *source* and the feature-space *operations*
are independent:

* the **operations** here -- farthest-point sampling, K-NN, clustering -- take
  an ``(N, D)`` array of features and know nothing about DINOv3, so they are
  testable on their own and reusable with any embedder;
* :func:`dino_embed` is the DINOv3(-Small) source that produces those features.

Three consumers build on this: general-model / decimation frame selection
(farthest-point sampling over the annotated pool -- the diverse subset to
train on), the cross-frame consistency assistant (K-NN to a query frame's
prior labels), and the bistability label-conflict scan (clustering to find
similar frames whose labels diverge). The blip / LK-consistency feature is
handled separately in :mod:`dustrack.flow_consistency`.
"""
from __future__ import annotations

import numpy as np

__all__ = ["dino_embed", "farthest_point_sample", "select_diverse", "knn",
           "cluster", "DINOV3_SMALL", "DINOV2_SMALL", "ACTIVE_MODEL"]

#: The production feature space (Corazon's ultrasound result): DINOv3 ViT-S/16,
#: ~21M params, chosen to fit the paper's 8 GB consumer-GPU constraint. It is
#: a **gated** HF repo -- request access once at the model page and the stored
#: token unlocks it.
DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"

#: Non-gated, same-size (22M), same API drop-in. Use it to run the pipeline
#: while DINOv3 access is pending -- switching back is one argument.
DINOV2_SMALL = "facebook/dinov2-small"

#: The model :func:`dino_embed` uses when a caller doesn't name one. It is
#: **DINOv2 while DINOv3 access is pending** -- flip this one line to
#: ``DINOV3_SMALL`` the moment access lands. Kept as an explicit constant
#: rather than a silent try-v3-fall-back-to-v2 so a set of embeddings always
#: has an unambiguous provenance (v2 and v3 are different variants, and the
#: general-model M3 comparison depends on knowing which produced it).
ACTIVE_MODEL = DINOV2_SMALL

#: (processor, model, device) memoized per model id so a batch job loads once.
_MODEL_CACHE: dict = {}


def _l2_normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _to_pil(img):
    """A gray ``(H, W)`` or ``(H, W, 3)`` array / PIL image -> RGB PIL."""
    from PIL import Image

    if isinstance(img, Image.Image):
        return img.convert("RGB")
    a = np.asarray(img)
    if a.dtype != np.uint8:                     # tolerate float [0,1] or [0,255]
        a = (a * 255 if a.max() <= 1.5 else a).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray(a)


def _load(model_id: str, device: str):
    key = (model_id, device)
    if key not in _MODEL_CACHE:
        from transformers import AutoImageProcessor, AutoModel

        proc = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval().to(device)
        _MODEL_CACHE[key] = (proc, model)
    return _MODEL_CACHE[key]


def dino_embed(
    images,
    *,
    model_id: "str | None" = None,
    batch_size: int = 32,
    device: "str | None" = None,
    pool: str = "cls",
    normalize: bool = False,
):
    """DINOv3 features for a sequence of images -> ``(N, D)`` array.

    The feature *source* for the index -- embed the ultrasound frames (or
    label-centred patches) whose similarity everything downstream reasons
    about. ``images`` are gray ``(H, W)`` or ``(H, W, 3)`` arrays (or PIL
    images), whatever the caller has; the processor handles resize and
    normalization.

    ``pool`` is ``"cls"`` (the global class token -- the whole-image
    descriptor decimation and M3 selection want) or ``"mean"`` (mean of the
    patch tokens). ``model_id`` defaults to :data:`ACTIVE_MODEL` (DINOv2
    while DINOv3 access is pending). transformers and torch are imported
    lazily, so importing this module never requires them (DUSTrack's
    standalone-import invariant).
    """
    import torch

    model_id = model_id or ACTIVE_MODEL
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    proc, model = _load(model_id, device)

    imgs = [_to_pil(im) for im in images]
    feats = []
    with torch.no_grad():
        for i in range(0, len(imgs), batch_size):
            batch = imgs[i : i + batch_size]
            inputs = proc(images=batch, return_tensors="pt").to(device)
            hidden = model(**inputs).last_hidden_state       # (B, 1+regs+patches, D)
            vec = hidden[:, 0] if pool == "cls" else hidden[:, 1:].mean(dim=1)
            feats.append(vec.float().cpu().numpy())
    out = np.concatenate(feats, axis=0) if feats else np.empty((0, 0))
    return _l2_normalized(out) if normalize and len(out) else out


def farthest_point_sample(
    features,
    n: int,
    *,
    start: "int | None" = 0,
    preselected=None,
    normalize: bool = True,
    seed: "int | None" = None,
) -> np.ndarray:
    """``n`` indices maximally spread out in feature space (greedy FPS).

    The selection primitive behind decimation and the general model's M3
    frame choice: pick a subset that *covers* the appearance variety of the
    annotated pool rather than sampling it uniformly in time, so redundant
    near-duplicate frames (the bulk of a dense refinement layer) don't
    dominate the training set. Each pick is the point farthest from every
    point already chosen.

    ``normalize`` L2-normalizes first, so distance is cosine (the right
    metric for DINOv3 features). ``start`` seeds the first pick; ``None``
    draws it at random from ``seed`` (the rest are deterministic given the
    first). ``preselected`` (indices already in the set) makes FPS *continue*
    from them -- the new picks are farthest from everything already chosen,
    and the returned list includes the preselected up front. This is how a
    coverage floor or a cluster share feeds back into one global spread.
    Returns fewer than ``n`` only if the pool is smaller.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be (N, D)")
    N = len(X)
    if n >= N:
        return np.arange(N)
    if normalize:
        X = _l2_normalized(X)

    if preselected is not None and len(preselected):
        chosen = list(dict.fromkeys(int(i) for i in preselected))
        if len(chosen) >= n:
            return np.array(chosen[:n])
        dist = np.full(N, np.inf)
        for p in chosen:
            dist = np.minimum(dist, np.linalg.norm(X - X[p], axis=1))
    else:
        if start is None:
            start = int(np.random.default_rng(seed).integers(N))
        chosen = [int(start)]
        dist = np.linalg.norm(X - X[chosen[0]], axis=1)

    while len(chosen) < n:
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
    return np.array(chosen)


def knn(features, queries, k: int, *, normalize: bool = True):
    """The ``k`` most similar rows of ``features`` to each query.

    Cosine similarity by default (``normalize``). The K-NN behind the
    cross-frame consistency assistant: at labeling time, surface the ``k``
    previously-labelled frames that look most like the current one so the
    same anatomical point is placed consistently across non-adjacent motion
    repeats. Returns ``(indices (Q, k), similarity (Q, k))``, nearest first.
    """
    X = np.asarray(features, dtype=float)
    Q = np.atleast_2d(np.asarray(queries, dtype=float))
    if normalize:
        X = _l2_normalized(X)
        Q = _l2_normalized(Q)
    sim = Q @ X.T
    k = min(k, X.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    top = np.take_along_axis(sim, idx, axis=1)
    return idx, top


def cluster(
    features,
    n_clusters: int,
    *,
    method: str = "kmeans",
    normalize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Cluster frames in feature space -- the substrate of the label-conflict
    scan (group visually-similar frames, then flag ones whose labels diverge).

    ``method`` is ``"kmeans"`` or ``"agglomerative"`` (cosine average-linkage).
    Returns a length-``N`` array of integer cluster labels. sklearn is
    imported lazily so it is not a hard dependency of importing this module.
    """
    X = np.asarray(features, dtype=float)
    if normalize:
        X = _l2_normalized(X)
    if method == "kmeans":
        from sklearn.cluster import KMeans

        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(X)
    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        return AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        ).fit_predict(X)
    raise ValueError(f"unknown method {method!r}")


def _balanced_alloc(sizes: dict, total: int) -> dict:
    """Split ``total`` across keys as evenly as possible, capped by each
    key's ``size``; leftover from a small key is redistributed to the rest.
    (The same equalizing allocation the label-harvest budget uses.)"""
    alloc = {c: 0 for c in sizes}
    pool = {c for c, s in sizes.items() if s > 0}
    remaining = total
    while pool and remaining > 0:
        share = max(1, remaining // len(pool))
        progressed = False
        for c in sorted(pool):
            take = min(share, sizes[c] - alloc[c], remaining)
            if take <= 0:
                pool.discard(c)
                continue
            alloc[c] += take
            remaining -= take
            progressed = True
            if alloc[c] >= sizes[c]:
                pool.discard(c)
        if not progressed:
            break
    return alloc


def _cluster_balanced_fps(X, m, *, exclude, n_clusters, normalize, seed):
    """Give each appearance cluster an equal share of ``m`` picks (FPS within
    each), so one dominant look can't swamp the selection."""
    N = len(X)
    k = min(N, n_clusters or max(2, min(m, 8)))
    lab = cluster(X, k, normalize=normalize, seed=seed)
    avail = {c: [int(i) for i in np.where(lab == c)[0] if i not in exclude]
             for c in range(k)}
    alloc = _balanced_alloc({c: len(v) for c, v in avail.items()}, m)
    picks: list[int] = []
    for c, take in alloc.items():
        if take and avail[c]:
            sub = np.array(avail[c])
            loc = farthest_point_sample(X[sub], take, normalize=normalize)
            picks.extend(int(sub[j]) for j in loc)
    return picks


def select_diverse(
    features,
    n: int,
    *,
    groups=None,
    min_per_group: int = 0,
    cluster_balance: bool = False,
    n_clusters: "int | None" = None,
    normalize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Select ``n`` diverse row indices of ``features`` -- the selection core.

    Fed by any frame-set generator (all labels across iterations, one video's
    interpolated stretches, a blip set) and routed to any consumer (a new
    DLC project, a new layer): this function only sees embeddings and returns
    which rows to keep.

    * **Base** -- farthest-point sampling: appearance spread, which already
      under-samples dense near-duplicate groups.
    * **Coverage floor** -- ``groups`` (an array parallel to ``features``,
      e.g. a participant id per frame) plus ``min_per_group`` guarantees at
      least that many picks from each group, so the general model never drops
      a participant entirely even when two participants look alike. Omit it
      for a single-video decimation.
    * **Cluster balance** -- ``cluster_balance`` gives each appearance cluster
      an equal share of the budget (FPS within each) rather than letting FPS
      alone decide, a stronger guard against one look dominating and against
      outlier frames stealing the budget.

    The floor is taken first; the remainder fills by continuing the same FPS
    spread (or the cluster-balanced version) so the two constraints compose.
    """
    X = np.asarray(features, dtype=float)
    N = len(X)
    if n >= N:
        return np.arange(N)

    chosen: list[int] = []
    if groups is not None and min_per_group > 0:
        groups = np.asarray(groups)
        for g in np.unique(groups):
            if len(chosen) >= n:
                break
            idx_g = np.where(groups == g)[0]
            take = min(min_per_group, len(idx_g), n - len(chosen))
            loc = farthest_point_sample(X[idx_g], take, normalize=normalize)
            chosen.extend(int(idx_g[j]) for j in loc)
        chosen = list(dict.fromkeys(chosen))

    remaining = n - len(chosen)
    if remaining > 0:
        if cluster_balance:
            chosen.extend(_cluster_balanced_fps(
                X, remaining, exclude=set(chosen),
                n_clusters=n_clusters, normalize=normalize, seed=seed))
        else:
            chosen = [int(i) for i in farthest_point_sample(
                X, n, preselected=chosen or None, normalize=normalize)]
    return np.array([int(i) for i in chosen[:n]])
