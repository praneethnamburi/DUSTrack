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

import os

import numpy as np

__all__ = ["dino_embed", "farthest_point_sample", "select_diverse", "knn",
           "cluster", "cluster_medoids", "select_within_radius",
           "DINOV3_SMALL", "DINOV2_SMALL", "ACTIVE_MODEL", "local_dinov3_usable"]

#: The production feature space (Corazon's ultrasound result): DINOv3 ViT-S/16,
#: ~21M params, chosen to fit the paper's 8 GB consumer-GPU constraint. It is
#: a **gated** HF repo -- request access once at the model page and the stored
#: token unlocks it.
DINOV3_SMALL = "facebook/dinov3-vits16-pretrain-lvd1689m"

#: Non-gated, same-size (22M), same API drop-in -- the fallback when no local
#: DINOv3 weights are present (e.g. CI, or a machine without the share).
DINOV2_SMALL = "facebook/dinov2-small"

#: Local Meta-native DINOv3 weights (a ``.pth`` state dict, loaded via
#: torch.hub's ``facebookresearch/dinov3`` -- NOT the transformers format).
#: Defaults to Corazon's ViT-B/16 on the P: share; override with the
#: ``DUSTRACK_DINOV3_WEIGHTS`` env var. The gated HF path (:data:`DINOV3_SMALL`)
#: is no longer needed while this is available.
DINOV3_LOCAL_WEIGHTS = os.environ.get(
    "DUSTRACK_DINOV3_WEIGHTS",
    r"P:\Roger\dino\dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
)

#: torch.hub model id for the local DINOv3-B -- the ``dinov3:<entry>`` scheme
#: :func:`dino_embed` routes to the hub loader.
DINOV3_B_LOCAL = "dinov3:vitb16"

def local_dinov3_usable() -> bool:
    """Whether the local DINOv3-B path can actually load here: the weights are
    on disk AND the facebookresearch/dinov3 hubconf's load-time deps are present
    (``torch`` + ``torchmetrics`` + ``termcolor`` -- the hubconf imports the
    segmentation stack at load, so the weights alone aren't enough).

    Single source of truth for BOTH :data:`ACTIVE_MODEL` and the GUI's button
    gate, so they never disagree: a weights-only environment must fall back to
    DINOv2, not enable the button and then crash on the hubconf's torchmetrics
    import (the failure mode this predicate closes).
    """
    import importlib.util

    return os.path.exists(DINOV3_LOCAL_WEIGHTS) and all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "torchmetrics", "termcolor"))


#: The model :func:`dino_embed` uses when a caller doesn't name one: the local
#: **DINOv3-B** (the production ultrasound feature space) when it's fully
#: loadable here, else non-gated **DINOv2-Small**. Resolved once at import so a
#: set of embeddings has unambiguous provenance (the two are different variants;
#: the M3 comparison depends on knowing which).
ACTIVE_MODEL = DINOV3_B_LOCAL if local_dinov3_usable() else DINOV2_SMALL

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


def _load_hub(entry: str, weights_path: str, device: str):
    key = ("hub", entry, str(weights_path), device)
    if key not in _MODEL_CACHE:
        from pathlib import Path

        import torch

        url = "file:///" + Path(weights_path).as_posix()
        model = torch.hub.load(
            "facebookresearch/dinov3", entry, weights=url, trust_repo=True
        ).eval().to(device)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _embed_hub(images, *, model_id, batch_size, device, normalize):
    """Embed via a Meta-native DINOv3 loaded through torch.hub (local weights).

    ``model_id`` is ``dinov3:<entry>`` (e.g. ``dinov3:vitb16``); weights come
    from :data:`DINOV3_LOCAL_WEIGHTS`. The native model ships no processor, so
    preprocessing is here: RGB, resize 224, ImageNet-normalize; the forward
    returns the global (class) token.
    """
    import cv2
    import torch

    entry = "dinov3_" + str(model_id).split(":", 1)[1]
    model = _load_hub(entry, DINOV3_LOCAL_WEIGHTS, device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    feats = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            arrs = []
            for im in images[i : i + batch_size]:
                a = np.asarray(im)
                if a.dtype != np.uint8:
                    a = (a * 255 if a.max() <= 1.5 else a).clip(0, 255).astype(np.uint8)
                if a.ndim == 2:
                    a = cv2.cvtColor(a, cv2.COLOR_GRAY2RGB)
                arrs.append(cv2.resize(a, (224, 224)).astype(np.float32) / 255.0)
            t = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).to(device)
            out = model((t - mean) / std)
            feats.append(out.float().cpu().numpy())
    out = np.concatenate(feats, axis=0) if feats else np.empty((0, 0))
    return _l2_normalized(out) if normalize and len(out) else out


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
    if str(model_id).startswith("dinov3:"):        # local Meta-native via torch.hub
        return _embed_hub(images, model_id=model_id, batch_size=batch_size,
                          device=device, normalize=normalize)
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
    return_radii: bool = False,
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

    ``return_radii`` also returns, per pick, its **capture radius** -- the
    distance from that point to the set already chosen when it was picked
    (``inf`` for the seed / any preselected). Being greedy FPS, this sequence
    is non-increasing, so thresholding it at a minimum distance yields a
    prefix: this is what lets the review modal turn "minimum distance between
    kept frames" into a data-driven count in one pass (see
    :func:`select_within_radius`).
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be (N, D)")
    N = len(X)
    n = min(int(n), N)
    if not return_radii and n >= N:
        return np.arange(N)
    if normalize:
        X = _l2_normalized(X)

    if preselected is not None and len(preselected):
        chosen = list(dict.fromkeys(int(i) for i in preselected))
        radii = [float("inf")] * len(chosen)
        if len(chosen) >= n:
            order = np.array(chosen[:n])
            return (order, np.array(radii[:n])) if return_radii else order
        dist = np.full(N, np.inf)
        for p in chosen:
            dist = np.minimum(dist, np.linalg.norm(X - X[p], axis=1))
    else:
        if start is None:
            start = int(np.random.default_rng(seed).integers(N))
        chosen = [int(start)]
        radii = [float("inf")]
        dist = np.linalg.norm(X - X[chosen[0]], axis=1)

    while len(chosen) < n:
        i = int(np.argmax(dist))
        radii.append(float(dist[i]))          # distance to the set before adding i
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
    order = np.array(chosen)
    return (order, np.array(radii)) if return_radii else order


def cluster_medoids(features, labels, *, normalize: bool = True) -> dict:
    """The medoid -- the frame nearest its cluster's mean -- per cluster.

    The canonical image that represents each appearance group in the review
    modal (a real frame, unlike the centroid). Returns ``{label: index}`` over
    the rows of ``features``; ``labels`` is the per-row cluster id from
    :func:`cluster`.
    """
    X = np.asarray(features, dtype=float)
    if normalize:
        X = _l2_normalized(X)
    labels = np.asarray(labels)
    medoids = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        centroid = X[members].mean(axis=0)
        d = np.linalg.norm(X[members] - centroid, axis=1)
        medoids[int(c)] = int(members[int(np.argmin(d))])
    return medoids


def select_within_radius(radii, min_dist, *, order=None, floor=()) -> list:
    """The frames kept when no two selected frames may sit closer than
    ``min_dist`` in feature space -- the data-driven count behind the review
    modal's one knob.

    ``radii`` are the non-increasing capture radii from
    ``farthest_point_sample(..., return_radii=True)``, so ``radii >= min_dist``
    is a prefix of the farthest-point order: loosening ``min_dist`` admits more
    (closer) frames, and dense/redundant regions -- covered early -- contribute
    few while diverse regions keep earning picks. ``order`` maps prefix
    positions back to original indices (returned as such when given). ``floor``
    (e.g. each cluster's medoid) is always included, so every appearance group
    keeps at least its canonical even when it's globally redundant. Returns a
    sorted list of indices.
    """
    radii = np.asarray(radii, dtype=float)
    keep = np.where(radii >= float(min_dist))[0]
    sel = {int(order[i]) for i in keep} if order is not None else {int(i) for i in keep}
    sel |= {int(i) for i in floor}
    return sorted(sel)


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
