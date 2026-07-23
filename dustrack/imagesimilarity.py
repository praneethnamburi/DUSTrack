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

__all__ = ["dino_embed", "farthest_point_sample", "knn", "cluster",
           "DINOV3_SMALL", "DINOV2_SMALL", "ACTIVE_MODEL"]

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
    first). Returns fewer than ``n`` only if the pool is smaller.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim != 2:
        raise ValueError("features must be (N, D)")
    N = len(X)
    if n >= N:
        return np.arange(N)
    if normalize:
        X = _l2_normalized(X)
    if start is None:
        start = int(np.random.default_rng(seed).integers(N))

    chosen = [int(start)]
    dist = np.linalg.norm(X - X[start], axis=1)
    for _ in range(n - 1):
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
