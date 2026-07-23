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

__all__ = ["farthest_point_sample", "knn", "cluster"]


def _l2_normalized(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


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
