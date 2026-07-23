"""Tests for the feature-space operations in dustrack.imagesimilarity.

These are the embedder-agnostic half -- selection, K-NN, clustering over an
(N, D) feature array -- so they run on synthetic features with a clear ground
truth (three well-separated blobs) and never touch DINOv3 or a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

from dustrack import imagesimilarity as ims


def three_blobs(per=8, d=16, spread=0.15, seed=0):
    """Three tight, well-separated blobs. Returns (features, blob_id)."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((3, d))
    centers[0, 0] = 10.0
    centers[1, 1] = 10.0
    centers[2, 2] = 10.0
    X, y = [], []
    for c in range(3):
        X.append(centers[c] + rng.normal(0, spread, (per, d)))
        y += [c] * per
    return np.concatenate(X), np.array(y)


class TestFarthestPointSample:
    def test_spreads_across_the_blobs(self):
        X, y = three_blobs()
        idx = ims.farthest_point_sample(X, 3, start=0)
        assert len(set(y[idx])) == 3            # one pick from each blob

    def test_returns_all_when_n_exceeds_pool(self):
        X, _ = three_blobs(per=2)
        idx = ims.farthest_point_sample(X, 100)
        assert sorted(idx) == list(range(len(X)))

    def test_deterministic_given_start(self):
        X, _ = three_blobs()
        a = ims.farthest_point_sample(X, 5, start=0)
        b = ims.farthest_point_sample(X, 5, start=0)
        assert a.tolist() == b.tolist()

    def test_first_pick_is_the_start(self):
        X, _ = three_blobs()
        assert ims.farthest_point_sample(X, 4, start=7)[0] == 7


class TestKnn:
    def test_neighbours_come_from_the_query_blob(self):
        X, y = three_blobs(per=8)
        q = X[y == 1].mean(0)                   # centre of blob 1
        idx, sim = ims.knn(X, q, k=5)
        assert (y[idx[0]] == 1).all()           # all neighbours in blob 1
        assert np.all(np.diff(sim[0]) <= 1e-9)  # sorted nearest-first

    def test_shapes_for_multiple_queries(self):
        X, _ = three_blobs()
        idx, sim = ims.knn(X, X[:4], k=3)
        assert idx.shape == (4, 3) and sim.shape == (4, 3)

    def test_k_capped_at_pool_size(self):
        X, _ = three_blobs(per=2)
        idx, _ = ims.knn(X, X[0], k=100)
        assert idx.shape[1] == len(X)


class TestCluster:
    def test_kmeans_recovers_the_blobs(self):
        X, y = three_blobs()
        lab = ims.cluster(X, 3, method="kmeans")
        # each true blob maps to a single predicted cluster
        assert all(len(set(lab[y == c])) == 1 for c in range(3))
        assert len(set(lab)) == 3

    def test_agglomerative_recovers_the_blobs(self):
        X, y = three_blobs()
        lab = ims.cluster(X, 3, method="agglomerative")
        assert all(len(set(lab[y == c])) == 1 for c in range(3))

    def test_unknown_method_raises(self):
        X, _ = three_blobs()
        with pytest.raises(ValueError):
            ims.cluster(X, 3, method="nope")
