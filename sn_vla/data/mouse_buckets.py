"""Non-uniform log-bucket + residual encoding for mouse deltas.

Locked spec (plan.md v3 §3):
  raw delta → signed-log compress → coarse bucket classification + residual regression
  Center region (aiming micro-adjustment) gets dense buckets, tails (flick shots) get coarse.
"""

from __future__ import annotations

import numpy as np


class MouseBucketer:
    """Signed-log non-uniform bucket + residual refinement.

    The transform in the compressed domain is:
        f(x) = sign(x) * log(1 + |x| / scale) / log(1 + d_max / scale)
    which maps [-d_max, d_max] → [-1, 1] with denser resolution near zero.
    Buckets are uniformly spaced in this compressed domain, hence non-uniform
    in the original pixel domain.

    Usage:
        bucketer = MouseBucketer.from_percentiles(d_max=200)
        bucket_idx, residual = bucketer.encode(dx)
        dx_reconstructed = bucketer.decode(bucket_idx, residual)
    """

    def __init__(self, d_max: float = 200.0, n_buckets: int = 64, scale: float = 4.0):
        self.d_max = float(d_max)
        self.n_buckets = int(n_buckets)
        self.scale = float(scale)
        self.bounds = self._compute_bounds()

    @classmethod
    def from_percentiles(cls, d_max: float = 200.0, n_buckets: int = 64) -> "MouseBucketer":
        return cls(d_max=d_max, n_buckets=n_buckets)

    def _signed_log(self, x):
        return np.sign(x) * np.log1p(np.abs(x) / self.scale) / np.log1p(self.d_max / self.scale)

    def _signed_log_inv(self, u):
        return np.sign(u) * (np.expm1(np.abs(u) * np.log1p(self.d_max / self.scale)) * self.scale)

    def _compute_bounds(self) -> np.ndarray:
        t = np.linspace(-1.0, 1.0, self.n_buckets + 1)
        bounds = self._signed_log_inv(t)
        bounds[0] = -np.inf
        bounds[-1] = np.inf
        return bounds

    @property
    def bucket_centers(self) -> np.ndarray:
        return np.array([
            (self.bounds[i] + self.bounds[i + 1]) / 2.0
            if np.isfinite(self.bounds[i]) and np.isfinite(self.bounds[i + 1])
            else 0.0
            for i in range(self.n_buckets)
        ])

    def encode(self, delta: float) -> tuple[int, float]:
        """raw delta → (bucket_idx, residual). residual = delta - bucket_center."""
        delta = float(np.clip(delta, -self.d_max, self.d_max))
        bucket_idx = int(np.searchsorted(self.bounds[1:-1], delta))
        bucket_idx = max(0, min(bucket_idx, self.n_buckets - 1))
        center = self.bucket_centers[bucket_idx]
        residual = float(delta - center)
        return bucket_idx, residual

    def encode_batch(self, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        deltas = np.clip(deltas, -self.d_max, self.d_max)
        bucket_idx = np.searchsorted(self.bounds[1:-1], deltas).clip(0, self.n_buckets - 1)
        centers = self.bucket_centers[bucket_idx]
        residuals = deltas - centers
        return bucket_idx.astype(np.int64), residuals.astype(np.float32)

    def decode(self, bucket_idx: int, residual: float = 0.0) -> float:
        center = self.bucket_centers[bucket_idx]
        return float(center + residual)

    def decode_batch(self, bucket_idx: np.ndarray, residuals: np.ndarray = None) -> np.ndarray:
        centers = self.bucket_centers[bucket_idx]
        if residuals is not None:
            return centers + residuals
        return centers

    def __repr__(self):
        return (f"MouseBucketer(d_max={self.d_max}, n_buckets={self.n_buckets}, "
                f"scale={self.scale})")
