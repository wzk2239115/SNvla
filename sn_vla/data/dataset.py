"""Torch Dataset for SN-VLA training.

Loads pre-indexed episodes + pre-computed codec caches and yields
(model_inputs, action_targets) pairs for Fast Action Head training.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .mouse_buckets import MouseBucketer
from .reconstruct import N_VK, N_MOUSE_BUTTONS


@dataclass
class SampleTarget:
    """Numpy-level target for one tick (before tensor conversion)."""
    kbd_multi_hot: np.ndarray        # [256]
    press_events: np.ndarray         # [256]
    release_events: np.ndarray       # [256]
    mdx_bucket: int
    mdx_residual: float
    mdy_bucket: int
    mdy_residual: float
    btn_multi_hot: np.ndarray        # [5]
    wheel: int                       # 0/1/2


def encode_mouse_target(dx: int, dy: int, bucketer: MouseBucketer) -> tuple[int, float, int, float]:
    """Encode raw dx/dy into (bucket, residual) pairs."""
    mdx_b, mdx_r = bucketer.encode(dx)
    mdy_b, mdy_r = bucketer.encode(dy)
    return mdx_b, mdx_r, mdy_b, mdy_r


class D2EDataset(Dataset):
    """Dataset for Fast Action Head behavior cloning.

    Each item returns:
        visual:    [P, C, H, W] or [n_canvas, C, H, W] — codec canvas pixels
        target:    SampleTarget with all action dimensions
        prev_action: SampleTarget of the previous tick (for TemporalMemory input)

    The visual is loaded from pre-computed codec canvas cache or extracted on-the-fly.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        codec_cache_dir: str | Path | None = None,
        frame_cache_dir: str | Path | None = None,
        frame_stride: int = 8,
        n_visual_frames: int = 8,
        frame_size: int = 224,
        tick_hz: int = 60,
        window_frames: int = 64,
        clip_frames: int = 64,
        bucketer: MouseBucketer | None = None,
        clip_stride: int = 1,
        transform=None,
    ):
        from .manifest import load_manifest
        from .align import build_episode_index_cached

        self.manifest = load_manifest(manifest_path)
        self.codec_cache_dir = Path(codec_cache_dir) if codec_cache_dir else None
        self.frame_cache_dir = Path(frame_cache_dir) if frame_cache_dir else None
        self.frame_stride = frame_stride
        self.n_visual_frames = n_visual_frames
        self.frame_size = frame_size
        self.tick_hz = tick_hz
        self.window_frames = window_frames
        self.clip_frames = clip_frames
        self.bucketer = bucketer or MouseBucketer()
        self.transform = transform

        # Build flat sample index: (episode_idx, tick_idx, clip_idx)
        self._episode_indices: list = []
        self._samples: list[tuple[int, int, int]] = []  # (ep_array_idx, tick, clip)

        for arr_idx, ep in enumerate(self.manifest["episodes"]):
            try:
                idx = build_episode_index_cached(
                    mcap_path=ep["mcap_path"],
                    mkv_path=ep["mkv_path"],
                    tick_hz=tick_hz,
                    window_frames=window_frames,
                    episode_id=ep["episode_id"],
                )
            except Exception:
                continue
            self._episode_indices.append(idx)
            ticks_per_clip = clip_frames  # ticks map 1:1 to frames at tick_hz=fps
            for tick in range(idx.n_ticks):
                clip_idx = tick // ticks_per_clip
                self._samples.append((arr_idx, tick, clip_idx))

    def __len__(self) -> int:
        return len(self._samples)

    def _load_clip_visual(self, ep_arr_idx: int, episode_id: str, clip_idx: int, tick: int) -> np.ndarray:
        """Load visual input for a tick. Returns [n_frames, H, W, 3] uint8.

        Priority:
          1. Offline JPEG frame cache (fast random access)
          2. Pre-computed codec canvas cache (if codec_cache_dir set)
          3. On-the-fly frame extraction from mkv (slow fallback)
        """
        idx = self._episode_indices[ep_arr_idx]
        if tick >= idx.n_ticks:
            return np.zeros((1, self.frame_size, self.frame_size, 3), dtype=np.uint8)
        end_frame = int(idx.visual_end_frame_idx[tick])

        # 1. JPEG frame cache (preferred)
        if self.frame_cache_dir is not None:
            import cv2
            from pathlib import Path as _P
            cache_dir = self.frame_cache_dir / _P(idx.mkv_path).parent.name / _P(idx.mkv_path).stem
            e = end_frame // self.frame_stride
            start = max(0, e - self.n_visual_frames + 1)
            frames = []
            for j in range(start, e + 1):
                fp = cache_dir / f"f_{j + 1:06d}.jpg"
                if not fp.exists():
                    continue
                img = cv2.imread(str(fp))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if frames:
                return np.stack(frames)

        # 2. Codec canvas cache
        if self.codec_cache_dir is not None:
            clip_dir = self.codec_cache_dir / episode_id / f"clip_{clip_idx:05d}"
            meta_path = clip_dir / "meta.json"
            if meta_path.exists():
                import cv2
                meta = json.loads(meta_path.read_text())
                canvases = []
                for fname in meta.get("canvas_files", []):
                    path = clip_dir / fname
                    if path.suffix == ".npy":
                        canvases.append(np.load(path))
                    elif path.exists():
                        img = cv2.imread(str(path))
                        if img is not None:
                            canvases.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if canvases:
                    return np.stack(canvases)

        # 3. On-the-fly extraction (slow)
        return self._extract_frames_onthefly(ep_arr_idx, tick)

    def _extract_frames_onthefly(self, ep_arr_idx: int, tick: int, target_size: int = 224) -> np.ndarray:
        """Extract a window of frames from the mkv at the given tick.

        Samples n_sample_frames evenly within the visual window and resizes
        to target_size × target_size.
        """
        import cv2
        idx = self._episode_indices[ep_arr_idx]

        if tick >= idx.n_ticks:
            return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)
        end_frame = int(idx.visual_end_frame_idx[tick])
        if end_frame < 0:
            return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

        start_frame = max(0, end_frame - self.window_frames + 1)
        n_sample = min(8, end_frame - start_frame + 1)  # sample 8 frames
        if n_sample <= 0:
            return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)

        frame_indices = np.linspace(start_frame, end_frame, n_sample, dtype=int)

        cap = cv2.VideoCapture(idx.mkv_path)
        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if ok:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (target_size, target_size))
                frames.append(frame)
        cap.release()

        if not frames:
            return np.zeros((1, target_size, target_size, 3), dtype=np.uint8)
        return np.stack(frames)

    def _get_target(self, ep_idx: int, tick: int) -> SampleTarget:
        idx = self._episode_indices[ep_idx]
        dx = int(idx.mouse_dx[tick])
        dy = int(idx.mouse_dy[tick])
        mdx_b, mdx_r, mdy_b, mdy_r = encode_mouse_target(dx, dy, self.bucketer)
        return SampleTarget(
            kbd_multi_hot=idx.kbd_multi_hot[tick],
            press_events=idx.press_events[tick],
            release_events=idx.release_events[tick],
            mdx_bucket=mdx_b,
            mdx_residual=mdx_r,
            mdy_bucket=mdy_b,
            mdy_residual=mdy_r,
            btn_multi_hot=idx.btn_multi_hot[tick],
            wheel=int(idx.wheel[tick]),
        )

    def __getitem__(self, i: int) -> dict:
        try:
            return self._get_item_impl(i)
        except Exception:
            # Robustness: a corrupted sample must not kill the DataLoader worker.
            # Return a zero/no-op sample (kept rare; underlying issue is logged once).
            import traceback
            if not getattr(self, "_logged_getitem_error", False):
                self._logged_getitem_error = True
                print(f"[D2EDataset] getitem error (logging once):\n"
                      f"{traceback.format_exc()[-800:]}")
            return self._zero_sample()

    def _zero_sample(self) -> dict:
        z = lambda *shape: torch.zeros(*shape)
        mid = self.bucketer.n_buckets // 2
        return {
            "visual": z(1, 3, self.frame_size, self.frame_size),
            "target_kbd": z(N_VK),
            "target_press": z(N_VK),
            "target_release": z(N_VK),
            "target_mdx_bucket": torch.tensor(mid, dtype=torch.long),
            "target_mdx_residual": torch.tensor(0.0),
            "target_mdy_bucket": torch.tensor(mid, dtype=torch.long),
            "target_mdy_residual": torch.tensor(0.0),
            "target_btn": z(N_MOUSE_BUTTONS),
            "target_wheel": torch.tensor(0, dtype=torch.long),
            "prev_kbd": z(N_VK),
            "prev_mdx_bucket": torch.tensor(mid, dtype=torch.long),
            "prev_mdy_bucket": torch.tensor(mid, dtype=torch.long),
            "prev_btn": z(N_MOUSE_BUTTONS),
            "prev_wheel": torch.tensor(0, dtype=torch.long),
        }

    def _get_item_impl(self, i: int) -> dict:
        ep_arr_idx, tick, clip_idx = self._samples[i]
        idx = self._episode_indices[ep_arr_idx]
        episode_id = idx.episode_id

        visual = self._load_clip_visual(ep_arr_idx, episode_id, clip_idx, tick)
        if self.transform:
            visual = self.transform(visual)

        target = self._get_target(ep_arr_idx, tick)
        prev_target = self._get_target(ep_arr_idx, max(0, tick - 1)) if tick > 0 else None
        if prev_target is None:
            prev_target = SampleTarget(
                kbd_multi_hot=np.zeros(N_VK, dtype=np.float32),
                press_events=np.zeros(N_VK, dtype=np.float32),
                release_events=np.zeros(N_VK, dtype=np.float32),
                mdx_bucket=self.bucketer.n_buckets // 2,
                mdx_residual=0.0,
                mdy_bucket=self.bucketer.n_buckets // 2,
                mdy_residual=0.0,
                btn_multi_hot=np.zeros(N_MOUSE_BUTTONS, dtype=np.float32),
                wheel=0,
            )

        return {
            "visual": torch.from_numpy(visual).permute(0, 3, 1, 2).float() / 255.0,  # [n_canvas, 3, H, W]
            "target_kbd": torch.from_numpy(target.kbd_multi_hot),
            "target_press": torch.from_numpy(target.press_events),
            "target_release": torch.from_numpy(target.release_events),
            "target_mdx_bucket": torch.tensor(target.mdx_bucket, dtype=torch.long),
            "target_mdx_residual": torch.tensor(target.mdx_residual, dtype=torch.float32),
            "target_mdy_bucket": torch.tensor(target.mdy_bucket, dtype=torch.long),
            "target_mdy_residual": torch.tensor(target.mdy_residual, dtype=torch.float32),
            "target_btn": torch.from_numpy(target.btn_multi_hot),
            "target_wheel": torch.tensor(target.wheel, dtype=torch.long),
            "prev_kbd": torch.from_numpy(prev_target.kbd_multi_hot),
            "prev_mdx_bucket": torch.tensor(prev_target.mdx_bucket, dtype=torch.long),
            "prev_mdy_bucket": torch.tensor(prev_target.mdy_bucket, dtype=torch.long),
            "prev_btn": torch.from_numpy(prev_target.btn_multi_hot),
            "prev_wheel": torch.tensor(prev_target.wheel, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-length visual inputs with padding.

    Visual canvases may differ in count; we pad to the max and track the mask.
    """
    out = {}
    keys_scalar = [
        "target_kbd", "target_press", "target_release",
        "target_mdx_bucket", "target_mdx_residual",
        "target_mdy_bucket", "target_mdy_residual",
        "target_btn", "target_wheel",
        "prev_kbd", "prev_mdx_bucket", "prev_mdy_bucket", "prev_btn", "prev_wheel",
    ]
    for k in keys_scalar:
        out[k] = torch.stack([b[k] for b in batch])

    # Visual: pad to max n_canvas
    visuals = [b["visual"] for b in batch]
    max_canvas = max(v.shape[0] for v in visuals)
    _, C, H, W = visuals[0].shape
    padded = torch.zeros(len(batch), max_canvas, C, H, W, dtype=visuals[0].dtype)
    canvas_mask = torch.zeros(len(batch), max_canvas, dtype=torch.bool)
    for i, v in enumerate(visuals):
        padded[i, :v.shape[0]] = v
        canvas_mask[i, :v.shape[0]] = True
    out["visual"] = padded
    out["canvas_mask"] = canvas_mask
    return out
