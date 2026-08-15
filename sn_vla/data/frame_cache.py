"""Offline frame cache builder.

Extracts every `stride`-th frame from each episode mkv, resizes, and stores
JPEGs in per-episode folders. Enables fast random access during training.

On-the-fly cv2 seeking is ~100x slower (each seek decodes from the nearest
keyframe). ffmpeg here does sequential decode + scale + encode in one pass.

Cache layout:  <cache_root>/<game>/<episode_stem>/f_%06d.jpg
Mapping:       file f_000001.jpg = original frame 0, f_000002.jpg = frame
`stride`, ... so cache index j (0-based) = original_frame // stride.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .manifest import find_episodes


def cache_dir_for(cache_root: Path, mkv_path: Path) -> Path:
    return cache_root / mkv_path.parent.name / mkv_path.stem


def build_episode_cache(
    mkv_path: Path,
    cache_root: Path,
    stride: int = 8,
    size: int = 224,
    quality: int = 3,
    timeout_s: int = 7200,
    hwaccel: str | None = None,
    segments: int = 1,
    gpu: int | None = None,
) -> dict:
    """Extract frames for one episode. Idempotent (skips if done marker exists).

    Args:
        hwaccel: None | "cuda". GPU decode via h264_cuvid/hevc_cuvid (5-10x faster).
        segments: split episode into N time-parallel ffmpeg jobs (CPU or GPU).
        gpu: CUDA device index for hwaccel (round-robin assigned by caller).
    """
    out_dir = cache_dir_for(cache_root, mkv_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "done.json"
    if marker.exists():
        return json.loads(marker.read_text())

    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "json", str(mkv_path)],
        capture_output=True, text=True, timeout=60,
    )
    try:
        dur = float(json.loads(probe.stdout)["format"]["duration"])
    except Exception:
        dur = 0.0

    procs = []
    errors = []
    if segments <= 1 or dur <= 0:
        procs.append(_spawn_ffmpeg(mkv_path, out_dir, 0, None, stride, size,
                                   quality, hwaccel, gpu, timeout_s, errors))
    else:
        seg = dur / segments
        for i in range(segments):
            procs.append(_spawn_ffmpeg(mkv_path, out_dir, i, seg, stride, size,
                                       quality, hwaccel, gpu, timeout_s, errors))

    # Wait all; on failure kill siblings
    failed = False
    for p, seg_idx, tmp_prefix in procs:
        try:
            rc = p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -1
        if rc != 0:
            failed = True
    if failed:
        for p, _, _ in procs:
            if p.poll() is None:
                p.kill()
        # cleanup partial tmp outputs
        for _, _, tmp_prefix in procs:
            if tmp_prefix:
                for f in out_dir.glob(f"{tmp_prefix}*.jpg"):
                    f.unlink()
        return {"status": "error", "error": "; ".join(errors[-2:]) or f"ffmpeg rc={rc}"}

    # Merge segment tmp outputs into final f_%06d.jpg sequence
    if len(procs) > 1:
        _merge_segments(out_dir, segments)

    n_frames = len(list(out_dir.glob("f_*.jpg")))
    meta = {
        "status": "ok",
        "stride": stride,
        "size": size,
        "n_frames": n_frames,
        "mkv_path": str(mkv_path),
    }
    marker.write_text(json.dumps(meta))
    return meta


def _spawn_ffmpeg(mkv_path, out_dir, seg_idx, seg_dur, stride, size, quality,
                  hwaccel, gpu, timeout_s, errors):
    """Spawn one ffmpeg process for a (whole episode | time segment)."""
    import shlex
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if hwaccel == "cuda":
        cmd += ["-hwaccel", "cuda"]
        if gpu is not None:
            cmd += ["-hwaccel_device", str(gpu)]
        # fall back to sw if a file can't be hw-decoded
        cmd += ["-hwaccel_output_format", "nv12"]
    if seg_dur is not None:
        cmd += ["-ss", f"{seg_idx * seg_dur:.3f}"]
        cmd += ["-t", f"{seg_dur:.3f}"]
    cmd += [
        "-i", str(mkv_path),
        "-vf", f"select=not(mod(n\\,{stride})),scale={size}:{size}",
        "-vsync", "0",
        "-q:v", str(quality),
    ]
    if seg_dur is not None:
        # tmp per-segment prefix; merged later
        out_pat = str(out_dir / f"tmpseg{seg_idx}_%06d.jpg")
    else:
        out_pat = str(out_dir / "f_%06d.jpg")
    cmd += [out_pat]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Drain stderr in background to avoid pipe deadlock
    import threading
    def _drain():
        err = p.stderr.read().decode(errors="replace")
        if err:
            errors.append(f"seg{seg_idx}: {err[-300:]}")
    threading.Thread(target=_drain, daemon=True).start()
    return p, seg_idx, (f"tmpseg{seg_idx}_" if seg_dur is not None else None)


def _merge_segments(out_dir: Path, segments: int):
    """Rename tmpseg{i}_XXXXXX.jpg files into a single ordered f_%06d.jpg sequence."""
    all_tmp = []
    for i in range(segments):
        files = sorted(out_dir.glob(f"tmpseg{i}_*.jpg"))
        all_tmp.extend(files)
    for n, f in enumerate(all_tmp, start=1):
        f.rename(out_dir / f"f_{n:06d}.jpg")


def _worker(args):
    mkv, cache_root, stride, size, quality, hwaccel, segments, gpu = args
    meta = build_episode_cache(mkv, Path(cache_root), stride, size, quality,
                               hwaccel=hwaccel, segments=segments, gpu=gpu)
    hw_tag = f" [gpu{gpu}]" if (hwaccel == "cuda" and gpu is not None) else ""
    return mkv.name, meta["status"], hw_tag


def build_frame_cache(
    d2e_dir: str | Path,
    cache_root: str | Path,
    stride: int = 8,
    size: int = 224,
    quality: int = 3,
    workers: int = 16,
    max_episodes: int = 0,
    hwaccel: str | None = None,
    segments: int = 1,
    n_gpus: int = 0,
):
    """Build frame cache for all episodes.

    Args:
        hwaccel: "cuda" for GPU decode (needs ffmpeg with cuvid), None for CPU.
        segments: time-parallel splits per episode (4 → ~4x speedup per episode).
        n_gpus: number of GPUs to round-robin when hwaccel=cuda. 0 → detect.
    """
    pairs = find_episodes(d2e_dir)
    if max_episodes > 0:
        pairs = pairs[:max_episodes]
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    if hwaccel == "cuda" and n_gpus <= 0:
        try:
            n_gpus = sum(1 for _ in Path("/dev/nvidia*").glob("nvidia[0-9]*")
                         if _.name[6:].isdigit())
        except Exception:
            n_gpus = 0
        if n_gpus <= 0:
            try:
                out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
                n_gpus = out.stdout.count("GPU ")
            except Exception:
                n_gpus = 1

    mode = f"GPU decode ({n_gpus} GPUs)" if hwaccel == "cuda" else "CPU decode"
    print(f"Building frame cache: {len(pairs)} episodes, stride={stride}, size={size}, "
          f"workers={workers}, segments/ep={segments}, {mode}")
    if hwaccel == "cuda":
        # Sanity-check: cuvid decoders present?
        chk = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                             capture_output=True, text=True)
        has_cuvid = "h264_cuvid" in chk.stdout
        if has_cuvid:
            print(f"[hwaccel] OK: cuvid decoders available, using GPU {0}-{n_gpus-1} "
                  f"round-robin (episode0→GPU0, episode1→GPU1, ...)")
            print("[hwaccel] Verify live with: nvidia-smi | grep ffmpeg")
        else:
            print("[hwaccel] WARNING: ffmpeg lacks cuvid support, falling back to CPU decode!")
            hwaccel = None
    else:
        print("[hwaccel] CPU decode (pass --hwaccel cuda to use GPU)")

    jobs = [
        (mkv, cache_root, stride, size, quality, hwaccel, segments,
         (i % n_gpus) if (hwaccel == "cuda" and n_gpus > 0) else None)
        for i, (_, mkv) in enumerate(pairs)
    ]
    done = error = 0
    from collections import Counter
    gpu_usage = Counter()
    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for name, status, hw_tag in tqdm(ex.map(_worker, jobs), total=len(jobs),
                                        desc="Extracting frames", unit="ep", smoothing=0.1):
            if status == "ok":
                done += 1
                if hw_tag:
                    gpu_usage[hw_tag] += 1
            elif status == "error":
                error += 1
                print(f"  ERROR: {name}")

    print(f"Done: {done} extracted, {error} errors. Cache at {cache_root}")
    if gpu_usage:
        print(f"GPU decode usage per episode: {dict(gpu_usage)}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("d2e_dir")
    ap.add_argument("cache_root")
    ap.add_argument("--stride", type=int, default=8, help="Keep every Nth frame (8 → 7.5fps)")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=3, help="JPEG quality (2=best, 31=worst)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--hwaccel", choices=["cuda", "cpu"], default="cpu")
    ap.add_argument("--segments", type=int, default=1,
                    help="Time-parallel splits per episode (e.g. 4)")
    ap.add_argument("--n-gpus", type=int, default=0, help="Override GPU count for cuda decode")
    args = ap.parse_args()
    build_frame_cache(
        args.d2e_dir, args.cache_root, args.stride, args.size, args.quality,
        args.workers, args.max_episodes,
        hwaccel=(args.hwaccel if args.hwaccel == "cuda" else None),
        segments=args.segments, n_gpus=args.n_gpus,
    )


if __name__ == "__main__":
    main()
