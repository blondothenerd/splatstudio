import errno
import hashlib
import html
import json
import math
import mimetypes
import os
import platform
import queue
import re
import shlex
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlencode, urlsplit

import numpy as np
import streamlit as st

# --- PAGE SETUP ---
APP_NAME = "Splat Studio"
APP_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = Path(os.getenv("SPLAT_WORKSPACE", str(APP_DIR / "projects"))).resolve()
APP_ICON_PATH = APP_DIR / "SplatStudio_icon.png"
RUNTIME_DIR = APP_DIR / ".splat_studio" / "runtime"
_DEFAULT_TRAINER_SCRIPT = RUNTIME_DIR / "SplatStudioFastMetalTrainer.py"
TRAINER_SCRIPT = Path(os.getenv("SPLAT_STUDIO_TRAINER", str(_DEFAULT_TRAINER_SCRIPT))).resolve()

# V19.0 high-performance native MLX/C++/Metal backend.
GSPLAT_METAL_ROOT = Path(os.getenv("SPLAT_METAL_ROOT", str(APP_DIR / "backend" / "gsplat-metal"))).expanduser().resolve()
GSPLAT_METAL_PYTHON = Path(os.getenv("SPLAT_METAL_PYTHON", str(APP_DIR / "runtime" / "gsplat_core" / "bin" / "python"))).expanduser().resolve()
GSPLAT_METAL_TRAINER = TRAINER_SCRIPT
GSPLAT_METAL_TEST_DIR = GSPLAT_METAL_ROOT / "scripts" / "test"
NATIVE_TRAINING_DIRNAME = ".splat_studio_native_metal"
AI_VISION_SCRIPT = RUNTIME_DIR / "SplatStudioAIVision.py"
AI_MODELS_ROOT = APP_DIR / "models" / "huggingface"
AI_OUTPUT_DIRNAME = ".splat_studio_ai"
AI_DEPTH_MODEL = os.getenv("SPLAT_AI_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Small-hf")
AI_SEGMENT_MODEL = os.getenv("SPLAT_AI_SEGMENT_MODEL", "nvidia/segformer-b5-finetuned-ade-640-640")


# Splat Studio owns a thin customization of the installed a1091150 native
# C++/Metal trainer. The upstream backend remains untouched; this embedded copy
# adds durable snapshots, resume, live previews, semantic masks and AI depth point injection.
_FAST_METAL_TRAINER_SOURCE = '#!/usr/bin/env python3\n\n# Splat Studio customized trainer.\n# Portions are derived from a1091150/gsplat-mlx (MIT License).\n# Copyright (c) 2026 DokiDokiPB.\n# See THIRD_PARTY_NOTICES.md and licenses/gsplat-mlx-MIT.txt.\n\nfrom __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport os\nimport shutil\nimport signal\nimport sys\nimport time\nfrom pathlib import Path\n\nimport mlx.core as mx\nimport mlx.nn as nn\nimport numpy as np\nfrom mlx.optimizers import Adam\nfrom mlx.utils import tree_flatten, tree_unflatten\nfrom PIL import Image\n\n_BACKEND_TEST_DIR = os.getenv("SPLAT_METAL_TEST_DIR")\nif _BACKEND_TEST_DIR:\n    backend_test_dir = str(Path(_BACKEND_TEST_DIR).expanduser().resolve())\n    if backend_test_dir not in sys.path:\n        sys.path.insert(0, backend_test_dir)\n\nfrom colmap_360_dataset import load_colmap_scene, prepare_colmap_points, select_colmap_cameras\nfrom render_random_3dgs_png import write_png\nfrom scanner_dataset_utils import load_target\nfrom scanner_points_training_utils import (\n    SH_C0,\n    FrameBatchSampler,\n    ScannerDefaultStrategyConfig,\n    ScannerDefaultStrategyRuntime,\n    ScannerPointsSHModel,\n    append_random_gaussians,\n    camera_batch_arrays,\n    concat_compare,\n    export_trained_spz,\n    image_to_u8,\n    lr_for_step,\n    make_lr_schedule,\n    normalize_quats,\n    opacity_diagnostics,\n    points_extent_diagnostics,\n    positions_to_spz,\n    render_sh_model,\n    save_model_parameters_npz,\n    sh_coeff_count,\n    spz_export_diagnostics,\n    ssim_index,\n    target_batch_array,\n)\n\n\nMAX_SUPPORTED_SH_DEGREE = 3\n\n\ndef log(message: str) -> None:\n    print(message, flush=True)\n\n\ndef gsplat_active_sh_degree(step: int, target: int, interval: int) -> int:\n    return int(min(step // interval, target))\n\n\ndef infer_image_size(data_dir: Path, factor: int) -> tuple[int, int]:\n    image_dir = data_dir / ("images" if factor <= 1 else f"images_{factor}")\n    for path in sorted(image_dir.rglob("*")):\n        if path.is_file() and not path.name.startswith("."):\n            with Image.open(path) as image:\n                return image.size\n    raise RuntimeError(f"No images found in {image_dir}")\n\n\ndef init_sh_model_from_points(\n    points: np.ndarray,\n    colors: np.ndarray,\n    log_scales: np.ndarray,\n    opacity: float,\n    max_sh_degree: int,\n) -> ScannerPointsSHModel:\n    n = int(points.shape[0])\n    means = mx.array(points[None, ...], dtype=mx.float32)\n    quats = mx.zeros((1, n, 4), dtype=mx.float32) + mx.array([1.0, 0.0, 0.0, 0.0], dtype=mx.float32)\n    features_dc = mx.array(((colors[None, ...] - 0.5) / SH_C0).astype(np.float32), dtype=mx.float32)\n    rest_count = sh_coeff_count(max_sh_degree) - 1\n    features_rest = mx.zeros((1, n, rest_count, 3), dtype=mx.float32)\n    opacity_logits = mx.log(mx.full((1, n), opacity, dtype=mx.float32) / (1.0 - opacity))\n    return ScannerPointsSHModel.from_arrays(\n        means,\n        normalize_quats(quats),\n        mx.array(log_scales[None, ...], dtype=mx.float32),\n        features_dc,\n        features_rest,\n        opacity_logits,\n    )\n\n\ndef knn_log_scales_from_points(points: np.ndarray, init_scale: float = 1.0) -> np.ndarray:\n    points = np.asarray(points, dtype=np.float32)\n    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:\n        raise ValueError("points must have shape [N, 3] and be nonempty")\n    if points.shape[0] < 4:\n        distances = np.linalg.norm(points - points.mean(axis=0, keepdims=True), axis=1)\n        fallback = float(max(np.mean(distances), 1.0e-6) * init_scale)\n        return np.full((points.shape[0], 3), np.log(fallback), dtype=np.float32)\n    try:\n        from scipy.spatial import cKDTree\n    except ImportError:\n        if points.shape[0] > 20_000:\n            raise ImportError(\n                "KNN scale initialization for full 360 data requires scipy.spatial.cKDTree; "\n                "install scipy or run a reduced --max-points smoke."\n            )\n        distances = []\n        for start in range(0, points.shape[0], 1024):\n            chunk = points[start : start + 1024]\n            dist2 = np.sum((chunk[:, None, :] - points[None, :, :]) ** 2, axis=-1)\n            nearest2 = np.partition(dist2, kth=3, axis=1)[:, :4]\n            distances.append(np.sqrt(nearest2))\n        distances = np.concatenate(distances, axis=0)\n    else:\n        distances, _ = cKDTree(points).query(points, k=4)\n    dist_avg = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))\n    scales = np.maximum(dist_avg * float(init_scale), 1.0e-8)\n    return np.repeat(np.log(scales)[:, None], 3, axis=1).astype(np.float32)\n\n\ndef loss_components(\n    image: mx.array,\n    target: mx.array,\n    ssim_lambda: float,\n    ssim_window_size: int,\n    mask: mx.array | None = None,\n) -> dict[str, mx.array]:\n    if mask is None:\n        l1 = nn.losses.l1_loss(image, target)\n        ssim_image = image\n        ssim_target = target\n    else:\n        mask3 = mx.broadcast_to(mask, image.shape)\n        denominator = mx.maximum(mx.sum(mask3), mx.array(1.0, dtype=mx.float32))\n        l1 = mx.sum(mx.abs(image - target) * mask3) / denominator\n        ssim_image = image * mask3\n        ssim_target = target * mask3\n    ssim = ssim_index(ssim_image, ssim_target, ssim_window_size)\n    ssim_loss = 1.0 - ssim\n    loss = (1.0 - ssim_lambda) * l1 + ssim_lambda * ssim_loss\n    return {"loss": loss, "l1": l1, "ssim": ssim, "ssim_loss": ssim_loss}\n\n\n\ndef load_training_mask(mask_dir: Path | None, camera, width: int, height: int) -> np.ndarray:\n    if mask_dir is None:\n        return np.ones((height, width, 1), dtype=np.float32)\n    relative = Path(camera.image_name).with_suffix(".png")\n    mask_path = mask_dir / relative\n    if not mask_path.exists():\n        return np.ones((height, width, 1), dtype=np.float32)\n    with Image.open(mask_path) as image:\n        mask = image.convert("L").resize((width, height), Image.Resampling.NEAREST)\n        array = np.asarray(mask, dtype=np.float32) / 255.0\n    return array[..., None]\n\n\ndef load_extra_points(path: Path | None) -> tuple[np.ndarray, np.ndarray]:\n    if path is None or not path.exists():\n        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)\n    with np.load(path, allow_pickle=False) as data:\n        points = np.asarray(data["points"], dtype=np.float32)\n        colors = np.asarray(data["colors"], dtype=np.float32)\n    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:\n        raise ValueError(f"AI extra point file must contain points/colors [N,3]: {path}")\n    finite = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(colors), axis=1)\n    points = points[finite]\n    colors = np.clip(colors[finite], 0.0, 1.0)\n    return points, colors\n\n\ndef ai_path_signature(path: Path | None) -> dict | None:\n    if path is None:\n        return None\n    path = path.expanduser().resolve()\n    if not path.exists():\n        return {"path": str(path), "missing": True}\n    if path.is_dir():\n        files = sorted(p for p in path.rglob("*.png") if p.is_file())\n        size = sum(p.stat().st_size for p in files[:5000])\n        newest = max((p.stat().st_mtime_ns for p in files), default=0)\n        return {"path": str(path), "files": len(files), "size": int(size), "mtime": int(newest)}\n    stat = path.stat()\n    return {"path": str(path), "size": int(stat.st_size), "mtime": int(stat.st_mtime_ns)}\n\ndef render_loss_stats(\n    model: ScannerPointsSHModel,\n    camera,\n    target: mx.array,\n    width: int,\n    height: int,\n    tile_size: int,\n    sh_degree: int,\n    ssim_lambda: float,\n    ssim_window_size: int,\n) -> dict:\n    viewmats, ks = camera_batch_arrays([camera], [0])\n    viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)\n    render = render_sh_model(model, viewspace_points, viewmats, ks, width, height, tile_size, sh_degree)\n    components = loss_components(render["render_colors"], target, ssim_lambda, ssim_window_size)\n    diff = render["render_colors"] - target\n    mse = mx.mean(diff * diff)\n    mx.eval(components["loss"], components["l1"], components["ssim"], components["ssim_loss"], mse, render["render_colors"], render["radii"], render["flatten_ids"])\n    mse_value = float(np.asarray(mse))\n    radii = np.asarray(render["radii"])\n    flatten_ids = np.asarray(render["flatten_ids"])\n    return {\n        "frame_index": int(camera.index),\n        "loss": float(np.asarray(components["loss"])),\n        "loss_components": {\n            "l1": float(np.asarray(components["l1"])),\n            "ssim": float(np.asarray(components["ssim"])),\n            "ssim_loss": float(np.asarray(components["ssim_loss"])),\n        },\n        "psnr": float(-10.0 * np.log10(max(mse_value, 1.0e-12))),\n        "visible_gaussians": int(np.count_nonzero(np.any(radii > 0, axis=-1))),\n        "intersections": int(flatten_ids.shape[0]),\n        "image": np.asarray(render["render_colors"][0], dtype=np.float32),\n    }\n\n\ndef evaluate_frames(\n    model: ScannerPointsSHModel,\n    cameras,\n    targets: list[mx.array],\n    width: int,\n    height: int,\n    tile_size: int,\n    sh_degree: int,\n    ssim_lambda: float,\n    ssim_window_size: int,\n) -> list[dict]:\n    return [\n        render_loss_stats(model, camera, target, width, height, tile_size, sh_degree, ssim_lambda, ssim_window_size)\n        for camera, target in zip(cameras, targets, strict=True)\n    ]\n\n\ndef render_step_grid(\n    model: ScannerPointsSHModel,\n    cameras,\n    width: int,\n    height: int,\n    tile_size: int,\n    sh_degree: int,\n) -> np.ndarray:\n    tiles = []\n    for index in range(16):\n        camera = cameras[index % len(cameras)]\n        viewmats, ks = camera_batch_arrays([camera], [0])\n        viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)\n        render = render_sh_model(model, viewspace_points, viewmats, ks, width, height, tile_size, sh_degree)\n        mx.eval(render["render_colors"])\n        tiles.append(image_to_u8(np.asarray(render["render_colors"][0], dtype=np.float32)))\n    rows = [np.concatenate(tiles[start : start + 4], axis=1) for start in range(0, 16, 4)]\n    return np.concatenate(rows, axis=0)\n\n\n\n_STOP_REQUESTED = False\n_CHECKPOINT_VERSION = 1\n_MODEL_FIELDS = (\n    "means",\n    "quats",\n    "log_scales",\n    "features_dc",\n    "features_rest",\n    "opacity_logits",\n)\n\n\ndef _json_safe(value):\n    if isinstance(value, dict):\n        return {str(key): _json_safe(item) for key, item in value.items()}\n    if isinstance(value, (list, tuple)):\n        return [_json_safe(item) for item in value]\n    if isinstance(value, np.ndarray):\n        return value.tolist()\n    if isinstance(value, np.generic):\n        return value.item()\n    if isinstance(value, Path):\n        return str(value)\n    return value\n\n\ndef checkpoint_signature(args: argparse.Namespace, cameras) -> str:\n    # Deliberately excludes render resolution/cache/refinement settings so the\n    # Splat Studio parent can lower GPU pressure after a crash and resume the\n    # same trained model. Dataset/camera selection and SH capacity must match.\n    payload = {\n        "data": str(args.data.expanduser().resolve()),\n        "data_factor": int(args.data_factor),\n        "test_every": int(args.test_every),\n        "normalize_world_space": bool(args.normalize_world_space),\n        "max_frames": int(args.max_frames),\n        "frame_step": int(args.frame_step),\n        "start_index": int(args.start_index),\n        "sh_degree": int(args.sh_degree),\n        "camera_frames": [\n            {\n                "index": int(camera.index),\n                "image": str(Path(camera.image_path).name),\n            }\n            for camera in cameras\n        ],\n    }\n    if args.mask_dir is not None:\n        payload["ai_mask"] = ai_path_signature(args.mask_dir)\n    if args.extra_points_npz is not None:\n        payload["ai_extra_points"] = ai_path_signature(args.extra_points_npz)\n    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")\n    return hashlib.sha256(raw).hexdigest()\n\n\ndef _model_payload(model: ScannerPointsSHModel) -> dict[str, np.ndarray]:\n    mx.eval(\n        model.means,\n        model.quats,\n        model.log_scales,\n        model.features_dc,\n        model.features_rest,\n        model.opacity_logits,\n    )\n    return {\n        "means": np.asarray(model.means, dtype=np.float32),\n        "quats": np.asarray(model.quats, dtype=np.float32),\n        "log_scales": np.asarray(model.log_scales, dtype=np.float32),\n        "features_dc": np.asarray(model.features_dc, dtype=np.float32),\n        "features_rest": np.asarray(model.features_rest, dtype=np.float32),\n        "opacity_logits": np.asarray(model.opacity_logits, dtype=np.float32),\n    }\n\n\ndef _model_from_checkpoint(path: Path) -> ScannerPointsSHModel:\n    with np.load(path, allow_pickle=False) as data:\n        arrays = {name: mx.array(np.asarray(data[name]), dtype=mx.float32) for name in _MODEL_FIELDS}\n    model = ScannerPointsSHModel.from_arrays(\n        arrays["means"],\n        normalize_quats(arrays["quats"]),\n        arrays["log_scales"],\n        arrays["features_dc"],\n        arrays["features_rest"],\n        arrays["opacity_logits"],\n    )\n    mx.eval(\n        model.means,\n        model.quats,\n        model.log_scales,\n        model.features_dc,\n        model.features_rest,\n        model.opacity_logits,\n    )\n    return model\n\n\ndef _save_optimizer_state(path: Path, optimizer: Adam) -> None:\n    # Follow MLX\'s documented optimizer-state serialization pattern exactly.\n    mx.eval(optimizer.state)\n    flattened = tree_flatten(optimizer.state, destination={})\n    mx.save_safetensors(str(path), flattened)\n\n\ndef _load_optimizer_state(path: Path, optimizer: Adam) -> None:\n    optimizer.state = tree_unflatten(mx.load(str(path)))\n    mx.eval(optimizer.state)\n\n\ndef _sampler_state(sampler: FrameBatchSampler) -> dict:\n    return {\n        "frame_count": int(sampler.frame_count),\n        "batch_size": int(sampler.batch_size),\n        "mode": str(sampler.mode),\n        "position": int(sampler.position),\n        "epoch": int(sampler.epoch),\n        "history": _json_safe(sampler.history),\n        "rng_state": _json_safe(sampler.rng.bit_generator.state),\n    }\n\n\ndef _restore_sampler_state(sampler: FrameBatchSampler, meta: dict, arrays_path: Path) -> None:\n    state = meta.get("sampler") or {}\n    if (\n        int(state.get("frame_count", -1)) != sampler.frame_count\n        or int(state.get("batch_size", -1)) != sampler.batch_size\n        or str(state.get("mode", "")) != sampler.mode\n    ):\n        log("CHECKPOINT sampler state ignored because frame sampling settings changed")\n        return\n    with np.load(arrays_path, allow_pickle=False) as data:\n        sampler.order = np.asarray(data["sampler_order"], dtype=np.int32)\n        sampler.usage_counts = np.asarray(data["sampler_usage_counts"], dtype=np.int64)\n    sampler.position = int(state.get("position", 0))\n    sampler.epoch = int(state.get("epoch", 0))\n    sampler.history = [[int(v) for v in batch] for batch in state.get("history", [])]\n    if state.get("rng_state"):\n        sampler.rng.bit_generator.state = state["rng_state"]\n\n\ndef _strategy_state(strategy: ScannerDefaultStrategyRuntime) -> dict:\n    return {\n        "initial_gaussians": int(strategy.initial_gaussians),\n        "last_gaussians": int(strategy.last_gaussians),\n        "events": _json_safe(strategy.events),\n        "totals": _json_safe(strategy.totals),\n        "last_grad2d_stats": _json_safe(strategy.last_grad2d_stats),\n        "last_grad2d_mode": str(strategy.last_grad2d_mode),\n        "absgrad_fallback_count": int(strategy.absgrad_fallback_count),\n        "rng_state": _json_safe(strategy.rng.bit_generator.state),\n        "has_radii": strategy.radii is not None,\n    }\n\n\ndef _restore_strategy_state(strategy: ScannerDefaultStrategyRuntime, meta: dict, arrays_path: Path) -> None:\n    state = meta.get("strategy") or {}\n    with np.load(arrays_path, allow_pickle=False) as data:\n        strategy.grad2d = np.asarray(data["strategy_grad2d"], dtype=np.float32)\n        strategy.count = np.asarray(data["strategy_count"], dtype=np.float32)\n        has_radii = bool(state.get("has_radii", False))\n        strategy.radii = np.asarray(data["strategy_radii"], dtype=np.float32) if has_radii and "strategy_radii" in data else None\n    strategy.initial_gaussians = int(state.get("initial_gaussians", strategy.grad2d.shape[0]))\n    strategy.last_gaussians = int(state.get("last_gaussians", strategy.grad2d.shape[0]))\n    strategy.events = list(state.get("events") or [])\n    strategy.totals.update(state.get("totals") or {})\n    strategy.last_grad2d_stats = dict(state.get("last_grad2d_stats") or strategy._grad2d_stats())\n    strategy.last_grad2d_mode = str(state.get("last_grad2d_mode") or "signed_grad_norm")\n    strategy.absgrad_fallback_count = int(state.get("absgrad_fallback_count", 0))\n    if state.get("rng_state"):\n        strategy.rng.bit_generator.state = state["rng_state"]\n\n\ndef _latest_checkpoint_dir(checkpoint_root: Path) -> Path | None:\n    pointer = checkpoint_root / "latest.json"\n    if pointer.exists():\n        try:\n            data = json.loads(pointer.read_text(encoding="utf-8"))\n            candidate = checkpoint_root / str(data.get("directory") or "")\n            if candidate.is_dir() and (candidate / "state.json").exists():\n                return candidate\n        except (OSError, json.JSONDecodeError):\n            pass\n    candidates = sorted(\n        (path for path in checkpoint_root.glob("step_*") if path.is_dir() and (path / "state.json").exists()),\n        key=lambda path: path.name,\n    )\n    return candidates[-1] if candidates else None\n\n\ndef load_checkpoint_metadata(checkpoint_root: Path, expected_signature: str) -> tuple[Path, dict] | None:\n    checkpoint_dir = _latest_checkpoint_dir(checkpoint_root)\n    if checkpoint_dir is None:\n        return None\n    try:\n        meta = json.loads((checkpoint_dir / "state.json").read_text(encoding="utf-8"))\n    except (OSError, json.JSONDecodeError) as exc:\n        log(f"CHECKPOINT ignored unreadable state: {exc}")\n        return None\n    if int(meta.get("version", 0)) != _CHECKPOINT_VERSION:\n        log("CHECKPOINT ignored unsupported snapshot version")\n        return None\n    if str(meta.get("signature") or "") != expected_signature:\n        log("CHECKPOINT ignored because dataset/camera signature changed")\n        return None\n    return checkpoint_dir, meta\n\n\ndef save_training_checkpoint(\n    checkpoint_root: Path,\n    keep: int,\n    signature: str,\n    step: int,\n    total_steps: int,\n    model: ScannerPointsSHModel,\n    optimizers: dict[str, Adam],\n    sampler: FrameBatchSampler,\n    strategy: ScannerDefaultStrategyRuntime,\n    active_sh_degree: int,\n    sh_degree_events: list[dict],\n    last_loss: float | None,\n    elapsed_seconds: float,\n    reason: str,\n) -> Path:\n    checkpoint_root.mkdir(parents=True, exist_ok=True)\n    name = f"step_{int(step):06d}"\n    temp_dir = checkpoint_root / f".{name}.{os.getpid()}.tmp"\n    final_dir = checkpoint_root / name\n    if temp_dir.exists():\n        shutil.rmtree(temp_dir)\n    temp_dir.mkdir(parents=True, exist_ok=True)\n\n    np.savez(temp_dir / "model.npz", **_model_payload(model))\n    for optimizer_name, optimizer in optimizers.items():\n        _save_optimizer_state(temp_dir / f"optimizer_{optimizer_name}.safetensors", optimizer)\n\n    arrays = {\n        "sampler_order": np.asarray(sampler.order, dtype=np.int32),\n        "sampler_usage_counts": np.asarray(sampler.usage_counts, dtype=np.int64),\n        "strategy_grad2d": np.asarray(strategy.grad2d, dtype=np.float32),\n        "strategy_count": np.asarray(strategy.count, dtype=np.float32),\n    }\n    if strategy.radii is not None:\n        arrays["strategy_radii"] = np.asarray(strategy.radii, dtype=np.float32)\n    np.savez(temp_dir / "runtime_arrays.npz", **arrays)\n\n    meta = {\n        "version": _CHECKPOINT_VERSION,\n        "signature": signature,\n        "step": int(step),\n        "total_steps": int(total_steps),\n        "active_sh_degree": int(active_sh_degree),\n        "sh_degree_events": _json_safe(sh_degree_events),\n        "loss": None if last_loss is None else float(last_loss),\n        "gaussians": int(model.means.shape[1]),\n        "elapsed_seconds": float(max(0.0, elapsed_seconds)),\n        "created_at": float(time.time()),\n        "reason": str(reason),\n        "sampler": _sampler_state(sampler),\n        "strategy": _strategy_state(strategy),\n    }\n    (temp_dir / "state.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")\n\n    if final_dir.exists():\n        shutil.rmtree(final_dir)\n    os.replace(temp_dir, final_dir)\n\n    pointer_temp = checkpoint_root / ".latest.tmp"\n    pointer_temp.write_text(\n        json.dumps({"directory": final_dir.name, "step": int(step), "updated_at": time.time()}, indent=2),\n        encoding="utf-8",\n    )\n    os.replace(pointer_temp, checkpoint_root / "latest.json")\n\n    snapshots = sorted(\n        (path for path in checkpoint_root.glob("step_*") if path.is_dir()),\n        key=lambda path: path.name,\n    )\n    for old in snapshots[:-max(1, int(keep))]:\n        if old != final_dir:\n            shutil.rmtree(old, ignore_errors=True)\n\n    log(\n        f"CHECKPOINT step={int(step)} total={int(total_steps)} gaussians={int(model.means.shape[1])} "\n        f"elapsed={float(elapsed_seconds):.1f}s reason={reason} path={final_dir}"\n    )\n    return final_dir\n\n\ndef restore_training_checkpoint(\n    checkpoint_dir: Path,\n    meta: dict,\n    model: ScannerPointsSHModel,\n    optimizers: dict[str, Adam],\n    sampler: FrameBatchSampler,\n    strategy: ScannerDefaultStrategyRuntime,\n) -> ScannerPointsSHModel:\n    model = _model_from_checkpoint(checkpoint_dir / "model.npz")\n    arrays_path = checkpoint_dir / "runtime_arrays.npz"\n    _restore_sampler_state(sampler, meta, arrays_path)\n    _restore_strategy_state(strategy, meta, arrays_path)\n    for optimizer_name, optimizer in optimizers.items():\n        optimizer_path = checkpoint_dir / f"optimizer_{optimizer_name}.safetensors"\n        if not optimizer_path.exists():\n            raise RuntimeError(f"Checkpoint is missing optimizer state: {optimizer_path.name}")\n        _load_optimizer_state(optimizer_path, optimizer)\n    return model\n\n\ndef _scaled_preview_camera_arrays(camera, scale: float) -> tuple[mx.array, mx.array]:\n    viewmats, ks = camera_batch_arrays([camera], [0])\n    if abs(scale - 1.0) < 1.0e-9:\n        return viewmats, ks\n    k_np = np.asarray(ks, dtype=np.float32).copy()\n    k_np[..., 0, 0] *= scale\n    k_np[..., 0, 2] *= scale\n    k_np[..., 1, 1] *= scale\n    k_np[..., 1, 2] *= scale\n    return viewmats, mx.array(k_np, dtype=mx.float32)\n\n\ndef render_live_preview(\n    model: ScannerPointsSHModel,\n    camera,\n    train_width: int,\n    train_height: int,\n    tile_size: int,\n    sh_degree: int,\n    max_side: int,\n) -> np.ndarray:\n    scale = min(1.0, float(max_side) / float(max(train_width, train_height)))\n    width = max(32, int(round(train_width * scale)))\n    height = max(32, int(round(train_height * scale)))\n    viewmats, ks = _scaled_preview_camera_arrays(camera, scale)\n    viewspace_points = mx.zeros((1, 1, model.means.shape[1], 2), dtype=mx.float32)\n    render = render_sh_model(model, viewspace_points, viewmats, ks, width, height, tile_size, sh_degree)\n    mx.eval(render["render_colors"])\n    return image_to_u8(np.asarray(render["render_colors"][0], dtype=np.float32))\n\n\ndef write_live_preview(\n    preview_dir: Path,\n    keep: int,\n    step: int,\n    image: np.ndarray,\n) -> Path:\n    preview_dir.mkdir(parents=True, exist_ok=True)\n    step_path = preview_dir / f"preview_{int(step):06d}.jpg"\n    temp_path = preview_dir / f".preview_{int(step):06d}.{os.getpid()}.tmp.jpg"\n    Image.fromarray(image).save(temp_path, format="JPEG", quality=84, optimize=True)\n    os.replace(temp_path, step_path)\n\n    latest_temp = preview_dir / ".preview_latest.tmp.jpg"\n    shutil.copyfile(step_path, latest_temp)\n    os.replace(latest_temp, preview_dir / "preview_latest.jpg")\n\n    previews = sorted(preview_dir.glob("preview_[0-9]*.jpg"), key=lambda path: path.name)\n    for old in previews[:-max(1, int(keep))]:\n        old.unlink(missing_ok=True)\n    return step_path\n\n\ndef _signal_stop(signum, _frame):\n    global _STOP_REQUESTED\n    _STOP_REQUESTED = True\n    log(f"STOP_REQUESTED signal={signum} checkpoint_at_next_safe_boundary=1")\n\ndef mean_loss(stats: list[dict]) -> float:\n    return float(np.mean([item["loss"] for item in stats])) if stats else 0.0\n\n\ndef validate_positive(name: str, value: float) -> None:\n    if value <= 0.0:\n        raise ValueError(f"{name} must be positive")\n\n\n\ndef camera_view_metadata(camera, scene_center: np.ndarray, psnr: float, visible_gaussians: int, max_visible: int, image_height: int) -> dict:\n    """Convert a trained COLMAP camera into the coordinate convention used by SPZ/SuperSplat."""\n    c2w = np.linalg.inv(np.asarray(camera.viewmat, dtype=np.float32))\n    position = c2w[:3, 3]\n    # Target the robust Gaussian scene centre rather than a near clip point.\n    target = np.asarray(scene_center, dtype=np.float32)\n    position_spz = positions_to_spz(position.reshape(1, 3))[0]\n    target_spz = positions_to_spz(target.reshape(1, 3))[0]\n    fy = max(float(camera.K[1, 1]), 1.0e-6)\n    fov = float(np.degrees(2.0 * np.arctan(float(image_height) / (2.0 * fy))))\n    visible_ratio = float(visible_gaussians) / max(1.0, float(max_visible))\n    # PSNR favours views where view-dependent SH/reflections reproduce the capture well;\n    # visibility prevents choosing a tiny close crop that happens to score highly.\n    score = float(psnr) + 1.5 * visible_ratio\n    return {\n        "frame_index": int(camera.index),\n        "image_name": str(camera.image_name),\n        "position": position_spz.astype(float).tolist(),\n        "target": target_spz.astype(float).tolist(),\n        "fov": float(np.clip(fov, 25.0, 100.0)),\n        "psnr": float(psnr),\n        "visible_gaussians": int(visible_gaussians),\n        "visibility_ratio": visible_ratio,\n        "score": score,\n    }\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--data", type=Path, default=Path("submodules/gsplat/examples/datasets/data/360_v2/garden"))\n    parser.add_argument("--out-dir", type=Path, default=Path("outputs/360_garden_train"))\n    parser.add_argument("--out-spz", type=Path, default=None)\n    parser.add_argument("--out-model-npz", type=Path, default=None)\n    parser.add_argument("--data-factor", type=int, default=4)\n    parser.add_argument("--test-every", type=int, default=8)\n    parser.add_argument("--normalize-world-space", action=argparse.BooleanOptionalAction, default=True)\n    parser.add_argument("--width", type=int, default=0)\n    parser.add_argument("--height", type=int, default=0)\n    parser.add_argument("--tile-size", type=int, default=16)\n    parser.add_argument("--max-frames", type=int, default=0)\n    parser.add_argument("--frame-step", type=int, default=1)\n    parser.add_argument("--start-index", type=int, default=0)\n    parser.add_argument("--eval-max-frames", type=int, default=0)\n    parser.add_argument("--eval-frame-step", type=int, default=None)\n    parser.add_argument("--eval-start-index", type=int, default=0)\n    parser.add_argument("--max-points", type=int, default=0)\n    parser.add_argument("--num-random-gaussians", type=int, default=0)\n    parser.add_argument("--random-gaussian-bounds-scale", type=float, default=1.05)\n    parser.add_argument("--init-scale", type=float, default=1.0)\n    parser.add_argument("--opacity", type=float, default=0.1)\n    parser.add_argument("--seed", type=int, default=42)\n    parser.add_argument("--steps", type=int, default=30000)\n    parser.add_argument("--batch-size", type=int, default=1)\n    parser.add_argument("--frame-sampling", choices=("sequential", "shuffle"), default="shuffle")\n    parser.add_argument("--frame-shuffle-seed", type=int, default=None)\n    parser.add_argument("--ssim-lambda", type=float, default=0.2)\n    parser.add_argument("--ssim-window-size", type=int, default=11)\n    parser.add_argument("--means-lr", type=float, default=1.6e-4)\n    parser.add_argument("--scales-lr", type=float, default=5.0e-3)\n    parser.add_argument("--opacities-lr", type=float, default=5.0e-2)\n    parser.add_argument("--quats-lr", type=float, default=1.0e-3)\n    parser.add_argument("--sh0-lr", type=float, default=2.5e-3)\n    parser.add_argument("--shn-lr", type=float, default=2.5e-3 / 20.0)\n    parser.add_argument("--sh-degree", type=int, default=3)\n    parser.add_argument("--sh-degree-interval", type=int, default=1000)\n    parser.add_argument("--global-scale", type=float, default=1.0)\n    parser.add_argument("--log-interval", type=int, default=100)\n    parser.add_argument("--step-image-interval", type=int, default=0)\n    parser.add_argument("--mask-dir", type=Path, default=None, help="Optional keep-mask directory; 255 trains, 0 ignores")\n    parser.add_argument("--extra-points-npz", type=Path, default=None, help="Optional AI depth points/colors in normalized COLMAP world space")\n    parser.add_argument("--checkpoint-dir", type=Path, default=None)\n    parser.add_argument("--checkpoint-every", type=int, default=250)\n    parser.add_argument("--checkpoint-keep", type=int, default=2)\n    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)\n    parser.add_argument("--runtime-base-seconds", type=float, default=0.0)\n    parser.add_argument("--preview-dir", type=Path, default=None)\n    parser.add_argument("--preview-interval", type=int, default=250)\n    parser.add_argument("--preview-max-side", type=int, default=300)\n    parser.add_argument("--preview-keep", type=int, default=8)\n    parser.add_argument("--mlx-cache-limit-gb", type=float, default=32.0)\n    parser.add_argument("--refine-enabled", action=argparse.BooleanOptionalAction, default=True)\n    parser.add_argument("--refine-prune-opa", type=float, default=0.005)\n    parser.add_argument("--refine-grow-grad2d", type=float, default=0.0002)\n    parser.add_argument("--refine-grow-scale3d", type=float, default=0.01)\n    parser.add_argument("--refine-grow-scale2d", type=float, default=0.05)\n    parser.add_argument("--refine-prune-scale3d", type=float, default=0.1)\n    parser.add_argument("--refine-prune-scale2d", type=float, default=0.15)\n    parser.add_argument("--refine-scale2d-stop-iter", type=int, default=0)\n    parser.add_argument("--refine-start-iter", type=int, default=500)\n    parser.add_argument("--refine-stop-iter", type=int, default=15000)\n    parser.add_argument("--refine-reset-every", type=int, default=3000)\n    parser.add_argument("--refine-every", type=int, default=100)\n    parser.add_argument("--refine-pause-after-reset", type=int, default=0)\n    parser.add_argument("--spz-scale-mode", choices=("direct", "scanner_axis"), default="direct")\n    parser.add_argument("--spz-rotation-mode", choices=("direct", "position_axis", "fastgs_conjugate", "position_conjugate"), default="position_axis")\n    parser.add_argument("--spz-quat-order", choices=("wxyz", "xyzw"), default="xyzw")\n    parser.add_argument("--spz-color-mode", choices=("sh", "raw_rgb"), default="sh")\n    return parser.parse_args()\n\n\ndef main() -> None:\n    args = parse_args()\n    if args.mask_dir is not None:\n        args.mask_dir = args.mask_dir.expanduser().resolve()\n    if args.extra_points_npz is not None:\n        args.extra_points_npz = args.extra_points_npz.expanduser().resolve()\n    if args.mlx_cache_limit_gb < 0.0:\n        raise ValueError("--mlx-cache-limit-gb must be nonnegative")\n    if args.checkpoint_every < 0 or args.preview_interval < 0:\n        raise ValueError("checkpoint/preview intervals must be nonnegative")\n    if args.checkpoint_keep < 1 or args.preview_keep < 1:\n        raise ValueError("checkpoint/preview retention must be >= 1")\n    if args.preview_max_side < 64:\n        raise ValueError("--preview-max-side must be >= 64")\n    signal.signal(signal.SIGTERM, _signal_stop)\n    signal.signal(signal.SIGINT, _signal_stop)\n    cache_limit_bytes = int(args.mlx_cache_limit_gb * 1024**3)\n    previous_cache_limit = mx.set_cache_limit(cache_limit_bytes)\n    log(\n        "mlx cache limit configured "\n        f"current={cache_limit_bytes} bytes ({args.mlx_cache_limit_gb:.2f} GiB) "\n        f"previous={previous_cache_limit} bytes"\n    )\n    if args.sh_degree < 0 or args.sh_degree > MAX_SUPPORTED_SH_DEGREE:\n        raise ValueError(f"--sh-degree must be in [0, {MAX_SUPPORTED_SH_DEGREE}]")\n    if args.sh_degree_interval <= 0:\n        raise ValueError("--sh-degree-interval must be positive")\n    if args.steps <= 0:\n        raise ValueError("--steps must be positive")\n    if args.batch_size <= 0:\n        raise ValueError("--batch-size must be positive")\n    if args.frame_step <= 0:\n        raise ValueError("--frame-step must be positive")\n    if args.eval_frame_step is not None and args.eval_frame_step <= 0:\n        raise ValueError("--eval-frame-step must be positive")\n    if args.refine_stop_iter <= args.refine_start_iter:\n        raise ValueError("--refine-stop-iter must be greater than --refine-start-iter")\n    for name, value in [\n        ("--init-scale", args.init_scale),\n        ("--opacity", args.opacity),\n        ("--means-lr", args.means_lr),\n        ("--scales-lr", args.scales_lr),\n        ("--opacities-lr", args.opacities_lr),\n        ("--quats-lr", args.quats_lr),\n        ("--sh0-lr", args.sh0_lr),\n        ("--shn-lr", args.shn_lr),\n    ]:\n        validate_positive(name, value)\n\n    width, height = args.width, args.height\n    if width <= 0 or height <= 0:\n        log(f"inferring image size data={args.data} factor={args.data_factor}")\n        inferred_width, inferred_height = infer_image_size(args.data, args.data_factor)\n        width = inferred_width if width <= 0 else width\n        height = inferred_height if height <= 0 else height\n\n    args.out_dir.mkdir(parents=True, exist_ok=True)\n    step_image_dir = args.out_dir / "step"\n    step_image_count = 0\n    if args.step_image_interval > 0:\n        step_image_dir.mkdir(parents=True, exist_ok=True)\n    checkpoint_root = (args.checkpoint_dir if args.checkpoint_dir is not None else args.out_dir / "checkpoints").expanduser().resolve()\n    preview_dir = (args.preview_dir if args.preview_dir is not None else args.out_dir / "previews").expanduser().resolve()\n    log(\n        "loading COLMAP scene "\n        f"data={args.data} factor={args.data_factor} size={width}x{height} "\n        f"normalize={args.normalize_world_space}"\n    )\n    scene = load_colmap_scene(\n        args.data,\n        factor=args.data_factor,\n        width=width,\n        height=height,\n        test_every=args.test_every,\n        normalize_world_space=args.normalize_world_space,\n    )\n    cameras = select_colmap_cameras(scene.cameras, "train", args.test_every, args.max_frames, args.frame_step, args.start_index)\n    eval_frame_step = args.frame_step if args.eval_frame_step is None else args.eval_frame_step\n    eval_cameras = (\n        select_colmap_cameras(scene.cameras, "val", args.test_every, args.eval_max_frames, eval_frame_step, args.eval_start_index)\n        if args.eval_max_frames > 0\n        else []\n    )\n    targets = [mx.array(load_target(camera.image_path, width, height)[None, ...], dtype=mx.float32) for camera in cameras]\n    eval_targets = [mx.array(load_target(camera.image_path, width, height)[None, ...], dtype=mx.float32) for camera in eval_cameras]\n    training_masks = [load_training_mask(args.mask_dir, camera, width, height) for camera in cameras]\n    if args.mask_dir is not None:\n        mean_keep = float(np.mean([mask.mean() for mask in training_masks])) if training_masks else 1.0\n        log(f"AI_MASK enabled=1 frames={len(training_masks)} mean_keep={mean_keep:.4f} dir={args.mask_dir}")\n    log(\n        "loaded targets "\n        f"train_frames={len(cameras)} eval_frames={len(eval_cameras)} "\n        f"scene_scale={scene.scene_scale:.8f}"\n    )\n\n    signature = checkpoint_signature(args, cameras)\n    resume_bundle = load_checkpoint_metadata(checkpoint_root, signature) if args.resume else None\n\n    log("preparing COLMAP sparse points")\n    points, colors, raw_point_count = prepare_colmap_points(scene, args.max_points, args.seed)\n    ai_extra_points, ai_extra_colors = load_extra_points(args.extra_points_npz)\n    ai_extra_count = int(ai_extra_points.shape[0])\n    if ai_extra_count:\n        points = np.concatenate([points, ai_extra_points], axis=0).astype(np.float32)\n        colors = np.concatenate([colors, ai_extra_colors], axis=0).astype(np.float32)\n        log(f"AI_EXTRA_POINTS count={ai_extra_count} path={args.extra_points_npz}")\n    points, colors = append_random_gaussians(points, colors, args.num_random_gaussians, args.seed + 1009, args.random_gaussian_bounds_scale)\n    log(f"initializing KNN log scales points={points.shape[0]} init_scale={args.init_scale}")\n    log_scales = knn_log_scales_from_points(points, args.init_scale)\n    point_diagnostics = points_extent_diagnostics(points)\n    resolved_scene_scale = float(scene.scene_scale * 1.1 * args.global_scale)\n    means_lr = float(args.means_lr * resolved_scene_scale)\n    means_lr_final = means_lr * 0.01\n\n    log(\n        "initializing SH model "\n        f"gaussians={points.shape[0]} sh_degree={args.sh_degree} "\n        f"opacity={args.opacity}"\n    )\n    model = init_sh_model_from_points(points, colors, log_scales, args.opacity, args.sh_degree)\n    strategy_config = ScannerDefaultStrategyConfig(\n        enabled=args.refine_enabled,\n        prune_opa=args.refine_prune_opa,\n        grow_grad2d=args.refine_grow_grad2d,\n        grow_scale3d=args.refine_grow_scale3d,\n        grow_scale2d=args.refine_grow_scale2d,\n        prune_scale3d=args.refine_prune_scale3d,\n        prune_scale2d=args.refine_prune_scale2d,\n        refine_scale2d_stop_iter=args.refine_scale2d_stop_iter,\n        refine_start_iter=args.refine_start_iter,\n        refine_stop_iter=args.refine_stop_iter,\n        reset_every=args.refine_reset_every,\n        refine_every=args.refine_every,\n        pause_refine_after_reset=args.refine_pause_after_reset,\n        scene_scale=resolved_scene_scale,\n        absgrad=False,\n        revised_opacity=False,\n    )\n    strategy = ScannerDefaultStrategyRuntime(strategy_config, initial_gaussians=model.means.shape[1])\n    start_step = 0\n    checkpoint_meta = None\n    checkpoint_dir = None\n    runtime_base_seconds = max(0.0, float(args.runtime_base_seconds))\n    resumed_loss = None\n\n    if resume_bundle is not None:\n        checkpoint_dir, checkpoint_meta = resume_bundle\n        model = _model_from_checkpoint(checkpoint_dir / "model.npz")\n        start_step = int(checkpoint_meta.get("step", 0))\n        if start_step > args.steps:\n            raise RuntimeError(f"Checkpoint step {start_step} is beyond requested total steps {args.steps}")\n        runtime_base_seconds = max(runtime_base_seconds, float(checkpoint_meta.get("elapsed_seconds", 0.0)))\n        resumed_loss = checkpoint_meta.get("loss")\n        _restore_strategy_state(strategy, checkpoint_meta, checkpoint_dir / "runtime_arrays.npz")\n        log(\n            f"RESUME step={start_step} total={args.steps} gaussians={int(model.means.shape[1])} "\n            f"elapsed={runtime_base_seconds:.1f}s path={checkpoint_dir}"\n        )\n\n    initial_active_sh_degree = gsplat_active_sh_degree(start_step, args.sh_degree, args.sh_degree_interval)\n    active_sh_degree = int(checkpoint_meta.get("active_sh_degree", initial_active_sh_degree)) if checkpoint_meta else initial_active_sh_degree\n    sh_degree_events = list(checkpoint_meta.get("sh_degree_events") or []) if checkpoint_meta else []\n    if not sh_degree_events:\n        sh_degree_events = [{"step": int(start_step), "active_sh_degree": int(active_sh_degree)}]\n\n    log(f"running initial evaluation frames={len(cameras)} eval_frames={len(eval_cameras)}")\n    initial_stats = evaluate_frames(model, cameras, targets, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)\n    eval_initial_stats = evaluate_frames(model, eval_cameras, eval_targets, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size) if eval_cameras else []\n    initial_mean_loss = mean_loss(initial_stats)\n    log(f"initial evaluation complete initial_mean_loss={initial_mean_loss:.8f}")\n\n    sampler = FrameBatchSampler(\n        frame_count=len(cameras),\n        batch_size=args.batch_size,\n        mode=args.frame_sampling,\n        seed=args.seed + 7919 if args.frame_shuffle_seed is None else args.frame_shuffle_seed,\n    )\n    if checkpoint_meta is not None and checkpoint_dir is not None:\n        _restore_sampler_state(sampler, checkpoint_meta, checkpoint_dir / "runtime_arrays.npz")\n\n    def sh_loss_fn(\n        means: mx.array,\n        quats: mx.array,\n        log_scales_: mx.array,\n        features_dc: mx.array,\n        features_rest: mx.array,\n        opacity_logits: mx.array,\n        viewspace_points: mx.array,\n        viewmats: mx.array,\n        ks: mx.array,\n        target: mx.array,\n        mask: mx.array,\n    ) -> mx.array:\n        local = ScannerPointsSHModel.from_arrays(means, quats, log_scales_, features_dc, features_rest, opacity_logits)\n        losses = []\n        radii = []\n        batch = int(viewmats.shape[1])\n        for idx in range(batch):\n            render = render_sh_model(\n                local,\n                viewspace_points[:, idx : idx + 1],\n                viewmats[:, idx : idx + 1],\n                ks[:, idx : idx + 1],\n                width,\n                height,\n                args.tile_size,\n                active_sh_degree,\n            )\n            losses.append(loss_components(render["render_colors"], target[idx : idx + 1], args.ssim_lambda, args.ssim_window_size, mask[idx : idx + 1])["loss"])\n            radii.append(render["radii"])\n        return mx.mean(mx.stack(losses)), mx.concatenate(radii, axis=1)\n\n    grad_fn = mx.value_and_grad(sh_loss_fn, argnums=(0, 1, 2, 3, 4, 5, 6))\n    optimizers = {\n        "means": Adam(learning_rate=means_lr),\n        "quats": Adam(learning_rate=args.quats_lr),\n        "log_scales": Adam(learning_rate=args.scales_lr),\n        "features_dc": Adam(learning_rate=args.sh0_lr),\n        "features_rest": Adam(learning_rate=args.shn_lr),\n        "opacity_logits": Adam(learning_rate=args.opacities_lr),\n    }\n    if checkpoint_meta is not None and checkpoint_dir is not None:\n        for optimizer_name, optimizer in optimizers.items():\n            optimizer_path = checkpoint_dir / f"optimizer_{optimizer_name}.safetensors"\n            if not optimizer_path.exists():\n                raise RuntimeError(f"Checkpoint missing optimizer state: {optimizer_path.name}")\n            _load_optimizer_state(optimizer_path, optimizer)\n\n    lr_schedules = {\n        "means": make_lr_schedule(means_lr, means_lr_final, 1.0, args.steps),\n        "quats": make_lr_schedule(args.quats_lr, None, 1.0, args.steps),\n        "log_scales": make_lr_schedule(args.scales_lr, None, 1.0, args.steps),\n        "features_dc": make_lr_schedule(args.sh0_lr, None, 1.0, args.steps),\n        "features_rest": make_lr_schedule(args.shn_lr, None, 1.0, args.steps),\n        "opacity_logits": make_lr_schedule(args.opacities_lr, None, 1.0, args.steps),\n    }\n\n    last_loss = None if resumed_loss is None else float(resumed_loss)\n    last_viewspace_grad = None\n    last_viewspace_grad_norm = None\n    run_started_at = time.time()\n    preview_camera = eval_cameras[0] if eval_cameras else cameras[len(cameras) // 2]\n    log(f"entering training loop steps={args.steps} start_step={start_step} batch_size={args.batch_size}")\n    for step in range(start_step + 1, args.steps + 1):\n        latest_lrs = {}\n        for name, schedule in lr_schedules.items():\n            lr = lr_for_step(schedule, step)\n            optimizers[name].learning_rate = lr\n            schedule["latest"] = float(lr)\n            latest_lrs[name] = float(lr)\n        if step == 1 or step == args.steps or step % args.log_interval == 0:\n            for schedule in lr_schedules.values():\n                schedule["history"].append({"step": int(step), "lr": float(schedule["latest"])})\n\n        next_active_sh_degree = gsplat_active_sh_degree(step, args.sh_degree, args.sh_degree_interval)\n        if next_active_sh_degree != active_sh_degree:\n            active_sh_degree = next_active_sh_degree\n            sh_degree_events.append({"step": int(step), "active_sh_degree": int(active_sh_degree)})\n\n        batch_ids = sampler.next_batch()\n        batch_frame_indices = [int(cameras[idx].index) for idx in batch_ids]\n        target = target_batch_array(targets, batch_ids)\n        mask = mx.array(np.stack([training_masks[idx] for idx in batch_ids], axis=0), dtype=mx.float32)\n        viewmats, ks = camera_batch_arrays(cameras, batch_ids)\n        viewspace_points = mx.zeros((1, len(batch_ids), model.means.shape[1], 2), dtype=mx.float32)\n        (loss, strategy_radii), grads = grad_fn(\n            model.means,\n            model.quats,\n            model.log_scales,\n            model.features_dc,\n            model.features_rest,\n            model.opacity_logits,\n            viewspace_points,\n            viewmats,\n            ks,\n            target,\n            mask,\n        )\n        d_means, d_quats, d_log_scales, d_features_dc, d_features_rest, d_opacity_logits, d_viewspace = grads\n        mx.eval(loss, d_viewspace)\n        last_loss = float(np.asarray(loss))\n        last_viewspace_grad = d_viewspace\n        last_viewspace_grad_norm = float(np.linalg.norm(np.asarray(d_viewspace)))\n\n        optimizers["means"].update(model, {"means": d_means})\n        optimizers["quats"].update(model, {"quats": d_quats})\n        optimizers["log_scales"].update(model, {"log_scales": d_log_scales})\n        optimizers["features_dc"].update(model, {"features_dc": d_features_dc})\n        optimizers["features_rest"].update(model, {"features_rest": d_features_rest})\n        optimizers["opacity_logits"].update(model, {"opacity_logits": d_opacity_logits})\n        model.quats = normalize_quats(model.quats)\n        mx.eval(model.means, model.quats, model.log_scales, model.features_dc, model.features_rest, model.opacity_logits)\n\n        if strategy.config.enabled:\n            strategy.update_state(d_viewspace, strategy_radii, width=width, height=height, n_cameras=len(batch_ids))\n        strategy.after_optimizer_step(step, model, optimizers, "sh")\n\n        if step == start_step + 1 or step == args.steps or step % args.log_interval == 0:\n            log(\n                f"step={step:04d} frames={batch_frame_indices} "\n                f"sh={active_sh_degree} loss={last_loss:.8f} "\n                f"gaussians={int(model.means.shape[1])} "\n                f"means_lr={latest_lrs[\'means\']:.8g} viewspace_grad_norm={last_viewspace_grad_norm:.8f}"\n            )\n\n        elapsed_seconds = runtime_base_seconds + (time.time() - run_started_at)\n        should_checkpoint = (\n            args.checkpoint_every > 0\n            and (step % args.checkpoint_every == 0 or step == args.steps)\n        )\n        if should_checkpoint:\n            save_training_checkpoint(\n                checkpoint_root,\n                args.checkpoint_keep,\n                signature,\n                step,\n                args.steps,\n                model,\n                optimizers,\n                sampler,\n                strategy,\n                active_sh_degree,\n                sh_degree_events,\n                last_loss,\n                elapsed_seconds,\n                "interval" if step < args.steps else "final",\n            )\n\n        if args.preview_interval > 0 and (step % args.preview_interval == 0 or step == args.steps):\n            preview = render_live_preview(\n                model,\n                preview_camera,\n                width,\n                height,\n                args.tile_size,\n                active_sh_degree,\n                args.preview_max_side,\n            )\n            preview_path = write_live_preview(preview_dir, args.preview_keep, step, preview)\n            log(\n                f"PREVIEW step={step} path={preview_path} "\n                f"width={int(preview.shape[1])} height={int(preview.shape[0])}"\n            )\n\n        if args.step_image_interval > 0 and step % args.step_image_interval == 0:\n            step_image_count += 1\n            image = render_step_grid(model, cameras, width, height, args.tile_size, active_sh_degree)\n            out_path = step_image_dir / f"out_{step_image_count:06d}.png"\n            write_png(out_path, image)\n            log(f"wrote step image step={step} path={out_path}")\n\n        if _STOP_REQUESTED:\n            if not should_checkpoint:\n                save_training_checkpoint(\n                    checkpoint_root,\n                    args.checkpoint_keep,\n                    signature,\n                    step,\n                    args.steps,\n                    model,\n                    optimizers,\n                    sampler,\n                    strategy,\n                    active_sh_degree,\n                    sh_degree_events,\n                    last_loss,\n                    elapsed_seconds,\n                    "signal",\n                )\n            log(f"STOPPED step={step} checkpoint_saved=1")\n            raise KeyboardInterrupt\n\n    log(f"running final evaluation frames={len(cameras)} eval_frames={len(eval_cameras)}")\n    final_stats = evaluate_frames(model, cameras, targets, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size)\n    eval_final_stats = evaluate_frames(model, eval_cameras, eval_targets, width, height, args.tile_size, active_sh_degree, args.ssim_lambda, args.ssim_window_size) if eval_cameras else []\n    final_mean_loss = mean_loss(final_stats)\n\n    for initial, final, target in zip(initial_stats, final_stats, targets, strict=True):\n        frame_index = final["frame_index"]\n        write_png(args.out_dir / f"compare_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))\n    for initial, final, target in zip(eval_initial_stats, eval_final_stats, eval_targets, strict=True):\n        frame_index = final["frame_index"]\n        write_png(args.out_dir / f"compare_eval_frame_{frame_index:05d}.png", image_to_u8(concat_compare(np.asarray(target[0]), initial["image"], final["image"])))\n\n    if last_loss is None or not np.isfinite(final_mean_loss):\n        raise AssertionError("360 training loss should be finite")\n    if start_step < args.steps and (last_viewspace_grad is None or not np.any(np.abs(np.asarray(last_viewspace_grad)) > 1.0e-8)):\n        raise AssertionError("360 training expected nonzero viewspace_points gradient")\n\n    out_spz = args.out_spz if args.out_spz is not None else args.out_dir / "trained_360_points.spz"\n    spz_diagnostics = spz_export_diagnostics(model, "sh", active_sh_degree, args.spz_scale_mode, args.spz_rotation_mode, args.spz_quat_order, args.spz_color_mode)\n    exported_gaussians = export_trained_spz(out_spz, model, "sh", active_sh_degree, args.spz_scale_mode, args.spz_rotation_mode, args.spz_quat_order, args.spz_color_mode)\n    spz_size = out_spz.stat().st_size\n    if spz_size <= 0:\n        raise AssertionError(f"SPZ output is empty: {out_spz}")\n\n    model_means_np = np.asarray(model.means[0], dtype=np.float32)\n    robust_scene_center = np.median(model_means_np, axis=0) if model_means_np.size else np.zeros(3, dtype=np.float32)\n    all_visible = [int(item["visible_gaussians"]) for item in final_stats + eval_final_stats]\n    max_visible = max(all_visible, default=1)\n\n    frame_summaries = [\n        {\n            "frame_index": int(final["frame_index"]),\n            "image_name": str(camera.image_name),\n            "initial_loss": float(initial["loss"]),\n            "final_loss": float(final["loss"]),\n            "initial_loss_components": initial["loss_components"],\n            "final_loss_components": final["loss_components"],\n            "initial_psnr": float(initial["psnr"]),\n            "final_psnr": float(final["psnr"]),\n            "initial_visible_gaussians": int(initial["visible_gaussians"]),\n            "final_visible_gaussians": int(final["visible_gaussians"]),\n            "initial_intersections": int(initial["intersections"]),\n            "final_intersections": int(final["intersections"]),\n            "view": camera_view_metadata(camera, robust_scene_center, float(final["psnr"]), int(final["visible_gaussians"]), max_visible, height),\n        }\n        for camera, initial, final in zip(cameras, initial_stats, final_stats, strict=True)\n    ]\n    eval_frame_summaries = [\n        {\n            "frame_index": int(final["frame_index"]),\n            "image_name": str(camera.image_name),\n            "initial_loss": float(initial["loss"]),\n            "final_loss": float(final["loss"]),\n            "initial_loss_components": initial["loss_components"],\n            "final_loss_components": final["loss_components"],\n            "initial_psnr": float(initial["psnr"]),\n            "final_psnr": float(final["psnr"]),\n            "initial_visible_gaussians": int(initial["visible_gaussians"]),\n            "final_visible_gaussians": int(final["visible_gaussians"]),\n            "initial_intersections": int(initial["intersections"]),\n            "final_intersections": int(final["intersections"]),\n            "view": camera_view_metadata(camera, robust_scene_center, float(final["psnr"]), int(final["visible_gaussians"]), max_visible, height),\n        }\n        for camera, initial, final in zip(eval_cameras, eval_initial_stats, eval_final_stats, strict=True)\n    ]\n\n    view_candidates = [item["view"] for item in frame_summaries + eval_frame_summaries if item.get("view")]\n    best_view = max(view_candidates, key=lambda item: float(item.get("score", -1.0e9))) if view_candidates else None\n    if best_view:\n        log(\n            f"BEST_VIEW image={best_view[\'image_name\']} psnr={best_view[\'psnr\']:.2f} "\n            f"visible={best_view[\'visible_gaussians\']} score={best_view[\'score\']:.2f}"\n        )\n    eval_final_mean_loss = mean_loss(eval_final_stats) if eval_final_stats else None\n    refinement_summary = strategy.summary()\n    final_opacity_diagnostics = opacity_diagnostics(model)\n    summary = {\n        "dataset_type": "colmap",\n        "dataset": str(args.data),\n        "data_factor": int(args.data_factor),\n        "test_every": int(args.test_every),\n        "normalize_world_space": bool(args.normalize_world_space),\n        "width": int(width),\n        "height": int(height),\n        "raw_point_count": int(raw_point_count),\n        "exported_gaussians": int(exported_gaussians),\n        "max_points": int(args.max_points),\n        "point_cloud_gaussians": int(points.shape[0] - args.num_random_gaussians),\n        "random_gaussians": int(args.num_random_gaussians),\n        "ai_mask_dir": str(args.mask_dir) if args.mask_dir is not None else None,\n        "ai_extra_points_npz": str(args.extra_points_npz) if args.extra_points_npz is not None else None,\n        "ai_extra_points": int(ai_extra_count),\n        "frames": len(cameras),\n        "eval_frames": len(eval_cameras),\n        "steps": int(args.steps),\n        "resumed_from_step": int(start_step),\n        "checkpoint_every": int(args.checkpoint_every),\n        "checkpoint_dir": str(checkpoint_root),\n        "preview_interval": int(args.preview_interval),\n        "preview_dir": str(preview_dir),\n        "total_runtime_seconds": float(runtime_base_seconds + (time.time() - run_started_at)),\n        "step_image_interval": int(args.step_image_interval),\n        "step_image_count": int(step_image_count),\n        "step_image_dir": str(step_image_dir) if args.step_image_interval > 0 else None,\n        "mlx_cache_limit_bytes": int(cache_limit_bytes),\n        "mlx_cache_limit_gb": float(args.mlx_cache_limit_gb),\n        "mlx_previous_cache_limit_bytes": int(previous_cache_limit),\n        "scene_scale": float(scene.scene_scale),\n        "resolved_scene_scale": float(resolved_scene_scale),\n        "initialization": {\n            "type": "sfm_plus_ai_depth" if ai_extra_count else "sfm",\n            "scale_rule": "average distance to 3 nearest neighbors times init_scale",\n            "init_scale": float(args.init_scale),\n            "opacity": float(args.opacity),\n            "point_extent": point_diagnostics,\n            "log_scale_min": float(log_scales.min()),\n            "log_scale_mean": float(log_scales.mean()),\n            "log_scale_max": float(log_scales.max()),\n        },\n        "gsplat_default_parity": {\n            "max_steps": args.steps == 30000,\n            "sh_degree": args.sh_degree == 3,\n            "sh_degree_interval": args.sh_degree_interval == 1000,\n            "ssim_lambda": args.ssim_lambda == 0.2,\n            "init_opacity": args.opacity == 0.1,\n            "default_strategy": bool(args.refine_enabled),\n        },\n        "dataloader": sampler.summary(cameras),\n        "loss_config": {\n            "mode": "l1_ssim",\n            "formula": "(1 - ssim_lambda) * L1 + ssim_lambda * (1 - SSIM)",\n            "ssim_lambda": float(args.ssim_lambda),\n            "ssim_window_size": int(args.ssim_window_size),\n        },\n        "learning_rate_schedule": lr_schedules,\n        "initial_mean_loss": float(initial_mean_loss),\n        "final_mean_loss": float(final_mean_loss),\n        "eval_final_mean_loss": eval_final_mean_loss,\n        "last_viewspace_grad_norm": last_viewspace_grad_norm,\n        "spz": str(out_spz),\n        "spz_file_size_bytes": int(spz_size),\n        "spz_export_diagnostics": spz_diagnostics,\n        "color_mode": "spherical_harmonics",\n        "active_sh_degree_final": int(active_sh_degree),\n        "sh_degree_schedule": {\n            "start": 0,\n            "target": int(args.sh_degree),\n            "interval": int(args.sh_degree_interval),\n            "formula": "min(step // sh_degree_interval, sh_degree)",\n            "events": sh_degree_events,\n        },\n        "final_opacity_diagnostics": final_opacity_diagnostics,\n        "refinement_strategy": refinement_summary,\n        "best_view": best_view,\n        "frame_summaries": frame_summaries,\n        "eval_frame_summaries": eval_frame_summaries,\n    }\n    out_model_npz = args.out_model_npz if args.out_model_npz is not None else args.out_dir / "trained_model_params.npz"\n    summary["model_npz"] = str(out_model_npz)\n    save_model_parameters_npz(out_model_npz, model, "sh", active_sh_degree, args.sh_degree, summary)\n    (args.out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")\n    out_spz.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")\n\n    log(\n        "360 points multi-view training ok "\n        f"initial_mean_loss={initial_mean_loss:.8f} final_mean_loss={final_mean_loss:.8f} "\n        f"last_viewspace_grad_norm={last_viewspace_grad_norm:.8f} "\n        f"resumed_from_step={start_step} runtime={runtime_base_seconds + (time.time() - run_started_at):.1f}s "\n        f"spz={out_spz} bytes={spz_size} output_dir={args.out_dir}"\n    )\n\n\nif __name__ == "__main__":\n    main()\n'

_AI_VISION_SOURCE = '#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport math\nimport os\nimport shutil\nimport subprocess\nimport sys\nimport time\nfrom dataclasses import dataclass\nfrom pathlib import Path\n\nimport numpy as np\nfrom PIL import Image, ImageDraw\n\nDEPTH_MODEL_DEFAULT = "depth-anything/Depth-Anything-V2-Small-hf"\nSEGMENT_MODEL_DEFAULT = "nvidia/segformer-b5-finetuned-ade-640-640"\n\n\ndef log(message: str) -> None:\n    print(message, flush=True)\n\n\ndef image_files(root: Path) -> list[Path]:\n    return sorted(\n        p for p in root.rglob("*")\n        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}\n    )\n\n\ndef resize_preview(image: Image.Image, max_side: int = 300) -> Image.Image:\n    image = image.convert("RGB")\n    scale = min(1.0, float(max_side) / max(image.size))\n    size = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))\n    return image.resize(size, Image.Resampling.LANCZOS)\n\n\ndef write_preview(image: Image.Image, path: Path, kind: str) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    preview = resize_preview(image, 300)\n    preview.save(path, format="JPEG", quality=88, optimize=True)\n    log(f"AI_PREVIEW kind={kind} path={path} width={preview.width} height={preview.height}")\n\n\ndef dataset_fingerprint(dataset: Path, args: argparse.Namespace) -> str:\n    root = dataset / "images"\n    files = image_files(root)\n    digest = hashlib.sha1()\n    digest.update(str(dataset.resolve()).encode())\n    digest.update(str(len(files)).encode())\n    for p in files[:8] + files[-8:]:\n        try:\n            stat = p.stat()\n            digest.update(str(p.relative_to(root)).encode())\n            digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())\n        except OSError:\n            pass\n    config = {\n        "depth": bool(args.depth_assist),\n        "mask_sky": bool(args.mask_sky),\n        "mask_dynamic": bool(args.mask_dynamic),\n        "depth_model": args.depth_model,\n        "segment_model": args.segment_model,\n        "max_depth_points": int(args.max_depth_points),\n        "depth_max_frames": int(args.depth_max_frames),\n        "mask_dilate": int(args.mask_dilate),\n    }\n    digest.update(json.dumps(config, sort_keys=True).encode())\n    return digest.hexdigest()\n\n\ndef torch_device():\n    import torch\n\n    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():\n        return torch.device("mps")\n    return torch.device("cpu")\n\n\ndef semantic_class_ids(id2label: dict, mask_sky: bool, mask_dynamic: bool) -> set[int]:\n    wanted: set[int] = set()\n    sky_terms = {"sky"}\n    dynamic_terms = {\n        "person", "car", "truck", "bus", "van", "bicycle", "bike",\n        "motorcycle", "motorbike", "minibike", "scooter"\n    }\n    for raw_id, raw_label in id2label.items():\n        try:\n            class_id = int(raw_id)\n        except (TypeError, ValueError):\n            continue\n        label = str(raw_label).strip().lower().replace("_", " ")\n        tokens = {part.strip() for part in label.replace("/", ",").split(",") if part.strip()}\n        if mask_sky and any(term in tokens or label == term for term in sky_terms):\n            wanted.add(class_id)\n        if mask_dynamic and any(term in tokens or term in label.split() for term in dynamic_terms):\n            wanted.add(class_id)\n    return wanted\n\n\ndef dilate_blocked(blocked: np.ndarray, pixels: int) -> np.ndarray:\n    if pixels <= 0:\n        return blocked\n    from PIL import ImageFilter\n    size = max(3, int(pixels) * 2 + 1)\n    if size % 2 == 0:\n        size += 1\n    image = Image.fromarray((blocked.astype(np.uint8) * 255), mode="L")\n    return np.asarray(image.filter(ImageFilter.MaxFilter(size))) > 0\n\n\n\ndef run_semantic_masks(args: argparse.Namespace, output_dir: Path, files: list[Path]) -> tuple[Path | None, Path | None, dict]:\n    if not (args.mask_sky or args.mask_dynamic):\n        return None, None, {"enabled": False}\n\n    import torch\n    import torch.nn.functional as F\n    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation\n\n    device = torch_device()\n    log(f"AI_MODEL semantic={args.segment_model} device={device}")\n    processor = AutoImageProcessor.from_pretrained(args.segment_model)\n    model = SegformerForSemanticSegmentation.from_pretrained(args.segment_model).to(device).eval()\n    blocked_ids = semantic_class_ids(model.config.id2label, args.mask_sky, args.mask_dynamic)\n    log(f"AI_MASK_CLASSES ids={\',\'.join(str(v) for v in sorted(blocked_ids)) or \'none\'}")\n\n    mask_dir = output_dir / "masks"\n    mask_dir.mkdir(parents=True, exist_ok=True)\n    preview_path: Path | None = None\n    masked_ratios: list[float] = []\n    middle = len(files) // 2\n\n    for index, path in enumerate(files, start=1):\n        image = Image.open(path).convert("RGB")\n        inputs = processor(images=image, return_tensors="pt")\n        inputs = {key: value.to(device) for key, value in inputs.items()}\n        with torch.inference_mode():\n            logits = model(**inputs).logits\n            logits = F.interpolate(logits, size=(image.height, image.width), mode="bilinear", align_corners=False)\n            labels = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)\n\n        blocked = np.isin(labels, list(blocked_ids)) if blocked_ids else np.zeros_like(labels, dtype=bool)\n        blocked = dilate_blocked(blocked, args.mask_dilate)\n        keep = (~blocked).astype(np.uint8) * 255\n        relative = path.relative_to(args.dataset / "images").with_suffix(".png")\n        mask_path = mask_dir / relative\n        mask_path.parent.mkdir(parents=True, exist_ok=True)\n        Image.fromarray(keep, mode="L").save(mask_path)\n        masked_ratios.append(float(blocked.mean()))\n\n        if index - 1 == middle:\n            rgb = np.asarray(image, dtype=np.uint8).copy()\n            overlay = rgb.copy()\n            overlay[blocked] = np.array([255, 48, 82], dtype=np.uint8)\n            blended = np.where(blocked[..., None], (0.42 * rgb + 0.58 * overlay).astype(np.uint8), rgb)\n            canvas = Image.fromarray(blended, mode="RGB")\n            draw = ImageDraw.Draw(canvas)\n            draw.rounded_rectangle((8, 8, 196, 34), radius=8, fill=(7, 18, 27))\n            draw.text((17, 15), "AI ignored regions", fill=(103, 215, 255))\n            preview_path = output_dir / "previews" / "ai_mask_preview.jpg"\n            write_preview(canvas, preview_path, "mask")\n\n        log(f"AI_PROGRESS stage=mask current={index} total={len(files)}")\n\n    del model\n    if device.type == "mps":\n        try:\n            torch.mps.empty_cache()\n        except Exception:\n            pass\n\n    return mask_dir, preview_path, {\n        "enabled": True,\n        "model": args.segment_model,\n        "device": str(device),\n        "frames": len(files),\n        "mean_masked_ratio": float(np.mean(masked_ratios)) if masked_ratios else 0.0,\n        "mask_sky": bool(args.mask_sky),\n        "mask_dynamic": bool(args.mask_dynamic),\n        "blocked_class_ids": sorted(blocked_ids),\n    }\n\n\ndef predict_depth(model, processor, image: Image.Image, device) -> np.ndarray:\n    import torch\n    import torch.nn.functional as F\n\n    inputs = processor(images=image, return_tensors="pt")\n    inputs = {key: value.to(device) for key, value in inputs.items()}\n    with torch.inference_mode():\n        prediction = model(**inputs).predicted_depth\n        prediction = F.interpolate(\n            prediction.unsqueeze(1),\n            size=(image.height, image.width),\n            mode="bicubic",\n            align_corners=False,\n        ).squeeze(1)[0]\n    return prediction.detach().float().cpu().numpy()\n\n\ndef robust_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray] | None:\n    finite = np.isfinite(x) & np.isfinite(y)\n    x = np.asarray(x[finite], dtype=np.float64)\n    y = np.asarray(y[finite], dtype=np.float64)\n    if x.size < 20 or float(np.std(x)) < 1e-8:\n        return None\n    keep = np.ones(x.shape[0], dtype=bool)\n    a = b = 0.0\n    for _ in range(3):\n        A = np.column_stack([x[keep], np.ones(int(keep.sum()), dtype=np.float64)])\n        if A.shape[0] < 12:\n            return None\n        a, b = np.linalg.lstsq(A, y[keep], rcond=None)[0]\n        resid = y - (a * x + b)\n        med = float(np.median(resid[keep]))\n        mad = float(np.median(np.abs(resid[keep] - med))) + 1e-8\n        keep = np.abs(resid - med) <= max(3.5 * 1.4826 * mad, 1e-5)\n    pred = a * x + b\n    scale = max(float(np.median(np.abs(y))), 1e-6)\n    nrms = float(np.sqrt(np.mean((pred[keep] - y[keep]) ** 2)) / scale)\n    return float(a), float(b), nrms, keep\n\n\ndef colmap_observations(reconstruction, image, scene_camera, transform: np.ndarray, depth_map: np.ndarray):\n    xs: list[float] = []\n    zs: list[float] = []\n    xys: list[tuple[float, float]] = []\n\n    points2d = getattr(image, "points2D", None)\n    if points2d is None:\n        return np.empty(0), np.empty(0), np.empty((0, 2))\n\n    for point2d in points2d:\n        try:\n            if hasattr(point2d, "has_point3D") and not point2d.has_point3D():\n                continue\n        except Exception:\n            pass\n        point_id = getattr(point2d, "point3D_id", None)\n        if point_id is None:\n            continue\n        try:\n            point_id = int(point_id)\n        except (TypeError, ValueError):\n            continue\n        if point_id < 0:\n            continue\n        try:\n            point3d = reconstruction.points3D[point_id]\n        except Exception:\n            continue\n        xy = np.asarray(getattr(point2d, "xy", []), dtype=np.float64).reshape(-1)\n        if xy.size < 2:\n            continue\n        u, v = float(xy[0]), float(xy[1])\n        ui, vi = int(round(u)), int(round(v))\n        if not (0 <= ui < depth_map.shape[1] and 0 <= vi < depth_map.shape[0]):\n            continue\n        raw = np.asarray(point3d.xyz, dtype=np.float32).reshape(1, 3)\n        normalized = (raw @ transform[:3, :3].T + transform[:3, 3]).reshape(3)\n        cam = scene_camera.viewmat @ np.array([normalized[0], normalized[1], normalized[2], 1.0], dtype=np.float32)\n        z = float(cam[2])\n        d = float(depth_map[vi, ui])\n        if z > 1e-6 and math.isfinite(z) and math.isfinite(d) and abs(d) > 1e-8:\n            xs.append(d)\n            zs.append(z)\n            xys.append((u, v))\n\n    return np.asarray(xs, dtype=np.float64), np.asarray(zs, dtype=np.float64), np.asarray(xys, dtype=np.float64)\n\n\ndef best_depth_alignment(depth_samples: np.ndarray, z_samples: np.ndarray):\n    candidates = []\n    direct = robust_linear_fit(depth_samples, z_samples)\n    if direct is not None:\n        candidates.append(("depth", direct))\n    reciprocal_samples = 1.0 / np.maximum(np.abs(depth_samples), 1e-8)\n    reciprocal = robust_linear_fit(reciprocal_samples, z_samples)\n    if reciprocal is not None:\n        candidates.append(("inverse", reciprocal))\n    if not candidates:\n        return None\n    mode, fit = min(candidates, key=lambda item: item[1][2])\n    a, b, nrms, _keep = fit\n    return mode, a, b, nrms\n\n\ndef apply_depth_alignment(depth: np.ndarray, mode: str, a: float, b: float) -> np.ndarray:\n    source = depth if mode == "depth" else 1.0 / np.maximum(np.abs(depth), 1e-8)\n    return (a * source + b).astype(np.float32)\n\n\ndef depth_preview(depth: np.ndarray) -> Image.Image:\n    valid = depth[np.isfinite(depth)]\n    if valid.size == 0:\n        return Image.new("RGB", (300, 180), (10, 16, 24))\n    lo, hi = np.percentile(valid, [3, 97])\n    scale = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)\n    # Small dependency-free cyan -> yellow -> magenta visualization.\n    r = np.clip(1.8 * scale - 0.4, 0, 1)\n    g = np.clip(1.4 - np.abs(scale - 0.45) * 2.2, 0, 1)\n    b = np.clip(1.25 - 1.4 * scale, 0, 1)\n    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)\n    return Image.fromarray(rgb, mode="RGB")\n\n\n\n@dataclass(frozen=True)\nclass AITrainingCamera:\n    index: int\n    viewmat: np.ndarray\n    K: np.ndarray\n    image_path: Path\n    image_name: str\n\n\n@dataclass(frozen=True)\nclass AIColmapScene:\n    cameras: list[AITrainingCamera]\n    points: np.ndarray\n    colors: np.ndarray\n    transform: np.ndarray\n\n\ndef _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:\n    q = np.asarray(qvec, dtype=np.float64)\n    q = q / max(float(np.linalg.norm(q)), 1e-12)\n    w, x, y, z = q\n    return np.asarray([\n        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],\n        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],\n        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],\n    ], dtype=np.float32)\n\n\ndef _camera_k(model: str, params: list[float]) -> np.ndarray:\n    model = str(model).upper()\n    if model in {\n        "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL",\n        "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE",\n    }:\n        if len(params) < 3:\n            raise ValueError(f"Camera model {model} has insufficient parameters")\n        f, cx, cy = params[:3]\n        fx = fy = float(f)\n    else:\n        if len(params) < 4:\n            raise ValueError(f"Camera model {model} has insufficient parameters")\n        fx, fy, cx, cy = map(float, params[:4])\n    return np.asarray([\n        [fx, 0.0, cx],\n        [0.0, fy, cy],\n        [0.0, 0.0, 1.0],\n    ], dtype=np.float32)\n\n\ndef _ensure_colmap_text_model(dataset: Path, output_dir: Path) -> Path:\n    sparse0 = dataset / "sparse" / "0"\n    sparse_dir = sparse0 if sparse0.exists() else dataset / "sparse"\n    if not sparse_dir.exists():\n        raise RuntimeError(f"COLMAP sparse model not found under {dataset}")\n\n    # Some datasets are already text models.\n    if all((sparse_dir / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt")):\n        return sparse_dir\n\n    text_dir = output_dir / "colmap_text"\n    required = [text_dir / name for name in ("cameras.txt", "images.txt", "points3D.txt")]\n    if all(path.exists() and path.stat().st_size > 0 for path in required):\n        return text_dir\n\n    colmap = shutil.which("colmap")\n    if not colmap:\n        homebrew_colmap = Path("/opt/homebrew/bin/colmap")\n        if homebrew_colmap.exists():\n            colmap = str(homebrew_colmap)\n    if not colmap:\n        raise RuntimeError("COLMAP executable not found for AI depth geometry export.")\n\n    text_dir.mkdir(parents=True, exist_ok=True)\n    command = [\n        str(colmap),\n        "model_converter",\n        "--input_path", str(sparse_dir.resolve()),\n        "--output_path", str(text_dir.resolve()),\n        "--output_type", "TXT",\n    ]\n    result = subprocess.run(command, capture_output=True, text=True)\n    if result.returncode != 0:\n        raise RuntimeError(\n            "COLMAP text-model conversion failed: "\n            + (result.stderr or result.stdout or f"exit {result.returncode}").strip()[-1200:]\n        )\n    log(f"AI_COLMAP_TEXT path={text_dir}")\n    return text_dir\n\n\ndef _read_cameras_txt(path: Path) -> dict[int, dict]:\n    cameras: dict[int, dict] = {}\n    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():\n        line = raw.strip()\n        if not line or line.startswith("#"):\n            continue\n        parts = line.split()\n        if len(parts) < 5:\n            continue\n        camera_id = int(parts[0])\n        model = parts[1]\n        cameras[camera_id] = {\n            "model": model,\n            "width": int(parts[2]),\n            "height": int(parts[3]),\n            "params": [float(v) for v in parts[4:]],\n        }\n    return cameras\n\n\ndef _read_points3d_txt(path: Path) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:\n    lookup: dict[int, np.ndarray] = {}\n    points: list[np.ndarray] = []\n    colors: list[np.ndarray] = []\n    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():\n        line = raw.strip()\n        if not line or line.startswith("#"):\n            continue\n        parts = line.split()\n        if len(parts) < 8:\n            continue\n        point_id = int(parts[0])\n        xyz = np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)\n        rgb = np.asarray([float(parts[4]), float(parts[5]), float(parts[6])], dtype=np.float32) / 255.0\n        lookup[point_id] = xyz\n        points.append(xyz)\n        colors.append(rgb)\n    if not points:\n        raise RuntimeError("COLMAP text model contains no 3D points.")\n    return (\n        lookup,\n        np.stack(points, axis=0).astype(np.float32),\n        np.stack(colors, axis=0).astype(np.float32),\n    )\n\n\ndef _read_images_txt(path: Path) -> list[dict]:\n    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()\n    records: list[dict] = []\n    index = 0\n    while index < len(raw_lines):\n        line = raw_lines[index].strip()\n        if not line or line.startswith("#"):\n            index += 1\n            continue\n\n        parts = line.split(maxsplit=9)\n        if len(parts) < 10:\n            index += 1\n            continue\n\n        image_id = int(parts[0])\n        qvec = np.asarray([float(v) for v in parts[1:5]], dtype=np.float32)\n        tvec = np.asarray([float(v) for v in parts[5:8]], dtype=np.float32)\n        camera_id = int(parts[8])\n        name = parts[9]\n\n        points_line = raw_lines[index + 1].strip() if index + 1 < len(raw_lines) else ""\n        point_tokens = points_line.split()\n        observations: list[tuple[float, float, int]] = []\n        for offset in range(0, len(point_tokens) - 2, 3):\n            try:\n                observations.append((\n                    float(point_tokens[offset]),\n                    float(point_tokens[offset + 1]),\n                    int(point_tokens[offset + 2]),\n                ))\n            except (TypeError, ValueError):\n                continue\n\n        records.append({\n            "image_id": image_id,\n            "qvec": qvec,\n            "tvec": tvec,\n            "camera_id": camera_id,\n            "name": name,\n            "observations": observations,\n        })\n        index += 2\n\n    return sorted(records, key=lambda item: item["name"])\n\n\ndef _load_text_colmap_scene(dataset: Path, output_dir: Path, backend_test_dir_path: Path):\n    text_dir = _ensure_colmap_text_model(dataset, output_dir)\n    cameras_raw = _read_cameras_txt(text_dir / "cameras.txt")\n    point_lookup, raw_points, colors = _read_points3d_txt(text_dir / "points3D.txt")\n    image_records = _read_images_txt(text_dir / "images.txt")\n    if not image_records:\n        raise RuntimeError("COLMAP text model contains no registered images.")\n\n    # Use the exact normalization function used by the native Metal trainer.\n    backend_test_dir = str(backend_test_dir_path.resolve())\n    if backend_test_dir not in sys.path:\n        sys.path.insert(0, backend_test_dir)\n    from colmap_360_dataset import normalize_scene\n\n    w2c_mats: list[np.ndarray] = []\n    for record in image_records:\n        w2c = np.eye(4, dtype=np.float32)\n        w2c[:3, :3] = _qvec_to_rotmat(record["qvec"])\n        w2c[:3, 3] = record["tvec"]\n        w2c_mats.append(w2c)\n    camtoworlds = np.linalg.inv(np.stack(w2c_mats, axis=0)).astype(np.float32)\n    camtoworlds, scene_points, transform = normalize_scene(camtoworlds, raw_points)\n\n    training_cameras: list[AITrainingCamera] = []\n    observations_by_name: dict[str, list[tuple[float, float, int]]] = {}\n    for camera_index, (record, camtoworld) in enumerate(zip(image_records, camtoworlds, strict=True)):\n        camera_info = cameras_raw.get(record["camera_id"])\n        if camera_info is None:\n            continue\n        image_path = dataset / "images" / record["name"]\n        if not image_path.exists():\n            continue\n        with Image.open(image_path) as pil:\n            raw_width, raw_height = pil.size\n\n        K = _camera_k(camera_info["model"], camera_info["params"])\n        K[0, :] *= float(raw_width) / max(float(camera_info["width"]), 1.0)\n        K[1, :] *= float(raw_height) / max(float(camera_info["height"]), 1.0)\n\n        training_cameras.append(AITrainingCamera(\n            index=int(camera_index),\n            viewmat=np.linalg.inv(camtoworld).astype(np.float32),\n            K=K.astype(np.float32),\n            image_path=image_path,\n            image_name=record["name"],\n        ))\n        observations_by_name[record["name"]] = record["observations"]\n\n    if not training_cameras:\n        raise RuntimeError("No usable COLMAP cameras were found for AI depth alignment.")\n\n    return (\n        AIColmapScene(\n            cameras=training_cameras,\n            points=scene_points.astype(np.float32),\n            colors=colors.astype(np.float32),\n            transform=transform.astype(np.float32),\n        ),\n        point_lookup,\n        observations_by_name,\n    )\n\n\ndef _text_colmap_observations(\n    point_lookup: dict[int, np.ndarray],\n    observations: list[tuple[float, float, int]],\n    scene_camera: AITrainingCamera,\n    transform: np.ndarray,\n    depth_map: np.ndarray,\n):\n    xs: list[float] = []\n    zs: list[float] = []\n    xys: list[tuple[float, float]] = []\n\n    for u, v, point_id in observations:\n        if point_id < 0:\n            continue\n        raw = point_lookup.get(int(point_id))\n        if raw is None:\n            continue\n        ui, vi = int(round(u)), int(round(v))\n        if not (0 <= ui < depth_map.shape[1] and 0 <= vi < depth_map.shape[0]):\n            continue\n\n        normalized = (raw.reshape(1, 3) @ transform[:3, :3].T + transform[:3, 3]).reshape(3)\n        cam = scene_camera.viewmat @ np.asarray(\n            [normalized[0], normalized[1], normalized[2], 1.0],\n            dtype=np.float32,\n        )\n        z = float(cam[2])\n        d = float(depth_map[vi, ui])\n        if z > 1e-6 and math.isfinite(z) and math.isfinite(d) and abs(d) > 1e-8:\n            xs.append(d)\n            zs.append(z)\n            xys.append((float(u), float(v)))\n\n    return (\n        np.asarray(xs, dtype=np.float64),\n        np.asarray(zs, dtype=np.float64),\n        np.asarray(xys, dtype=np.float64),\n    )\n\n\ndef run_depth_assist(args: argparse.Namespace, output_dir: Path, mask_dir: Path | None) -> tuple[Path | None, Path | None, dict]:\n    if not args.depth_assist:\n        return None, None, {"enabled": False}\n\n    # Deliberately do NOT import pycolmap here. On macOS, pycolmap and PyTorch\n    # may each bring their own libomp runtime into the same Python process,\n    # causing OpenMP Error #15. COLMAP model conversion happens in a separate\n    # executable process and this worker parses the resulting text itself.\n    import torch\n    from transformers import AutoImageProcessor, AutoModelForDepthEstimation\n\n    scene, point_lookup, observations_by_name = _load_text_colmap_scene(\n        args.dataset,\n        output_dir,\n        args.backend_test_dir,\n    )\n    camera_by_name = {camera.image_name: camera for camera in scene.cameras}\n\n    candidate_names = [\n        name for name in sorted(camera_by_name)\n        if observations_by_name.get(name)\n    ]\n    if not candidate_names:\n        raise RuntimeError("No COLMAP camera observations are available for AI depth alignment.")\n\n    max_frames = max(1, min(int(args.depth_max_frames), len(candidate_names)))\n    selected_indices = np.linspace(0, len(candidate_names) - 1, max_frames, dtype=int)\n    selected_names = [candidate_names[i] for i in selected_indices]\n\n    device = torch_device()\n    log(f"AI_MODEL depth={args.depth_model} device={device}")\n    processor = AutoImageProcessor.from_pretrained(args.depth_model)\n    model = AutoModelForDepthEstimation.from_pretrained(args.depth_model).to(device).eval()\n\n    per_frame_budget = max(64, int(math.ceil(args.max_depth_points / max(1, len(selected_names)))))\n    all_points: list[np.ndarray] = []\n    all_colors: list[np.ndarray] = []\n    alignment_scores: list[float] = []\n    aligned_frames = 0\n    preview_path: Path | None = None\n    preview_target = len(selected_names) // 2\n\n    scene_center = np.median(scene.points, axis=0)\n    scene_radius = float(np.percentile(np.linalg.norm(scene.points - scene_center, axis=1), 98)) if len(scene.points) else 10.0\n    scene_radius = max(scene_radius, 1e-3)\n\n    for index, name in enumerate(selected_names, start=1):\n        camera = camera_by_name[name]\n        image = Image.open(camera.image_path).convert("RGB")\n        rgb = np.asarray(image, dtype=np.uint8)\n        raw_depth = predict_depth(model, processor, image, device)\n\n        depth_samples, z_samples, _xys = _text_colmap_observations(\n            point_lookup,\n            observations_by_name.get(name, []),\n            camera,\n            scene.transform,\n            raw_depth,\n        )\n        alignment = best_depth_alignment(depth_samples, z_samples)\n        log(f"AI_PROGRESS stage=depth current={index} total={len(selected_names)}")\n        if alignment is None:\n            continue\n\n        mode, a, b, nrms = alignment\n        if not math.isfinite(nrms) or nrms > 0.35:\n            log(f"AI_DEPTH_SKIP image={name} reason=weak_alignment nrms={nrms:.4f}")\n            continue\n\n        aligned = apply_depth_alignment(raw_depth, mode, a, b)\n        positive_z = z_samples[z_samples > 0]\n        if positive_z.size < 20:\n            continue\n        z_lo = max(1e-5, float(np.percentile(positive_z, 2)) * 0.45)\n        z_hi = float(np.percentile(positive_z, 98)) * 2.0\n\n        keep_mask = None\n        if mask_dir is not None:\n            candidate = mask_dir / Path(name).with_suffix(".png")\n            if candidate.exists():\n                keep_mask = np.asarray(Image.open(candidate).convert("L"), dtype=np.uint8) > 127\n                if keep_mask.shape != aligned.shape:\n                    keep_mask = np.asarray(\n                        Image.fromarray(keep_mask.astype(np.uint8) * 255).resize(\n                            image.size,\n                            Image.Resampling.NEAREST,\n                        )\n                    ) > 127\n\n        grid_n = max(8, int(math.ceil(math.sqrt(per_frame_budget))))\n        us = np.linspace(0, image.width - 1, grid_n, dtype=int)\n        vs = np.linspace(0, image.height - 1, grid_n, dtype=int)\n        uu, vv = np.meshgrid(us, vs)\n        uu = uu.reshape(-1)\n        vv = vv.reshape(-1)\n        z = aligned[vv, uu]\n        valid = np.isfinite(z) & (z >= z_lo) & (z <= z_hi)\n        if keep_mask is not None:\n            valid &= keep_mask[vv, uu]\n        uu, vv, z = uu[valid], vv[valid], z[valid]\n        if z.size == 0:\n            continue\n\n        fx, fy = float(camera.K[0, 0]), float(camera.K[1, 1])\n        cx, cy = float(camera.K[0, 2]), float(camera.K[1, 2])\n        x = (uu.astype(np.float32) - cx) / fx * z\n        y = (vv.astype(np.float32) - cy) / fy * z\n        cam_points = np.column_stack([x, y, z, np.ones_like(z)]).astype(np.float32)\n        world = (np.linalg.inv(camera.viewmat) @ cam_points.T).T[:, :3].astype(np.float32)\n        colors = rgb[vv, uu].astype(np.float32) / 255.0\n\n        bounds = np.linalg.norm(world - scene_center[None, :], axis=1) <= scene_radius * 3.0\n        world = world[bounds]\n        colors = colors[bounds]\n        if world.shape[0] > per_frame_budget:\n            pick = np.linspace(0, world.shape[0] - 1, per_frame_budget, dtype=int)\n            world = world[pick]\n            colors = colors[pick]\n        if world.size:\n            all_points.append(world)\n            all_colors.append(colors)\n            alignment_scores.append(float(nrms))\n            aligned_frames += 1\n\n        if index - 1 == preview_target:\n            preview_path = output_dir / "previews" / "ai_depth_preview.jpg"\n            preview_image = depth_preview(aligned)\n            draw = ImageDraw.Draw(preview_image)\n            draw.rounded_rectangle((8, 8, 196, 34), radius=8, fill=(7, 18, 27))\n            draw.text((17, 15), "AI relative depth", fill=(103, 215, 255))\n            write_preview(preview_image, preview_path, "depth")\n\n    del model\n    if device.type == "mps":\n        try:\n            torch.mps.empty_cache()\n        except Exception:\n            pass\n\n    if not all_points:\n        log(f"AI_DEPTH_POINTS count=0 aligned_frames={aligned_frames}")\n        return None, preview_path, {\n            "enabled": True,\n            "model": args.depth_model,\n            "device": str(device),\n            "extra_points": 0,\n            "aligned_frames": int(aligned_frames),\n            "warning": "Depth predictions could not be aligned reliably to the COLMAP reconstruction.",\n        }\n\n    points = np.concatenate(all_points, axis=0)\n    colors = np.concatenate(all_colors, axis=0)\n    if points.shape[0] > args.max_depth_points:\n        rng = np.random.default_rng(20260816)\n        keep = rng.choice(points.shape[0], size=int(args.max_depth_points), replace=False)\n        points = points[keep]\n        colors = colors[keep]\n\n    output_npz = output_dir / "depth_points.npz"\n    np.savez_compressed(\n        output_npz,\n        points=points.astype(np.float32),\n        colors=colors.astype(np.float32),\n    )\n    log(\n        f"AI_DEPTH_POINTS count={points.shape[0]} "\n        f"aligned_frames={aligned_frames} path={output_npz}"\n    )\n    return output_npz, preview_path, {\n        "enabled": True,\n        "model": args.depth_model,\n        "device": str(device),\n        "extra_points": int(points.shape[0]),\n        "aligned_frames": int(aligned_frames),\n        "mean_alignment_nrms": float(np.mean(alignment_scores)) if alignment_scores else None,\n    }\n\n\n\ndef parse_args() -> argparse.Namespace:\n    parser = argparse.ArgumentParser(description="Splat Studio local AI vision preprocessing")\n    parser.add_argument("--dataset", type=Path, required=True)\n    parser.add_argument("--backend-test-dir", type=Path, required=True)\n    parser.add_argument("--output-dir", type=Path, required=True)\n    parser.add_argument("--depth-assist", action=argparse.BooleanOptionalAction, default=False)\n    parser.add_argument("--mask-sky", action=argparse.BooleanOptionalAction, default=False)\n    parser.add_argument("--mask-dynamic", action=argparse.BooleanOptionalAction, default=False)\n    parser.add_argument("--max-depth-points", type=int, default=40000)\n    parser.add_argument("--depth-max-frames", type=int, default=64)\n    parser.add_argument("--mask-dilate", type=int, default=7)\n    parser.add_argument("--depth-model", default=DEPTH_MODEL_DEFAULT)\n    parser.add_argument("--segment-model", default=SEGMENT_MODEL_DEFAULT)\n    return parser.parse_args()\n\n\ndef main() -> None:\n    args = parse_args()\n    args.dataset = args.dataset.expanduser().resolve()\n    args.backend_test_dir = args.backend_test_dir.expanduser().resolve()\n    args.output_dir = args.output_dir.expanduser().resolve()\n    args.output_dir.mkdir(parents=True, exist_ok=True)\n\n    files = image_files(args.dataset / "images")\n    if not files:\n        raise RuntimeError(f"No images found under {args.dataset / \'images\'}")\n\n    signature = dataset_fingerprint(args.dataset, args)\n    manifest_path = args.output_dir / "ai_manifest.json"\n    if manifest_path.exists():\n        try:\n            existing = json.loads(manifest_path.read_text(encoding="utf-8"))\n        except Exception:\n            existing = {}\n        mask_ok = not (args.mask_sky or args.mask_dynamic) or (args.output_dir / "masks").is_dir()\n        depth_path = args.output_dir / "depth_points.npz"\n        depth_ok = not args.depth_assist or depth_path.exists() or existing.get("depth", {}).get("extra_points") == 0\n        if existing.get("signature") == signature and mask_ok and depth_ok:\n            log(f"AI_CACHE_HIT manifest={manifest_path}")\n            preview = existing.get("latest_preview")\n            if preview and Path(preview).exists():\n                log(f"AI_PREVIEW kind={existing.get(\'latest_preview_kind\', \'ai\')} path={preview} width=300 height=300")\n            if depth_path.exists():\n                try:\n                    with np.load(depth_path) as data:\n                        count = int(data["points"].shape[0])\n                    log(f"AI_DEPTH_POINTS count={count} aligned_frames={existing.get(\'depth\', {}).get(\'aligned_frames\', 0)} path={depth_path}")\n                except Exception:\n                    pass\n            return\n\n    started = time.time()\n    mask_dir, mask_preview, mask_summary = run_semantic_masks(args, args.output_dir, files)\n    depth_npz, depth_preview_path, depth_summary = run_depth_assist(args, args.output_dir, mask_dir)\n    latest_preview = depth_preview_path or mask_preview\n    manifest = {\n        "version": 1,\n        "signature": signature,\n        "created_at": time.time(),\n        "runtime_seconds": time.time() - started,\n        "dataset": str(args.dataset),\n        "mask_dir": str(mask_dir) if mask_dir else None,\n        "extra_points_npz": str(depth_npz) if depth_npz else None,\n        "latest_preview": str(latest_preview) if latest_preview else None,\n        "latest_preview_kind": "depth" if depth_preview_path else ("mask" if mask_preview else None),\n        "mask": mask_summary,\n        "depth": depth_summary,\n    }\n    temporary = manifest_path.with_suffix(".tmp")\n    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")\n    temporary.replace(manifest_path)\n    log(f"AI_DONE manifest={manifest_path} runtime={manifest[\'runtime_seconds\']:.1f}s")\n\n\nif __name__ == "__main__":\n    main()\n'


def ensure_runtime_ai_vision():
    """Materialise the local AI preprocessing worker without importing PyTorch into Streamlit."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        current = AI_VISION_SCRIPT.read_text(encoding="utf-8") if AI_VISION_SCRIPT.exists() else None
    except OSError:
        current = None
    if current != _AI_VISION_SOURCE:
        AI_VISION_SCRIPT.write_text(_AI_VISION_SOURCE, encoding="utf-8")


def ensure_runtime_trainer():
    """Materialise Splat Studio's resumable native Metal trainer beside the app runtime."""
    if os.getenv("SPLAT_STUDIO_TRAINER"):
        return
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        current = TRAINER_SCRIPT.read_text(encoding="utf-8") if TRAINER_SCRIPT.exists() else None
    except OSError:
        current = None
    if current != _FAST_METAL_TRAINER_SOURCE:
        TRAINER_SCRIPT.write_text(_FAST_METAL_TRAINER_SOURCE, encoding="utf-8")


ensure_runtime_trainer()
ensure_runtime_ai_vision()

# PlayCanvas open-source components. These stable refs can be overridden with env vars.
SUPERSPLAT_REF = os.getenv("SUPERSPLAT_REF", "v2.32.3")
SUPERSPLAT_VIEWER_REF = os.getenv("SUPERSPLAT_VIEWER_REF", "v1.29.1")
THIRD_PARTY_ROOT = Path(os.getenv("SPLAT_THIRD_PARTY", str(APP_DIR / ".splat_studio" / "third_party"))).resolve()
SETTINGS_FILE = Path(os.getenv("SPLAT_SETTINGS_FILE", str(APP_DIR / ".splat_studio/settings.json"))).resolve()
RUN_HISTORY_FILE = APP_DIR / ".splat_studio" / "run_history.json"
PROJECT_STATE_FILENAME = ".splat_studio_project.json"
SUPERSPLAT_DIR = THIRD_PARTY_ROOT / "supersplat"
SUPERSPLAT_VIEWER_DIR = THIRD_PARTY_ROOT / "supersplat-viewer"
SUPERSPLAT_EDITOR_DIST = SUPERSPLAT_DIR / "dist"
SUPERSPLAT_VIEWER_PUBLIC = SUPERSPLAT_VIEWER_DIR / "public"
SPLAT_TRANSFORM_ROOT = THIRD_PARTY_ROOT / "splat-transform"
SPLAT_TRANSFORM_BIN = SPLAT_TRANSFORM_ROOT / "node_modules" / ".bin" / "splat-transform"

# SuperSplat and SuperSplat Viewer are MIT licensed. Their original LICENSE files
# remain inside the cloned repositories under THIRD_PARTY_ROOT.

# Overall progress remains weighted by workload, while the UI separately shows
# an explicit stable pipeline stage number. 100% is reserved for verified completion.
EXTRACT_RANGE = (3.0, 12.0)
COLMAP_RANGE = (12.0, 58.0)
TRAIN_RANGE = (70.0, 96.0)

PIPELINE_STAGES = (
    "Prepare workspace",
    "Prepare source frames",
    "Extract features",
    "Match frames",
    "Solve cameras",
    "Validate cameras",
    "Prepare training data",
    "Prepare training views",
    "Initialise Gaussians",
    "Optimise splats",
    "Export result",
    "Finalise result",
)
PIPELINE_STAGE_TOTAL = len(PIPELINE_STAGES)

# Weighted internal progress ranges for each explicit pipeline stage.
# These let the UI display a separate percentage for the CURRENT stage.
PIPELINE_STAGE_RANGES = {
    1: (0.0, 3.0),    # Prepare workspace
    2: (3.0, 12.0),   # Prepare source frames (video sampling or uploaded photos)
    3: (12.0, 24.0),  # Extract features
    4: (24.0, 36.0),  # Match frames
    5: (36.0, 50.0),  # Solve cameras
    6: (50.0, 58.0),  # Validate cameras / repair if required
    7: (58.0, 62.0),  # Prepare training data
    8: (62.0, 66.0),  # Prepare training views / optional AI preprocessing
    9: (66.0, 70.0),  # Initialise Gaussians
    10: (70.0, 96.0), # Optimise splats
    11: (96.0, 99.0), # Export result
    12: (99.0, 100.0),# Finalise result
}


def pipeline_stage_progress(stage_index, overall_percent):
    """Convert weighted overall progress into 0..100 progress inside a stage."""
    start, end = PIPELINE_STAGE_RANGES.get(int(stage_index), (0.0, 100.0))
    if end <= start:
        return 0.0
    value = (float(overall_percent) - start) / (end - start) * 100.0
    return max(0.0, min(100.0, value))


def pipeline_stage_for(label, detail=""):
    """Return a stable 1-based stage index for the current pipeline activity."""
    text = f"{label} {detail}".lower()

    if "complete" in text and "camera solve complete" not in text:
        return 12
    if any(token in text for token in ("finalis", "validating gaussian export", "optimization complete")):
        return 12
    if any(token in text for token in ("export", "writing ply", "writing .splat", "spz")):
        return 11
    if any(token in text for token in ("optimizing splats", "optimising splats", "training step", "checkpoint", "refinement", "resuming safely")):
        return 10
    if any(token in text for token in ("initialising gaussian", "initializing gaussian", "stage init")):
        return 9
    if any(token in text for token in (
        "undistort",
        "training views",
        "ai preprocessing",
        "ai scene masks",
        "ai depth assistance",
        "semantic masking",
        "depth anything",
    )):
        return 8
    if any(token in text for token in ("preparing registered", "dataset", "training data")):
        return 7
    if any(token in text for token in (
        "validating camera",
        "repairing camera",
        "camera repair",
        "camera rescue",
        "connectivity repair",
        "sparse repair",
        "exhaustive repair",
    )):
        return 6
    if any(token in text for token in ("solving cameras", "sparse reconstruction", "camera solve complete", "camera solve reused")):
        return 5
    if any(token in text for token in ("matching frames", "sequential matching")):
        return 4
    if any(token in text for token in ("extracting features", "feature extraction")):
        return 3
    if any(token in text for token in (
        "sampling video",
        "frames ready",
        "preparing photos",
        "photos ready",
        "source frames",
    )):
        return 2
    return 1

st.set_page_config(
    page_title="Splat Studio",
    page_icon="SplatStudio_icon.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def _hex_rgb(value):
    value = value.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, round(channel))):02X}" for channel in rgb)


def _mix_hex(colour_a, colour_b, amount):
    """Mix colour_a toward colour_b by amount (0..1)."""
    a = _hex_rgb(colour_a)
    b = _hex_rgb(colour_b)
    return _rgb_hex(tuple(a[i] + (b[i] - a[i]) * amount for i in range(3)))


def _relative_luminance(hex_colour):
    def linearise(channel):
        channel = channel / 255.0
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (_hex_rgb(hex_colour))
    r, g, b = linearise(r), linearise(g), linearise(b)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def render_theme_css():
    """Apply Dark, Light or Custom appearance and explicitly style Streamlit widgets."""
    accent = st.session_state.get("theme_accent", "#67D7FF")
    ui_scale = st.session_state.get("theme_scale", 100)
    console_size = st.session_state.get("theme_console_size", 12)
    reduced_motion = st.session_state.get("theme_reduce_motion", False)
    mode = st.session_state.get("theme_mode", "Dark")
    background_name = st.session_state.get("theme_background", "Midnight")
    custom_background = st.session_state.get("theme_custom_background", "#10141C")

    dark_backgrounds = {
        "Midnight": ("#07090D", "#0D1118", "#111722"),
        "Graphite": ("#0B0C0E", "#121417", "#181B20"),
        "Deep Blue": ("#070B13", "#0C1220", "#111B2E"),
    }

    light_backgrounds = {
        "Cloud": ("#F4F7FB", "#FFFFFF", "#F9FBFD"),
        "Paper": ("#F7F7F5", "#FFFFFF", "#FAFAF8"),
        "Warm": ("#F6F2EC", "#FFFDFC", "#FAF6F0"),
    }

    if mode == "Light":
        bg, panel, panel2 = light_backgrounds.get(background_name, light_backgrounds["Cloud"])
        text = "#172033"
        muted = "#667085"
        line = "#D8DEE8"
        line_strong = "#C8D0DC"
        control_bg = "#FFFFFF"
        control_hover = "#F2F5F9"
        control_active = "#E9EEF5"
        soft = "#00000005"
        soft_strong = "#0000000A"
        shadow = "#15223816"
        good_text = "#156B4A"
        error_text = "#A2293B"
        color_scheme = "light"
    elif mode == "Custom":
        bg = custom_background
        is_light = _relative_luminance(bg) > 0.43

        if is_light:
            panel = _mix_hex(bg, "#FFFFFF", 0.72)
            panel2 = _mix_hex(bg, "#FFFFFF", 0.48)
            text = "#172033"
            muted = "#626D7E"
            line = _mix_hex(bg, "#233047", 0.18)
            line_strong = _mix_hex(bg, "#233047", 0.28)
            control_bg = _mix_hex(bg, "#FFFFFF", 0.82)
            control_hover = _mix_hex(bg, "#FFFFFF", 0.60)
            control_active = _mix_hex(bg, "#233047", 0.08)
            soft = "#00000005"
            soft_strong = "#0000000A"
            shadow = "#15223816"
            good_text = "#156B4A"
            error_text = "#A2293B"
            color_scheme = "light"
        else:
            panel = _mix_hex(bg, "#FFFFFF", 0.055)
            panel2 = _mix_hex(bg, "#FFFFFF", 0.09)
            text = "#F4F7FB"
            muted = "#98A3B5"
            line = _mix_hex(bg, "#FFFFFF", 0.13)
            line_strong = _mix_hex(bg, "#FFFFFF", 0.20)
            control_bg = _mix_hex(bg, "#FFFFFF", 0.075)
            control_hover = _mix_hex(bg, "#FFFFFF", 0.12)
            control_active = _mix_hex(bg, "#FFFFFF", 0.16)
            soft = "#FFFFFF07"
            soft_strong = "#FFFFFF0E"
            shadow = "#0000002A"
            good_text = "#8EF0C6"
            error_text = "#FFD1D7"
            color_scheme = "dark"
    else:
        bg, panel, panel2 = dark_backgrounds.get(background_name, dark_backgrounds["Midnight"])
        text = "#F4F7FB"
        muted = "#8D98AA"
        line = "#FFFFFF14"
        line_strong = "#FFFFFF22"
        control_bg = "#12171F"
        control_hover = "#19212C"
        control_active = "#202A37"
        soft = "#FFFFFF07"
        soft_strong = "#FFFFFF0E"
        shadow = "#0000002A"
        good_text = "#8EF0C6"
        error_text = "#FFD1D7"
        color_scheme = "dark"

    # Keep button labels readable even if the user picks an unusually dark accent.
    primary_text = "#071018" if _relative_luminance(accent) > 0.38 else "#FFFFFF"
    motion_css = "*{animation:none!important;transition:none!important}" if reduced_motion else ""

    st.markdown(
        f"""
        <style>
        :root{{
            color-scheme:{color_scheme};
            --bg:{bg};
            --panel:{panel};
            --panel2:{panel2};
            --line:{line};
            --line-strong:{line_strong};
            --text:{text};
            --muted:{muted};
            --accent:{accent};

            /* Mirror the live Splat Studio palette into Streamlit's own
               internal theme tokens so BaseWeb widgets don't fall back to
               Streamlit's default red / dark palette in Custom mode. */
            --primary-color:{accent};
            --background-color:{bg};
            --secondary-background-color:{panel2};
            --text-color:{text};
            --border-color:{line_strong};

            --control-bg:{control_bg};
            --control-hover:{control_hover};
            --control-active:{control_active};
            --soft:{soft};
            --soft-strong:{soft_strong};
            --shadow:{shadow};
            --good:#56E0A5;
            --warn:#F4B860;
            --bad:#FF6678;
            --good-text:{good_text};
            --error-text:{error_text};
            --primary-text:{primary_text};
        }}

        html{{font-size:{ui_scale}%;background:var(--bg);color:var(--text)}}
        html,body,[class*="css"]{{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Inter","Segoe UI",sans-serif}}
        body,.stApp,[data-testid="stAppViewContainer"]{{color:var(--text)!important}}

        .stApp{{
            background:
                radial-gradient(circle at 84% -10%,color-mix(in srgb,var(--accent) 14%,transparent),transparent 30rem),
                radial-gradient(circle at 4% 4%,color-mix(in srgb,var(--accent) 8%,transparent),transparent 24rem),
                var(--bg);
            color:var(--text);
        }}

        /* Splat Studio owns its chrome. */
        [data-testid="stHeader"]{{background:transparent!important;height:0!important;min-height:0!important}}
        [data-testid="stToolbar"],
        [data-testid="stToolbarActions"],
        [data-testid="stDeployButton"],
        [data-testid="stStatusWidget"]{{display:none!important;visibility:hidden!important}}
        #MainMenu,footer{{visibility:hidden}}
        [data-testid="stSidebar"]{{display:none!important}}
        .block-container{{max-width:1440px;padding-top:1.35rem;padding-bottom:3rem}}

        /* General text: do not let Streamlit's base theme leak through. */
        .stApp p,.stApp label,.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
        [data-testid="stMarkdownContainer"]{{color:var(--text)}}
        .stCaptionContainer,[data-testid="stCaptionContainer"],small{{color:var(--muted)!important}}

        .app-shell{{animation:pageIn .28s ease-out}}
        @keyframes pageIn{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1;transform:translateY(0)}}}}
        .app-top{{display:flex;align-items:center;justify-content:space-between;gap:1rem;margin-bottom:.45rem}}
        .brand{{font-size:1.18rem;font-weight:760;letter-spacing:-.035em;color:var(--text)}}
        .brand-sub{{font-size:.68rem;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin-top:.12rem}}

        .studio-brand-title{{
            font-size:2.72rem;
            line-height:1;
            font-weight:790;
            letter-spacing:-.055em;
            color:var(--text);
            margin:0 0 .28rem;
        }}
        .studio-brand-sub{{
            color:var(--muted);
            font-size:.90rem;
            font-weight:680;
            text-transform:uppercase;
            letter-spacing:.16em;
            margin:0;
        }}
        .studio-brand-meta{{
            display:flex;
            align-items:center;
            gap:.42rem;
            flex-wrap:wrap;
        }}
        .studio-brand-tag{{
            display:inline-flex;
            align-items:center;
            border:1px solid var(--line);
            background:var(--soft);
            border-radius:999px;
            padding:.22rem .46rem;
            color:var(--muted);
            font-size:.72rem;
            font-weight:720;
            letter-spacing:.04em;
            white-space:nowrap;
        }}
        .studio-local{{
            display:inline-flex;
            align-items:center;
            gap:.35rem;
            color:var(--good-text);
            font-size:.76rem;
            font-weight:720;
            white-space:nowrap;
        }}
        .studio-local:before{{
            content:"";
            width:.4rem;
            height:.4rem;
            border-radius:50%;
            background:var(--good);
            box-shadow:0 0 10px color-mix(in srgb,var(--good) 55%,transparent);
        }}
        .header-status-row{{
            display:flex;
            align-items:center;
            justify-content:flex-end;
            gap:.42rem;
            flex-wrap:wrap;
            margin:0 0 .48rem;
            min-height:1.65rem;
        }}
        .checkpoint-chip{{
            display:inline-flex;
            align-items:center;
            gap:.38rem;
            border-radius:999px;
            padding:.22rem .48rem;
            font-size:.73rem;
            font-weight:740;
            white-space:nowrap;
            border:1px solid color-mix(in srgb,var(--good) 34%,var(--line));
            background:color-mix(in srgb,var(--good) 9%,var(--panel));
            color:var(--good-text);
        }}
        .checkpoint-chip:before{{
            content:"";
            width:.42rem;
            height:.42rem;
            border-radius:50%;
            background:var(--good);
            box-shadow:0 0 10px color-mix(in srgb,var(--good) 62%,transparent);
            flex:none;
        }}
        .checkpoint-chip.neutral{{
            border-color:var(--line);
            background:var(--soft);
            color:var(--muted);
        }}
        .checkpoint-chip.neutral:before{{
            background:var(--muted);
            box-shadow:none;
            opacity:.65;
        }}
        .checkpoint-chip.finished{{
            border-color:color-mix(in srgb,var(--accent) 38%,var(--line));
            background:color-mix(in srgb,var(--accent) 9%,var(--panel));
            color:var(--text);
        }}
        .checkpoint-chip.finished:before{{
            background:var(--accent);
            box-shadow:0 0 10px color-mix(in srgb,var(--accent) 55%,transparent);
        }}
        .checkpoint-chip.warning{{
            border-color:color-mix(in srgb,#FFB84D 55%,var(--line));
            background:color-mix(in srgb,#FFB84D 12%,var(--panel));
            color:var(--text);
        }}
        .checkpoint-chip.warning:before{{
            background:#FFB84D;
            box-shadow:0 0 10px color-mix(in srgb,#FFB84D 60%,transparent);
        }}
        .eyebrow{{font-size:.68rem;text-transform:uppercase;letter-spacing:.16em;font-weight:760;color:var(--accent);margin-bottom:.45rem}}
        .page-title{{font-size:clamp(2rem,4vw,3.15rem);letter-spacing:-.055em;line-height:1.02;font-weight:760;margin:.1rem 0 .65rem;color:var(--text)}}
        .page-sub{{font-size:.96rem;color:var(--muted);max-width:760px;line-height:1.55;margin-bottom:1.25rem}}
        .status-pill{{display:inline-flex;align-items:center;gap:.45rem;border:1px solid var(--line);border-radius:999px;padding:.36rem .62rem;background:var(--soft);color:var(--text);font-size:.7rem;white-space:nowrap}}
        .status-dot{{height:.45rem;width:.45rem;border-radius:50%;background:var(--good);box-shadow:0 0 12px color-mix(in srgb,var(--good) 55%,transparent)}}
        .wizard{{display:grid;grid-template-columns:repeat(5,1fr);gap:.55rem;margin:1.15rem 0 1.55rem}}
        .wiz-step{{position:relative;padding:.72rem .62rem;border-top:3px solid var(--line);color:var(--muted);font-size:.88rem;font-weight:620;letter-spacing:.025em}}
        .wiz-step.active{{border-color:var(--accent);color:var(--text)}}
        .wiz-step.done{{border-color:color-mix(in srgb,var(--accent) 55%,transparent);color:var(--muted)}}
        .wiz-num{{display:block;font-size:.69rem;font-weight:700;color:var(--muted);margin-bottom:.22rem}}

        div[data-testid="stVerticalBlockBorderWrapper"]{{
            border-color:var(--line)!important;
            background:linear-gradient(180deg,var(--panel2),var(--panel));
            box-shadow:0 18px 44px var(--shadow);
            border-radius:14px!important;
        }}

        .card-title{{font-size:1.1rem;font-weight:720;letter-spacing:-.025em;margin:.15rem 0 .25rem;color:var(--text)}}
        .card-sub{{font-size:.76rem;color:var(--muted);margin-bottom:.75rem}}
        .file-strip{{display:flex;justify-content:space-between;gap:.8rem;align-items:center;border:1px solid var(--line);border-radius:11px;background:var(--soft);padding:.72rem .82rem;margin:.5rem 0}}
        .file-name{{font-weight:680;font-size:.83rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--text)}}
        .file-meta{{color:var(--muted);font-size:.68rem;margin-top:.16rem}}
        .ready{{border:1px solid color-mix(in srgb,var(--good) 30%,transparent);background:color-mix(in srgb,var(--good) 9%,transparent);color:var(--good-text);border-radius:999px;padding:.26rem .48rem;font-size:.65rem}}
        .tip{{border-left:2px solid var(--accent);background:var(--soft);border-radius:0 10px 10px 0;padding:.7rem .85rem;color:var(--muted);font-size:.76rem;line-height:1.55}}
        .summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin:.65rem 0}}
        .summary-box{{border:1px solid var(--line);background:var(--soft);border-radius:10px;padding:.7rem .75rem}}
        .summary-k{{font-size:.59rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}}
        .summary-v{{font-size:.9rem;font-weight:680;margin-top:.18rem;color:var(--text)}}
        .project-state{{border:1px solid var(--line);background:var(--soft);border-radius:11px;padding:.72rem .82rem;margin:.55rem 0 .72rem}}
        .project-state.good{{border-color:color-mix(in srgb,var(--good) 34%,var(--line))}}
        .project-state.warn{{border-color:color-mix(in srgb,var(--warn) 42%,var(--line))}}
        .project-state-title{{font-size:.78rem;font-weight:720;color:var(--text)}}
        .project-state-sub{{font-size:.68rem;color:var(--muted);margin-top:.18rem;line-height:1.45}}
        .resource-card{{border:1px solid var(--line);background:linear-gradient(180deg,var(--panel2),var(--panel));border-radius:13px;padding:.85rem .9rem;margin:.75rem 0}}
        .resource-top{{display:flex;justify-content:space-between;gap:1rem;align-items:center;margin-bottom:.55rem}}
        .resource-title{{font-size:.9rem;font-weight:730;color:var(--text)}}
        .resource-grade{{font-size:.64rem;font-weight:760;text-transform:uppercase;letter-spacing:.09em;border-radius:999px;padding:.28rem .48rem}}
        .resource-grade.good{{color:var(--good-text);background:color-mix(in srgb,var(--good) 11%,transparent);border:1px solid color-mix(in srgb,var(--good) 30%,transparent)}}
        .resource-grade.warn{{color:var(--warn);background:color-mix(in srgb,var(--warn) 10%,transparent);border:1px solid color-mix(in srgb,var(--warn) 32%,transparent)}}
        .resource-grade.bad{{color:var(--bad);background:color-mix(in srgb,var(--bad) 10%,transparent);border:1px solid color-mix(in srgb,var(--bad) 34%,transparent)}}
        .resource-meter{{height:7px;background:var(--soft-strong);border-radius:999px;overflow:hidden;margin:.45rem 0 .55rem}}
        .resource-fill{{height:100%;border-radius:999px}}
        .resource-fill.good{{background:var(--good)}}
        .resource-fill.warn{{background:var(--warn)}}
        .resource-fill.bad{{background:var(--bad)}}
        .resource-note{{font-size:.68rem;color:var(--muted);line-height:1.45}}
        .tip-carousel{{
            position:relative;
            min-height:3.05rem;
            border:1px solid color-mix(in srgb,#42D9FF 36%,var(--line));
            background:color-mix(in srgb,#42D9FF 9%,var(--panel));
            border-radius:11px;
            margin:.72rem 0 .9rem;
            overflow:hidden;
            box-shadow:inset 3px 0 0 color-mix(in srgb,#42D9FF 72%,transparent);
        }}
        .tip-carousel:before{{
            content:"TIP";
            position:absolute;
            left:.78rem;
            top:.52rem;
            color:#42D9FF;
            font-size:.58rem;
            font-weight:800;
            letter-spacing:.12em;
            z-index:2;
        }}
        .tip-carousel span{{
            position:absolute;
            left:3.25rem;
            right:.8rem;
            top:.48rem;
            opacity:0;
            color:var(--text);
            font-size:.72rem;
            line-height:1.5;
            animation:studioTipCycle 28s infinite;
        }}
        .tip-carousel span:nth-child(1){{animation-delay:0s}}
        .tip-carousel span:nth-child(2){{animation-delay:7s}}
        .tip-carousel span:nth-child(3){{animation-delay:14s}}
        .tip-carousel span:nth-child(4){{animation-delay:21s}}
        @keyframes studioTipCycle{{
            0%{{opacity:0;transform:translateY(4px)}}
            5%,22%{{opacity:1;transform:translateY(0)}}
            25%,100%{{opacity:0;transform:translateY(-4px)}}
        }}

        .progress-shell{{padding:.15rem 0 .25rem}}
        .progress-meta{{display:flex;justify-content:space-between;gap:1rem;margin-bottom:.5rem}}
        .progress-label{{font-size:.88rem;font-weight:690;color:var(--text)}}
        .progress-value{{font-size:.76rem;color:var(--muted);font-variant-numeric:tabular-nums}}
        .progress-value-large{{font-size:1.38rem;font-weight:790;letter-spacing:-.035em;color:var(--text);font-variant-numeric:tabular-nums;line-height:1}}
        .progress-kicker{{font-size:.60rem;font-weight:720;letter-spacing:.10em;text-transform:uppercase;color:var(--muted);margin-bottom:.22rem}}
        .progress-section{{margin:.35rem 0 .85rem}}
        .progress-track.stage-track{{height:5px;opacity:.82}}
        .progress-track{{height:8px;border-radius:999px;overflow:hidden;background:var(--soft-strong);position:relative}}
        .progress-fill{{height:100%;border-radius:inherit;background:linear-gradient(90deg,var(--accent),color-mix(in srgb,var(--accent) 62%,#A879FF));box-shadow:0 0 18px color-mix(in srgb,var(--accent) 35%,transparent);transition:width .22s ease;position:relative}}
        .progress-fill:after{{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 25%,#ffffff42 45%,transparent 65%);transform:translateX(-100%);animation:shimmer 2.6s infinite}}
        @keyframes shimmer{{to{{transform:translateX(160%)}}}}
        .progress-detail{{margin-top:.48rem;color:var(--muted);font-size:.72rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
        .run-live{{display:flex;align-items:center;gap:.75rem;border:1px solid var(--line);background:var(--soft);border-radius:12px;padding:.62rem .78rem;min-height:48px}}
        .run-spinner{{width:18px;height:18px;border-radius:50%;border:2px solid color-mix(in srgb,var(--accent) 20%,transparent);border-top-color:var(--accent);animation:runSpin .9s linear infinite;flex:0 0 auto}}
        @keyframes runSpin{{to{{transform:rotate(360deg)}}}}
        .run-live-title{{font-size:.76rem;font-weight:720;color:var(--text);line-height:1.15}}
        .run-live-sub{{font-size:.64rem;color:var(--muted);margin-top:.18rem}}
        .st-key-run_stop_button button{{background:color-mix(in srgb,var(--bad) 9%,transparent)!important;color:var(--bad)!important;border:1px solid color-mix(in srgb,var(--bad) 55%,transparent)!important;box-shadow:none!important}}
        .st-key-run_stop_button button *{{color:var(--bad)!important;-webkit-text-fill-color:var(--bad)!important;fill:currentColor!important}}
        .st-key-run_stop_button button:hover{{background:color-mix(in srgb,var(--bad) 17%,transparent)!important;border-color:var(--bad)!important}}
        .review-guide{{border:1px solid var(--line);background:linear-gradient(135deg,var(--panel2),var(--panel));border-radius:13px;padding:.8rem .9rem;margin:.2rem 0 .8rem}}
        .review-guide strong{{color:var(--text)}}
        .review-guide span{{color:var(--muted);font-size:.75rem;line-height:1.5}}
        .health-good{{border-left:3px solid var(--good)}}
        .health-warn{{border-left:3px solid var(--warn)}}
        .health-bad{{border-left:3px solid var(--bad)}}
        .review-hint{{font-size:.7rem;color:var(--muted);margin:.15rem 0 .55rem}}
        .output-path{{font-family:"SFMono-Regular",Consolas,monospace;font-size:.69rem;color:var(--text);border:1px solid var(--line);background:var(--soft);border-radius:9px;padding:.58rem .65rem;word-break:break-all}}
        .success-banner{{border:1px solid color-mix(in srgb,var(--good) 28%,transparent);background:color-mix(in srgb,var(--good) 8%,transparent);border-radius:13px;padding:.8rem 1rem;color:var(--good-text)}}
        .error-banner{{border:1px solid color-mix(in srgb,var(--bad) 30%,transparent);background:color-mix(in srgb,var(--bad) 8%,transparent);border-radius:13px;padding:.8rem 1rem;color:var(--error-text)}}

        /* ---------------------------------------------------------------
           BUTTON FIX
           Streamlit's base theme was providing white secondary buttons,
           while this app was providing light text. Explicitly own both.
           --------------------------------------------------------------- */
        .stButton>button,
        .stDownloadButton>button,
        .stLinkButton>a{{
            border-radius:10px!important;
            font-weight:650!important;
            transition:background .16s ease,border-color .16s ease,transform .16s ease,box-shadow .16s ease!important;
        }}

        .stButton>button:not([kind="primary"]),
        .stDownloadButton>button,
        .stLinkButton>a{{
            background:var(--control-bg)!important;
            color:var(--text)!important;
            border:1px solid var(--line-strong)!important;
            box-shadow:none!important;
        }}

        .stButton>button:not([kind="primary"]) *,
        .stDownloadButton>button *,
        .stLinkButton>a *{{
            color:var(--text)!important;
            fill:currentColor!important;
        }}

        .stButton>button:not([kind="primary"]):hover,
        .stDownloadButton>button:hover,
        .stLinkButton>a:hover{{
            background:var(--control-hover)!important;
            border-color:color-mix(in srgb,var(--accent) 48%,var(--line-strong))!important;
            color:var(--text)!important;
        }}

        .stButton>button:not([kind="primary"]):active,
        .stDownloadButton>button:active,
        .stLinkButton>a:active{{
            background:var(--control-active)!important;
            transform:translateY(1px);
        }}

        .stButton>button[kind="primary"]{{
            border:0!important;
            border-radius:10px!important;
            min-height:2.75rem;
            background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 56%,#8874FF))!important;
            color:var(--primary-text)!important;
            font-weight:760!important;
            box-shadow:0 12px 26px color-mix(in srgb,var(--accent) 20%,transparent)!important;
        }}

        .stButton>button[kind="primary"] *,
        .stButton>button[kind="primary"] p,
        .stButton>button[kind="primary"] span{{
            color:var(--primary-text)!important;
            fill:currentColor!important;
        }}

        .stButton>button:disabled,
        .stDownloadButton>button:disabled{{
            opacity:.46!important;
            cursor:not-allowed!important;
        }}

        /* Inputs / selectboxes / number fields. */
        [data-baseweb="input"],
        [data-baseweb="base-input"],
        [data-baseweb="select"]>div,
        [data-testid="stTextInputRootElement"],
        [data-testid="stNumberInputContainer"]{{
            background:var(--control-bg)!important;
            color:var(--text)!important;
            border-color:var(--line-strong)!important;
        }}

        [data-baseweb="input"] input,
        [data-baseweb="base-input"] input,
        [data-testid="stTextInputRootElement"] input,
        [data-testid="stNumberInputContainer"] input{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            caret-color:var(--accent)!important;
        }}

        [data-baseweb="select"] *,
        [data-baseweb="popover"] *,
        [role="listbox"] *{{
            color:var(--text)!important;
        }}

        [data-baseweb="popover"],
        [role="listbox"]{{
            background:var(--panel2)!important;
            border-color:var(--line)!important;
        }}

        /* File upload is one of the most obvious places the Streamlit base
           theme can otherwise stay white when the app is dark. */
        [data-testid="stFileUploaderDropzone"]{{
            background:var(--control-bg)!important;
            border-color:var(--line-strong)!important;
            color:var(--text)!important;
        }}
        [data-testid="stFileUploaderDropzone"] *{{
            color:var(--text)!important;
        }}

        /* Expander/popover/dialog surfaces. */
        [data-testid="stExpander"],
        [data-testid="stPopoverBody"],
        [data-testid="stDialog"]>div{{
            background:var(--panel)!important;
            color:var(--text)!important;
            border-color:var(--line)!important;
        }}

        [data-testid="stCode"] code{{font-family:"SFMono-Regular",Consolas,monospace!important;font-size:{console_size}px!important}}

        /* Run + Review metrics. Streamlit otherwise keeps its own light card
           surface, which is what caused the white boxes in Dark mode. */
        [data-testid="stMetric"]{{
            background:var(--panel2)!important;
            color:var(--text)!important;
            border:1px solid var(--line)!important;
            border-radius:12px!important;
            padding:.72rem .82rem!important;
            min-height:88px;
            box-shadow:none!important;
        }}
        [data-testid="stMetric"] *,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *{{
            color:var(--muted)!important;
            -webkit-text-fill-color:var(--muted)!important;
        }}

        /* Keep messages, status cards and debug output inside the same theme. */
        [data-testid="stAlert"],
        [data-testid="stNotification"]{{
            background:var(--panel2)!important;
            color:var(--text)!important;
            border-color:var(--line)!important;
        }}
        [data-testid="stAlert"] *,
        [data-testid="stNotification"] *{{
            color:var(--text)!important;
        }}
        [data-testid="stCode"],
        [data-testid="stJson"]{{
            background:var(--panel)!important;
            color:var(--text)!important;
            border-color:var(--line)!important;
        }}
        [data-testid="stProgress"] > div > div{{
            background:var(--soft-strong)!important;
        }}

        /* Segmented controls: style every BaseWeb layer, not only <button>.
           Streamlit may render each segment as a button or role=radio with one
           or more nested wrappers that otherwise inherit the default dark UI. */
        html body .stApp [data-baseweb="button-group"],
        html body .stApp [data-baseweb="button-group"] > div,
        html body .stApp [role="radiogroup"]{{
            background:transparent!important;
            background-color:transparent!important;
            border-color:var(--line-strong)!important;
            box-shadow:none!important;
        }}

        html body .stApp [data-baseweb="button-group"] button,
        html body .stApp [data-baseweb="button-group"] [role="radio"],
        html body .stApp [role="radiogroup"] button,
        html body .stApp [role="radiogroup"] [role="radio"]{{
            background:var(--control-bg)!important;
            background-color:var(--control-bg)!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            border-color:var(--line-strong)!important;
            outline-color:var(--line-strong)!important;
            box-shadow:none!important;
        }}

        html body .stApp [data-baseweb="button-group"] button > div,
        html body .stApp [data-baseweb="button-group"] [role="radio"] > div,
        html body .stApp [role="radiogroup"] button > div,
        html body .stApp [role="radiogroup"] [role="radio"] > div{{
            background:transparent!important;
            background-color:transparent!important;
            color:inherit!important;
        }}

        html body .stApp [data-baseweb="button-group"] button *,
        html body .stApp [data-baseweb="button-group"] [role="radio"] *,
        html body .stApp [role="radiogroup"] button *,
        html body .stApp [role="radiogroup"] [role="radio"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}

        html body .stApp [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body .stApp [data-baseweb="button-group"] button[aria-pressed="true"],
        html body .stApp [data-baseweb="button-group"] button[aria-checked="true"],
        html body .stApp [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body .stApp [role="radiogroup"] button[aria-pressed="true"],
        html body .stApp [role="radiogroup"] [role="radio"][aria-checked="true"]{{
            background:color-mix(in srgb,var(--accent) 16%,var(--panel2))!important;
            background-color:color-mix(in srgb,var(--accent) 16%,var(--panel2))!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            border-color:var(--accent)!important;
            outline-color:var(--accent)!important;
            box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 36%,transparent)!important;
        }}

        html body .stApp [data-baseweb="button-group"] button:not([kind="segmented_controlActive"]):hover,
        html body .stApp [data-baseweb="button-group"] [role="radio"]:not([aria-checked="true"]):hover,
        html body .stApp [role="radiogroup"] [role="radio"]:not([aria-checked="true"]):hover{{
            background:var(--control-hover)!important;
            background-color:var(--control-hover)!important;
            color:var(--text)!important;
            border-color:color-mix(in srgb,var(--accent) 48%,var(--line-strong))!important;
        }}

        /* Hard overrides for the two segmented controls the user sees most.
           These scoped selectors intentionally outrank Streamlit's built-in
           component theme styles in Custom mode. */
        /* Theme and Source selectors share the same exact visual treatment.
           IMPORTANT: theme_mode is the real Streamlit widget key; targeting
           only the surrounding theme_segment_control container misses the
           inner BaseWeb component in current Streamlit versions. */
        html body .stApp .st-key-theme_mode [data-baseweb="button-group"],
        html body .stApp .st-key-theme_mode [role="radiogroup"],
        html body .stApp .st-key-source_mode [data-baseweb="button-group"],
        html body .stApp .st-key-source_mode [role="radiogroup"]{{
            background:transparent!important;
            background-color:transparent!important;
        }}

        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] button,
        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] [role="radio"],
        html body .stApp .st-key-theme_mode [role="radiogroup"] button,
        html body .stApp .st-key-theme_mode [role="radiogroup"] [role="radio"],
        html body .stApp .st-key-source_mode [data-baseweb="button-group"] button,
        html body .stApp .st-key-source_mode [data-baseweb="button-group"] [role="radio"],
        html body .stApp .st-key-source_mode [role="radiogroup"] button,
        html body .stApp .st-key-source_mode [role="radiogroup"] [role="radio"]{{
            background:var(--control-bg)!important;
            background-color:var(--control-bg)!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            border-color:var(--line-strong)!important;
            outline-color:var(--line-strong)!important;
            box-shadow:none!important;
        }}

        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] button > div,
        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] [role="radio"] > div,
        html body .stApp .st-key-theme_mode [role="radiogroup"] button > div,
        html body .stApp .st-key-theme_mode [role="radiogroup"] [role="radio"] > div{{
            background:transparent!important;
            background-color:transparent!important;
            color:inherit!important;
        }}

        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] button *,
        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] [role="radio"] *,
        html body .stApp .st-key-theme_mode [role="radiogroup"] button *,
        html body .stApp .st-key-theme_mode [role="radiogroup"] [role="radio"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}

        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] button[aria-pressed="true"],
        html body .stApp .st-key-theme_mode [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body .stApp .st-key-theme_mode [role="radiogroup"] [role="radio"][aria-checked="true"],
        html body .stApp .st-key-source_mode [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body .stApp .st-key-source_mode [data-baseweb="button-group"] button[aria-pressed="true"],
        html body .stApp .st-key-source_mode [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body .stApp .st-key-source_mode [role="radiogroup"] [role="radio"][aria-checked="true"]{{
            background:color-mix(in srgb,var(--accent) 16%,var(--panel2))!important;
            background-color:color-mix(in srgb,var(--accent) 16%,var(--panel2))!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            border-color:var(--accent)!important;
            outline-color:var(--accent)!important;
            box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 34%,transparent)!important;
        }}

        /* Keep the legacy selectors as a fallback for older Streamlit builds. */
        html body .stApp [data-testid="stSegmentedControl"] button,
        html body .stApp [data-testid="stSegmentedControl"] [role="radio"]{{
            background:var(--control-bg)!important;
            color:var(--text)!important;
            border-color:var(--line-strong)!important;
        }}

        /* Current Streamlit base-button test IDs. This supplements .stButton
           selectors because Streamlit may change the wrapper DOM between releases. */
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-tertiary"]{{
            background:var(--control-bg)!important;
            color:var(--text)!important;
            border:1px solid var(--line-strong)!important;
        }}

        button[data-testid="stBaseButton-secondary"] *,
        button[data-testid="stBaseButton-tertiary"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}

        button[data-testid="stBaseButton-primary"]{{
            color:var(--primary-text)!important;
        }}
        button[data-testid="stBaseButton-primary"] *{{
            color:var(--primary-text)!important;
            -webkit-text-fill-color:var(--primary-text)!important;
        }}

        /* Popover triggers (Settings / Close) and other secondary buttons. */
        button[kind="secondary"],
        button[kind="tertiary"],
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-tertiary"],
        [data-testid*="Popover"] > button,
        [data-testid*="popover"] > button{{
            background:var(--control-bg)!important;
            color:var(--text)!important;
            border:1px solid var(--line-strong)!important;
            box-shadow:none!important;
        }}

        button[kind="secondary"] *,
        button[kind="tertiary"] *,
        button[data-testid="stBaseButton-secondary"] *,
        button[data-testid="stBaseButton-tertiary"] *,
        [data-testid*="Popover"] > button *,
        [data-testid*="popover"] > button *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            fill:currentColor!important;
        }}

        /* The recovery action is deliberately green: it means saved work exists. */
        .st-key-failure_continue_checkpoint button,
        .st-key-failure_continue_checkpoint button[kind="primary"]{{
            background:linear-gradient(135deg,#2CCB81,#159B61)!important;
            color:#06150E!important;
            border:1px solid #55E7A6!important;
            box-shadow:0 10px 24px #1DBB7530!important;
        }}
        .st-key-failure_continue_checkpoint button *,
        .st-key-failure_continue_checkpoint button[kind="primary"] *{{
            color:#06150E!important;
            -webkit-text-fill-color:#06150E!important;
            fill:currentColor!important;
        }}
        .st-key-failure_continue_checkpoint button:hover{{
            background:linear-gradient(135deg,#43DC95,#1CAE6C)!important;
            border-color:#7CF1BD!important;
        }}

        button[kind="secondary"]:hover,
        button[kind="tertiary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="stBaseButton-tertiary"]:hover,
        [data-testid*="Popover"] > button:hover,
        [data-testid*="popover"] > button:hover{{
            background:var(--control-hover)!important;
            color:var(--text)!important;
            border-color:color-mix(in srgb,var(--accent) 48%,var(--line-strong))!important;
        }}

        /* Help icons and their tooltip bubbles.
           Escaped braces are required because this CSS lives inside a Python f-string. */
        html body .stApp [data-testid="stTooltipIcon"],
        html body .stApp [data-testid="stTooltipHoverTarget"],
        html body .stApp button[aria-label*="help" i],
        html body .stApp button[aria-label*="tooltip" i]{{
            color:color-mix(in srgb,var(--text) 84%,var(--accent))!important;
            opacity:.82!important;
            background:transparent!important;
            background-color:transparent!important;
            box-shadow:none!important;
        }}

        html body .stApp [data-testid="stTooltipIcon"]:hover,
        html body .stApp [data-testid="stTooltipHoverTarget"]:hover,
        html body .stApp button[aria-label*="help" i]:hover,
        html body .stApp button[aria-label*="tooltip" i]:hover{{
            color:var(--accent)!important;
            opacity:1!important;
        }}

        html body .stApp [data-testid="stTooltipIcon"] svg,
        html body .stApp [data-testid="stTooltipHoverTarget"] svg,
        html body .stApp button[aria-label*="help" i] svg,
        html body .stApp button[aria-label*="tooltip" i] svg{{
            color:inherit!important;
            fill:none!important;
            stroke:currentColor!important;
            stroke-width:2.25!important;
            opacity:1!important;
        }}

        html body .stApp [data-testid="stTooltipIcon"] svg *,
        html body .stApp [data-testid="stTooltipHoverTarget"] svg *,
        html body .stApp button[aria-label*="help" i] svg *,
        html body .stApp button[aria-label*="tooltip" i] svg *{{
            fill:none!important;
            stroke:currentColor!important;
            opacity:1!important;
        }}
        html body div[role="tooltip"],
        html body [data-baseweb="tooltip"],
        html body [data-testid="stTooltipContent"]{{
            background:var(--panel2)!important;
            background-color:var(--panel2)!important;
            color:var(--text)!important;
            border:1px solid var(--line-strong)!important;
            border-radius:10px!important;
            box-shadow:0 10px 28px color-mix(in srgb,#000 22%,transparent)!important;
        }}
        html body div[role="tooltip"] *,
        html body [data-baseweb="tooltip"] *,
        html body [data-testid="stTooltipContent"] *{{
            background-color:transparent!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}
        html body div[role="tooltip"]::before,
        html body div[role="tooltip"]::after,
        html body [data-baseweb="tooltip"]::before,
        html body [data-baseweb="tooltip"]::after{{
            border-color:var(--panel2)!important;
        }}

        /* Toggle switches: BaseWeb otherwise keeps a near-black track in
           custom/light themes. The first child is the slider track and its
           child is the moving thumb in Streamlit's checkbox/toggle widget. */
        html body .stApp [data-baseweb="checkbox"] > div:first-child{{
            background:var(--control-bg)!important;
            background-color:var(--control-bg)!important;
            border:1px solid var(--line-strong)!important;
            box-shadow:none!important;
        }}
        html body .stApp [data-baseweb="checkbox"]:has(input:checked) > div:first-child{{
            background:var(--accent)!important;
            background-color:var(--accent)!important;
            border-color:color-mix(in srgb,var(--accent) 72%,var(--line-strong))!important;
        }}
        html body .stApp [data-baseweb="checkbox"] > div:first-child > div{{
            background:var(--panel2)!important;
            background-color:var(--panel2)!important;
            border:1px solid color-mix(in srgb,var(--text) 14%,transparent)!important;
            box-shadow:0 1px 4px color-mix(in srgb,#000 18%,transparent)!important;
        }}
        html body .stApp [data-baseweb="checkbox"] [data-testid="stWidgetLabel"],
        html body .stApp [data-baseweb="checkbox"] [data-testid="stWidgetLabel"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}
        html body .stApp [role="switch"][aria-checked="false"]{{
            background:var(--control-bg)!important;
            border-color:var(--line-strong)!important;
        }}
        html body .stApp [role="switch"][aria-checked="true"]{{
            background:var(--accent)!important;
            border-color:var(--accent)!important;
        }}

        /* Final segmented-control guard. Keep this late in the stylesheet
           so Streamlit's default primary/red border cannot win by source order. */
        html body .stApp [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body .stApp [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body .stApp [role="radiogroup"] [role="radio"][aria-checked="true"]{{
            border-color:var(--accent)!important;
            outline-color:var(--accent)!important;
            box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 36%,transparent)!important;
        }}

        /* ===============================================================
           SETTINGS POPOVER SEGMENTED CONTROLS
           BaseWeb renders popovers in a portal outside .stApp. Therefore
           these selectors intentionally DO NOT require a .stApp ancestor.
           =============================================================== */

        html body [data-baseweb="popover"] [data-baseweb="button-group"],
        html body [data-baseweb="popover"] [role="radiogroup"],
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"],
        html body [data-testid="stPopoverBody"] [role="radiogroup"],
        html body .st-key-theme_segment_control [data-baseweb="button-group"],
        html body .st-key-theme_segment_control [role="radiogroup"]{{
            background:transparent!important;
            background-color:transparent!important;
            border-color:var(--line-strong)!important;
            box-shadow:none!important;
        }}

        html body [data-baseweb="popover"] [data-baseweb="button-group"] > div,
        html body [data-baseweb="popover"] [role="radiogroup"] > div,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] > div,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] > div,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] > div,
        html body .st-key-theme_segment_control [role="radiogroup"] > div{{
            background:transparent!important;
            background-color:transparent!important;
        }}

        html body [data-baseweb="popover"] [data-baseweb="button-group"] button,
        html body [data-baseweb="popover"] [data-baseweb="button-group"] [role="radio"],
        html body [data-baseweb="popover"] [role="radiogroup"] button,
        html body [data-baseweb="popover"] [role="radiogroup"] [role="radio"],
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] button,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] [role="radio"],
        html body [data-testid="stPopoverBody"] [role="radiogroup"] button,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] [role="radio"],
        html body .st-key-theme_segment_control [data-baseweb="button-group"] button,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] [role="radio"],
        html body .st-key-theme_segment_control [role="radiogroup"] button,
        html body .st-key-theme_segment_control [role="radiogroup"] [role="radio"]{{
            background:var(--control-bg)!important;
            background-color:var(--control-bg)!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            border-color:var(--line-strong)!important;
            outline-color:var(--line-strong)!important;
            box-shadow:none!important;
            opacity:1!important;
        }}

        html body [data-baseweb="popover"] [data-baseweb="button-group"] button > div,
        html body [data-baseweb="popover"] [data-baseweb="button-group"] [role="radio"] > div,
        html body [data-baseweb="popover"] [role="radiogroup"] button > div,
        html body [data-baseweb="popover"] [role="radiogroup"] [role="radio"] > div,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] button > div,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] [role="radio"] > div,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] button > div,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] [role="radio"] > div,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] button > div,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] [role="radio"] > div{{
            background:transparent!important;
            background-color:transparent!important;
            color:inherit!important;
        }}

        html body [data-baseweb="popover"] [data-baseweb="button-group"] button *,
        html body [data-baseweb="popover"] [data-baseweb="button-group"] [role="radio"] *,
        html body [data-baseweb="popover"] [role="radiogroup"] button *,
        html body [data-baseweb="popover"] [role="radiogroup"] [role="radio"] *,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] button *,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] [role="radio"] *,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] button *,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] [role="radio"] *,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] button *,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] [role="radio"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}

        html body [data-baseweb="popover"] [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body [data-baseweb="popover"] [data-baseweb="button-group"] button[aria-pressed="true"],
        html body [data-baseweb="popover"] [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body [data-baseweb="popover"] [role="radiogroup"] [role="radio"][aria-checked="true"],
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] button[aria-pressed="true"],
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body [data-testid="stPopoverBody"] [role="radiogroup"] [role="radio"][aria-checked="true"],
        html body .st-key-theme_segment_control [data-baseweb="button-group"] button[kind="segmented_controlActive"],
        html body .st-key-theme_segment_control [data-baseweb="button-group"] button[aria-pressed="true"],
        html body .st-key-theme_segment_control [data-baseweb="button-group"] [role="radio"][aria-checked="true"],
        html body .st-key-theme_segment_control [role="radiogroup"] [role="radio"][aria-checked="true"]{{
            background:color-mix(in srgb,var(--accent) 16%,var(--panel2))!important;
            background-color:color-mix(in srgb,var(--accent) 16%,var(--panel2))!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            border-color:var(--accent)!important;
            outline-color:var(--accent)!important;
            box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 34%,transparent)!important;
            opacity:1!important;
        }}

        html body [data-baseweb="popover"] [data-baseweb="button-group"] button:hover,
        html body [data-baseweb="popover"] [data-baseweb="button-group"] [role="radio"]:hover,
        html body [data-baseweb="popover"] [role="radiogroup"] [role="radio"]:hover,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] button:hover,
        html body [data-testid="stPopoverBody"] [data-baseweb="button-group"] [role="radio"]:hover,
        html body [data-testid="stPopoverBody"] [role="radiogroup"] [role="radio"]:hover,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] button:hover,
        html body .st-key-theme_segment_control [data-baseweb="button-group"] [role="radio"]:hover{{
            background:var(--control-hover)!important;
            background-color:var(--control-hover)!important;
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}

        /* ===============================================================
           FINAL THEME CONSISTENCY PASS
           Streamlit has several button/surface DOM variants depending on
           widget and release. These intentionally high-specificity rules are
           last so no Streamlit base theme can turn controls white in Dark mode.
           =============================================================== */
        html body .stApp div[data-testid="stButton"] > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]),
        html body .stApp .stButton > button:not([kind="primary"]):not([data-testid="stBaseButton-primary"]),
        html body .stApp button[kind="secondary"],
        html body .stApp button[kind="tertiary"],
        html body .stApp button[data-testid="stBaseButton-secondary"],
        html body .stApp button[data-testid="stBaseButton-tertiary"],
        html body .stApp [data-testid="stDownloadButton"] button,
        html body .stApp [data-testid="stLinkButton"] a,
        html body .stApp [data-testid*="Popover"] > button:not([kind="primary"]),
        html body .stApp [data-testid*="popover"] > button:not([kind="primary"]){{
            background-color:var(--control-bg)!important;
            background-image:none!important;
            color:var(--text)!important;
            border:1px solid var(--line-strong)!important;
            box-shadow:none!important;
        }}

        html body .stApp div[data-testid="stButton"] > button:not([kind="primary"]) *,
        html body .stApp .stButton > button:not([kind="primary"]) *,
        html body .stApp button[kind="secondary"] *,
        html body .stApp button[kind="tertiary"] *,
        html body .stApp button[data-testid="stBaseButton-secondary"] *,
        html body .stApp button[data-testid="stBaseButton-tertiary"] *,
        html body .stApp [data-testid="stDownloadButton"] button *,
        html body .stApp [data-testid="stLinkButton"] a *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            fill:currentColor!important;
        }}

        html body .stApp div[data-testid="stButton"] > button:not([kind="primary"]):hover,
        html body .stApp .stButton > button:not([kind="primary"]):hover,
        html body .stApp button[kind="secondary"]:hover,
        html body .stApp button[kind="tertiary"]:hover,
        html body .stApp button[data-testid="stBaseButton-secondary"]:hover,
        html body .stApp button[data-testid="stBaseButton-tertiary"]:hover{{
            background-color:var(--control-hover)!important;
            border-color:color-mix(in srgb,var(--accent) 55%,var(--line-strong))!important;
            color:var(--text)!important;
        }}

        /* Primary buttons are reasserted after the generic rule above. */
        html body .stApp div[data-testid="stButton"] > button[kind="primary"],
        html body .stApp button[data-testid="stBaseButton-primary"],
        html body .stApp button[kind="primary"]{{
            background:linear-gradient(135deg,var(--accent),color-mix(in srgb,var(--accent) 56%,#8874FF))!important;
            color:var(--primary-text)!important;
            border:0!important;
        }}
        html body .stApp div[data-testid="stButton"] > button[kind="primary"] *,
        html body .stApp button[data-testid="stBaseButton-primary"] *,
        html body .stApp button[kind="primary"] *{{
            color:var(--primary-text)!important;
            -webkit-text-fill-color:var(--primary-text)!important;
        }}

        /* Segmented navigation/profile controls. */
        html body .stApp [data-testid="stSegmentedControl"] button,
        html body .stApp [data-testid="stSegmentedControl"] [role="radio"]{{
            background-color:var(--control-bg)!important;
            color:var(--text)!important;
            border-color:var(--line-strong)!important;
        }}
        html body .stApp [data-testid="stSegmentedControl"] button *,
        html body .stApp [data-testid="stSegmentedControl"] [role="radio"] *{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
        }}
        html body .stApp [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        html body .stApp [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"]{{
            background:color-mix(in srgb,var(--accent) 16%,var(--control-bg))!important;
            border-color:var(--accent)!important;
            color:var(--text)!important;
        }}

        /* Code/console surfaces were another Streamlit light-theme leak. */
        html body .stApp [data-testid="stCode"],
        html body .stApp [data-testid="stCodeBlock"],
        html body .stApp [data-testid="stCode"] pre,
        html body .stApp [data-testid="stCodeBlock"] pre,
        html body .stApp pre{{
            background-color:var(--panel)!important;
            color:var(--text)!important;
            border-color:var(--line)!important;
        }}
        html body .stApp [data-testid="stCode"] code,
        html body .stApp [data-testid="stCodeBlock"] code,
        html body .stApp pre code{{
            color:var(--text)!important;
            -webkit-text-fill-color:var(--text)!important;
            background:transparent!important;
        }}

        /* Inputs and uploader/dropdown surfaces across every wizard page. */
        html body .stApp [data-baseweb="input"],
        html body .stApp [data-baseweb="base-input"],
        html body .stApp [data-baseweb="textarea"],
        html body .stApp [data-baseweb="select"] > div,
        html body .stApp [data-testid="stTextInputRootElement"],
        html body .stApp [data-testid="stNumberInputContainer"],
        html body .stApp [data-testid="stFileUploaderDropzone"]{{
            background-color:var(--control-bg)!important;
            color:var(--text)!important;
            border-color:var(--line-strong)!important;
        }}

        /* Header danger control: Stop while running, Close while idle. */
        html body .stApp .st-key-header_danger_control button,
        html body .stApp .st-key-header_stop_action button,
        html body .stApp .st-key-header_stop_close_action button,
        html body .stApp .st-key-header_close_action button{{
            background:var(--bad)!important;
            color:#FFFFFF!important;
            border:1px solid color-mix(in srgb,var(--bad) 88%,#000000)!important;
            box-shadow:0 10px 22px color-mix(in srgb,var(--bad) 22%,transparent)!important;
            font-weight:760!important;
        }}
        html body .stApp .st-key-header_danger_control button *,
        html body .stApp .st-key-header_stop_action button *,
        html body .stApp .st-key-header_stop_close_action button *,
        html body .stApp .st-key-header_close_action button *{{
            color:#FFFFFF!important;
            -webkit-text-fill-color:#FFFFFF!important;
            fill:currentColor!important;
        }}
        html body .stApp .st-key-header_danger_control button:hover,
        html body .stApp .st-key-header_stop_action button:hover,
        html body .stApp .st-key-header_stop_close_action button:hover,
        html body .stApp .st-key-header_close_action button:hover{{
            background:color-mix(in srgb,var(--bad) 84%,#7A0000)!important;
            border-color:var(--bad)!important;
        }}

        /* Legacy run-workspace stop styling retained for old session DOM only. */
        html body .stApp .st-key-run_stop_button button{{
            background:color-mix(in srgb,var(--bad) 9%,transparent)!important;
            color:var(--bad)!important;
            border:1px solid color-mix(in srgb,var(--bad) 55%,transparent)!important;
            box-shadow:none!important;
        }}
        html body .stApp .st-key-run_stop_button button *{{
            color:var(--bad)!important;
            -webkit-text-fill-color:var(--bad)!important;
            fill:currentColor!important;
        }}
        html body .stApp .st-key-run_stop_button button:hover{{
            background:color-mix(in srgb,var(--bad) 17%,transparent)!important;
            border-color:var(--bad)!important;
        }}

        {motion_css}
        @media(max-width:900px){{.wizard{{grid-template-columns:1fr 1fr}}.summary-grid{{grid-template-columns:1fr 1fr}}}}
        </style>
        """,
        unsafe_allow_html=True,
    )


class PipelineError(RuntimeError):
    """Raised when an external reconstruction command fails."""


class Console:
    """Live terminal view with an on-disk log for diagnostics and interrupted runs."""

    def __init__(self, slot, max_lines=240, log_path=None, activity_callback=None):
        self.slot = slot
        self.lines = deque(maxlen=max_lines)
        self.log_path = Path(log_path) if log_path else None
        self.activity_callback = activity_callback
        self.last_draw = 0.0

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if self.log_path.exists():
                try:
                    existing = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    self.lines.extend(existing[-max_lines:])
                except OSError:
                    pass

    def write(self, line, force=False):
        line = str(line).rstrip()
        if not line:
            return

        self.lines.append(line)
        if self.log_path:
            try:
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass

        if self.activity_callback:
            self.activity_callback()

        now = time.monotonic()
        if force or now - self.last_draw >= 0.15:
            self.slot.code("\n".join(self.lines), language="text")
            self.last_draw = now

    def flush(self):
        self.slot.code("\n".join(self.lines), language="text")


class RunProgress:
    """Live progress with a conservative ETA that learns for five minutes first."""

    ETA_RECALC_SECONDS = 60
    ETA_VISIBLE_AFTER_SECONDS = 5 * 60

    def __init__(self, progress_slot, elapsed_slot, eta_slot, model_slot, activity_slot, preview_slot=None, preview_meta_slot=None):
        self.progress_slot = progress_slot
        self.elapsed_slot = elapsed_slot
        self.eta_slot = eta_slot
        self.model_slot = model_slot
        self.activity_slot = activity_slot
        self.preview_slot = preview_slot
        self.preview_meta_slot = preview_meta_slot
        self.last_preview_signature = None
        self.percent = float(st.session_state.get("run_progress", 0.0))
        self.label = st.session_state.get("run_label", "Preparing")
        self.detail = st.session_state.get("run_detail", "Starting reconstruction")
        self.stage_index = int(st.session_state.get("run_stage_index") or pipeline_stage_for(self.label, self.detail))
        self.started_at = float(st.session_state.get("run_started_ts") or time.time())
        self.accumulated_elapsed = float(st.session_state.get("run_accumulated_runtime_seconds") or 0.0)
        self.last_activity = float(st.session_state.get("run_last_activity_ts") or self.started_at)
        self.eta_seconds = st.session_state.get("run_eta_seconds")
        self.last_eta_calc = float(st.session_state.get("run_eta_calc_ts") or 0.0)
        self.native_eta_seconds = st.session_state.get("run_native_eta_seconds")
        self.native_eta_updated = float(st.session_state.get("run_native_eta_updated_ts") or 0.0)
        self.last_draw = 0.0
        self.render(force=True)

    @staticmethod
    def _format_duration(seconds):
        if seconds is None:
            return "Learning..."
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    def touch_activity(self):
        self.last_activity = time.time()
        st.session_state.run_last_activity_ts = self.last_activity

    def note_native_eta(self, seconds):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            return
        if not math.isfinite(seconds) or seconds < 0:
            return
        self.native_eta_seconds = seconds
        self.native_eta_updated = time.time()
        st.session_state.run_native_eta_seconds = seconds
        st.session_state.run_native_eta_updated_ts = self.native_eta_updated

    def update(self, percent, label, detail="", force=False):
        label = str(label)
        detail = str(detail)
        requested = max(0.0, min(100.0, float(percent)))

        # Keep progress monotonic and reserve 100% for the final verified
        # "Complete" update. Optional repair/validation stages must not make
        # the bar jump backwards or claim completion early.
        is_true_complete = label.strip().lower() == "complete"
        if requested >= 100.0 and not is_true_complete:
            requested = 99.0
        if not is_true_complete:
            requested = min(requested, 99.0)
        self.percent = 100.0 if is_true_complete else max(self.percent, requested)

        self.label = label
        self.detail = detail
        self.stage_index = pipeline_stage_for(label, detail)

        st.session_state.run_progress = self.percent
        st.session_state.run_label = self.label
        st.session_state.run_detail = self.detail
        st.session_state.run_stage_index = self.stage_index

        self._maybe_recalculate_eta(force=is_true_complete)
        self.render(force=force)

    def heartbeat(self):
        self._maybe_recalculate_eta()
        self.render()

    def _maybe_recalculate_eta(self, force=False):
        now = time.time()
        segment_elapsed = max(0.0, now - self.started_at)
        elapsed = self.accumulated_elapsed + segment_elapsed
        due = self.last_eta_calc <= 0 and segment_elapsed >= self.ETA_RECALC_SECONDS
        due = due or (self.last_eta_calc > 0 and now - self.last_eta_calc >= self.ETA_RECALC_SECONDS)
        due = due or force
        if not due:
            return

        if self.percent >= 100:
            estimate = 0.0
        elif self.label.lower().startswith(("optimizing", "optimising")) and self.native_eta_seconds is not None and now - self.native_eta_updated <= 120:
            estimate = float(self.native_eta_seconds) + 20.0
        elif self.percent >= 2.0 and segment_elapsed > 0:
            estimate = segment_elapsed * (100.0 - self.percent) / self.percent
            estimate = min(estimate, 7 * 24 * 3600)
        else:
            estimate = None

        if estimate is not None and self.eta_seconds is not None and not force:
            estimate = 0.35 * float(self.eta_seconds) + 0.65 * estimate
        self.eta_seconds = estimate
        self.last_eta_calc = now
        st.session_state.run_eta_seconds = estimate
        st.session_state.run_eta_calc_ts = now

    def render(self, force=False):
        now_mono = time.monotonic()
        if not force and now_mono - self.last_draw < 0.45:
            return
        elapsed = self.accumulated_elapsed + max(0.0, time.time() - self.started_at)
        activity_age = max(0.0, time.time() - self.last_activity)
        if self.percent >= 100:
            eta_text = "Complete"
        elif elapsed < self.ETA_VISIBLE_AFTER_SECONDS:
            eta_text = "Learning..."
        else:
            eta_text = self._format_duration(self.eta_seconds)
        stage_percent = pipeline_stage_progress(self.stage_index, self.percent)
        pipeline_fraction = max(0.0, min(100.0, self.stage_index / PIPELINE_STAGE_TOTAL * 100.0))

        self.progress_slot.markdown(
            f"""
            <div class="progress-shell">
              <div class="progress-section">
                <div class="progress-kicker">Pipeline progress</div>
                <div class="progress-meta">
                  <span class="progress-label">Stage {self.stage_index} / {PIPELINE_STAGE_TOTAL}</span>
                  <span class="progress-value">{html.escape(PIPELINE_STAGES[self.stage_index - 1])}</span>
                </div>
                <div class="progress-track stage-track"><div class="progress-fill" style="width:{pipeline_fraction:.2f}%"></div></div>
              </div>

              <div class="progress-section">
                <div class="progress-kicker">Current stage</div>
                <div class="progress-meta">
                  <span class="progress-label">{html.escape(self.label)}</span>
                  <span class="progress-value-large">{stage_percent:.1f}%</span>
                </div>
                <div class="progress-track"><div class="progress-fill" style="width:{stage_percent:.2f}%"></div></div>
                <div class="progress-detail">{html.escape(self.detail)}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        self.elapsed_slot.metric("Elapsed", self._format_duration(elapsed))
        self.eta_slot.metric("ETA remaining", eta_text)
        gaussians = st.session_state.get("run_model_gaussians")
        stats = st.session_state.get("run_stats") or {}
        if gaussians:
            loss_value = st.session_state.get("run_model_loss")
            delta = f"loss {float(loss_value):.5f}" if loss_value is not None else None
            self.model_slot.metric(
                "Gaussian model",
                f"{int(gaussians):,} splats",
                delta=delta,
                delta_color="off",
            )
        elif stats.get("registered"):
            self.model_slot.metric("Camera model", f"{int(stats['registered']):,} cameras")
        elif stats.get("frame_count"):
            self.model_slot.metric("Working set", f"{int(stats['frame_count']):,} frames")
        else:
            self.model_slot.metric("Scene model", "Building…")
        self.activity_slot.metric("Terminal activity", "Now" if activity_age < 4 else self._format_duration(activity_age) + " ago")

        if self.preview_slot is not None:
            preview_path_value = st.session_state.get("run_preview_path")
            preview_path = Path(preview_path_value) if preview_path_value else None
            if preview_path and preview_path.exists():
                try:
                    signature = (str(preview_path), preview_path.stat().st_mtime_ns)
                except OSError:
                    signature = (str(preview_path), 0)
                if signature != self.last_preview_signature:
                    self.preview_slot.image(str(preview_path), width=300)
                    self.last_preview_signature = signature
                if self.preview_meta_slot is not None:
                    preview_step = int(st.session_state.get("run_preview_step") or 0)
                    updated = float(st.session_state.get("run_preview_updated_ts") or time.time())
                    age = self._format_duration(max(0.0, time.time() - updated))
                    preview_kind = st.session_state.get("run_preview_kind") or "Training preview"
                    step_text = f" · step {preview_step:,}" if preview_step else ""
                    self.preview_meta_slot.caption(f"{preview_kind}{step_text} · updated {age} ago")
            else:
                if self.last_preview_signature != ("waiting", 0):
                    self.preview_slot.markdown(
                        '<div style="height:190px;max-width:300px;border:1px dashed var(--line);border-radius:12px;display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:.78rem;text-align:center;padding:1rem;">AI preprocessing previews appear here first; native training replaces them at the first preview interval.</div>',
                        unsafe_allow_html=True,
                    )
                    self.last_preview_signature = ("waiting", 0)
                if self.preview_meta_slot is not None:
                    self.preview_meta_slot.caption("Maximum preview render size · 300 × 300 px")

        self.last_draw = now_mono

def slugify(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return value or "my-scene"


def human_size(num_bytes):
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _active_process_file(project_dir):
    return Path(project_dir) / ".splat_studio_active_process"


def terminate_process_tree(pid):
    """Stop a native child process and its process group on macOS/Linux."""
    try:
        if os.name == "posix":
            os.killpg(int(pid), signal.SIGTERM)
        else:
            os.kill(int(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        return

    # MLX catches SIGTERM and checkpoints at the next safe step boundary. Give
    # it a short grace window before escalating; FFmpeg/COLMAP normally exit
    # much sooner, so this does not meaningfully slow ordinary cancellation.
    deadline = time.time() + 30.0
    while time.time() < deadline:
        try:
            os.kill(int(pid), 0)
        except (ProcessLookupError, PermissionError, ValueError, OSError):
            return
        time.sleep(0.20)

    try:
        if os.name == "posix":
            os.killpg(int(pid), signal.SIGKILL)
        else:
            os.kill(int(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, ValueError, OSError):
        pass


def stop_stale_pipeline_process(project_dir):
    """Stop a command left alive by a previous interrupted Streamlit rerun."""
    pid_file = _active_process_file(project_dir)
    if not pid_file.exists():
        return

    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        pid_file.unlink(missing_ok=True)
        return

    terminate_process_tree(pid)
    pid_file.unlink(missing_ok=True)


def run_streaming(command, console, on_line=None, env=None, project_dir=None, on_heartbeat=None):
    """Run a command with live output, heartbeat updates, and no orphaned native child process."""
    command = [str(part) for part in command]

    # Prevent macOS idle sleep during long native reconstruction commands when requested.
    if (
        sys.platform == "darwin"
        and st.session_state.get("cfg_prevent_sleep", True)
        and shutil.which("caffeinate")
        and command[0] != "caffeinate"
    ):
        command = ["caffeinate", "-i"] + command

    console.write(f"$ {shlex.join(command)}", force=True)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
        start_new_session=(os.name == "posix"),
    )

    pid_file = _active_process_file(project_dir) if project_dir else None
    if pid_file:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(process.pid), encoding="utf-8")

    output_queue = queue.Queue()
    recent_output = deque(maxlen=40)

    def reader():
        try:
            if process.stdout is not None:
                for raw_line in iter(process.stdout.readline, ""):
                    output_queue.put(raw_line)
        finally:
            output_queue.put(None)

    reader_thread = threading.Thread(target=reader, daemon=True, name="splat-studio-subprocess-reader")
    reader_thread.start()

    try:
        stream_finished = False
        while not stream_finished:
            try:
                raw_line = output_queue.get(timeout=0.5)
            except queue.Empty:
                if on_heartbeat:
                    on_heartbeat()
                if process.poll() is not None and not reader_thread.is_alive():
                    break
                continue

            if raw_line is None:
                stream_finished = True
                continue

            line = raw_line.rstrip("\r\n")
            if line:
                recent_output.append(line)
                console.write(line)
                if on_line:
                    try:
                        on_line(line)
                    except Exception as telemetry_error:
                        # Progress and ETA parsing are UI-only. A display bug
                        # must never terminate an expensive native process.
                        console.write(
                            f"[ui warning] Progress telemetry error ignored: {telemetry_error}",
                            force=True,
                        )
            if on_heartbeat:
                on_heartbeat()

        return_code = process.wait()
        console.flush()

        if return_code != 0:
            message = f"Command exited with code {return_code}: {shlex.join(command)}"
            if recent_output:
                message += "\n\nLast terminal output:\n" + "\n".join(recent_output)
            raise PipelineError(message)

    except BaseException:
        if process.poll() is None:
            terminate_process_tree(process.pid)
        raise

    finally:
        if pid_file and pid_file.exists():
            try:
                if pid_file.read_text(encoding="utf-8").strip() == str(process.pid):
                    pid_file.unlink(missing_ok=True)
            except OSError:
                pass

def probe_duration(video_path):
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        return None


def hms_to_seconds(value):
    try:
        hours, minutes, seconds = value.split(":")
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    except (ValueError, AttributeError):
        return None


def _frame_quality_metrics(path):
    try:
        import cv2
    except ImportError:
        return 0.5, None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0, None
    h, w = image.shape[:2]
    scale = min(1.0, 320.0 / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (max(32, int(w * scale)), max(32, int(h * scale))), interpolation=cv2.INTER_AREA)
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    contrast = float(image.std())
    dark = float((image < 10).mean())
    bright = float((image > 245).mean())
    exposure = max(0.0, 1.0 - 2.2 * (dark + bright))
    score = math.log1p(max(0.0, sharpness)) * 0.62 + math.log1p(max(0.0, contrast)) * 0.24 + exposure * 1.35
    thumb = cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb = (thumb - thumb.mean()) / max(float(thumb.std()), 1.0)
    return float(score), thumb


def _thumbnail_similarity(first, second):
    if first is None or second is None:
        return 0.0
    return float(np.clip(np.mean(first * second), -1.0, 1.0))


def _select_smart_frames(candidates, target_frames, report_path, console):
    candidates = list(candidates)
    if len(candidates) <= target_frames:
        selected = candidates
    else:
        metrics = []
        for index, path in enumerate(candidates):
            score, thumb = _frame_quality_metrics(path)
            metrics.append((index, path, score, thumb))

        selected = []
        previous_thumb = None
        edges = np.linspace(0, len(metrics), int(target_frames) + 1, dtype=int)
        for bin_index in range(int(target_frames)):
            start, stop = int(edges[bin_index]), int(edges[bin_index + 1])
            bucket = metrics[start:max(start + 1, stop)]
            ranked = sorted(bucket, key=lambda item: item[2], reverse=True)
            chosen = ranked[0]
            for option in ranked:
                similarity = _thumbnail_similarity(previous_thumb, option[3])
                if similarity < 0.992:
                    chosen = option
                    break
            selected.append(chosen[1])
            previous_thumb = chosen[3]

    report = {
        "candidate_frames": len(candidates),
        "selected_frames": len(selected),
        "method": "temporal_bins+sharpness+contrast+exposure+duplicate_rejection",
        "selected": [path.name for path in selected],
    }
    try:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except OSError:
        pass
    console.write(f"[studio] Smart frame curation: {len(candidates):,} candidates → {len(selected):,} selected.", force=True)
    return selected


def extract_frames(
    video_path,
    images_dir,
    progress,
    console,
    target_frames,
    smart_select=True,
    progress_range=None,
    progress_label=None,
):
    """Sample video frames, optionally oversampling then retaining the sharpest well-spaced views."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PipelineError("ffmpeg was not found on PATH.")

    images_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(video_path)
    start, end = progress_range if progress_range is not None else EXTRACT_RANGE
    stage_label = str(progress_label or "Sampling video")
    project_dir = images_dir.parent

    if smart_select:
        candidate_dir = project_dir / ".splat_studio" / "frame_candidates"
        remove_path_robustly(candidate_dir)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_target = max(int(target_frames) + 20, int(target_frames) * 2)
        if duration and duration > 0:
            sample_fps = min(12.0, max(0.15, float(candidate_target) / duration))
            expected_candidates = max(2, int(round(duration * sample_fps)))
        else:
            sample_fps = 6.0
            expected_candidates = candidate_target
        output_pattern = candidate_dir / "%06d.jpg"
        progress.update(start, stage_label, f"Smart curation · collecting ~{expected_candidates:,} candidate frames", force=True)
    else:
        if duration and duration > 0:
            sample_fps = min(8.0, max(0.10, float(target_frames) / duration))
            expected_candidates = max(2, min(target_frames, int(round(duration * sample_fps))))
        else:
            sample_fps = 4.0
            expected_candidates = target_frames
        output_pattern = images_dir / "%05d.jpg"
        progress.update(start, stage_label, f"Evenly sampling ~{expected_candidates:,} frames", force=True)

    command = [
        ffmpeg, "-y", "-i", str(video_path), "-vf", f"fps={sample_fps:.6f}",
        "-qscale:v", "2", "-progress", "pipe:1", "-nostats", str(output_pattern),
    ]

    def parse_ffmpeg(line):
        elapsed = None
        if line.startswith(("out_time_us=", "out_time_ms=")):
            try:
                elapsed = float(line.split("=", 1)[1]) / 1_000_000
            except ValueError:
                pass
        elif line.startswith("out_time="):
            elapsed = hms_to_seconds(line.split("=", 1)[1])
        if duration and elapsed is not None:
            ratio = min(1.0, max(0.0, elapsed / duration))
            progress.update(start + (end - start) * ratio * (0.78 if smart_select else 1.0), stage_label, f"{elapsed:.1f}s / {duration:.1f}s")

    run_streaming(command, console, on_line=parse_ffmpeg, project_dir=project_dir, on_heartbeat=getattr(progress, "heartbeat", None))

    if smart_select:
        progress.update(start + (end - start) * 0.80, stage_label, "Curating sharpness, exposure and near-duplicate views", force=True)
        candidates = sorted(candidate_dir.glob("*.jpg"))
        selected = _select_smart_frames(candidates, int(target_frames), project_dir / "smart_frame_report.json", console)
        for old in images_dir.glob("*.jpg"):
            old.unlink(missing_ok=True)
        for index, selected_path in enumerate(selected, start=1):
            shutil.copy2(selected_path, images_dir / f"{index:05d}.jpg")
        remove_path_robustly(candidate_dir)
        frame_count = len(selected)
    else:
        frame_count = len(list(images_dir.glob("*.jpg")))

    progress.update(end, stage_label if progress_label else "Frames ready", f"{frame_count:,} images selected", force=True)
    return frame_count, sample_fps




def _fraction_from_line(line):
    """Return the most useful x/y counter found in a COLMAP terminal line."""
    counters = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", line) if int(b) > 0]
    if not counters:
        return None

    # When COLMAP emits nested block counters, flatten them into one fraction.
    if len(counters) >= 2:
        outer, outer_total = counters[0]
        inner, inner_total = counters[1]
        outer = min(max(outer, 1), outer_total)
        inner = min(max(inner, 0), inner_total)
        return ((outer - 1) + inner / inner_total) / outer_total, f"{outer}/{outer_total} · {inner}/{inner_total}"

    current, total = counters[0]
    if current <= total:
        return current / total, f"{current}/{total}"
    return None


def make_stage_parser(progress, stage_name, start_percent, end_percent):
    def parse(line):
        fraction = _fraction_from_line(line)
        if fraction:
            ratio, counter_text = fraction
            progress.update(
                start_percent + (end_percent - start_percent) * min(1.0, max(0.0, ratio)),
                stage_name,
                f"{stage_name} · {counter_text}",
            )
    return parse


def colmap_help(colmap, command):
    try:
        result = subprocess.run(
            [colmap, command, "-h"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return (result.stdout or "") + "\n" + (result.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return ""


def choose_supported_option(help_text, candidates):
    for option in candidates:
        if option in help_text:
            return option
    return None


def run_colmap(
    project_dir,
    images_dir,
    progress,
    console,
    sequential_overlap,
    max_features,
    max_image_size,
    shared_camera,
    camera_model,
    quadratic_overlap,
    guided_matching,
    robust_sift,
    rescue_mode=False,
    photo_mode=False,
):
    """Run the sparse COLMAP stages required by Gaussian training."""
    colmap = shutil.which("colmap")
    if not colmap:
        raise PipelineError("COLMAP was not found on PATH.")

    database_path = project_dir / "database.db"
    sparse_dir = project_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    feature_help = colmap_help(colmap, "feature_extractor")
    image_count = len([
        item for item in Path(images_dir).iterdir()
        if item.is_file() and item.suffix.lower() in PHOTO_EXTENSIONS
    ])
    use_exhaustive = bool(photo_mode and image_count <= 320)
    matcher_name = "exhaustive_matcher" if use_exhaustive else "sequential_matcher"
    matcher_help = colmap_help(colmap, matcher_name)

    max_image_flag = choose_supported_option(
        feature_help,
        ["--FeatureExtraction.max_image_size", "--SiftExtraction.max_image_size"],
    )
    overlap_flag = None if use_exhaustive else choose_supported_option(
        matcher_help,
        ["--SequentialMatching.overlap", "--SequentialPairing.overlap"],
    )
    quadratic_flag = None if use_exhaustive else choose_supported_option(
        matcher_help,
        ["--SequentialMatching.quadratic_overlap", "--SequentialPairing.quadratic_overlap"],
    )
    guided_flag = choose_supported_option(
        matcher_help,
        ["--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"],
    )

    feature_command = [
        colmap,
        "feature_extractor",
        "--database_path",
        str(database_path.resolve()),
        "--image_path",
        str(images_dir.resolve()),
        "--ImageReader.camera_model",
        str(camera_model),
        "--ImageReader.single_camera",
        "1" if shared_camera else "0",
        "--SiftExtraction.max_num_features",
        str(max_features),
    ]

    if max_image_flag:
        feature_command.extend([max_image_flag, str(max_image_size)])

    if robust_sift:
        if "--SiftExtraction.estimate_affine_shape" in feature_help:
            feature_command.extend(["--SiftExtraction.estimate_affine_shape", "1"])
        if "--SiftExtraction.domain_size_pooling" in feature_help:
            feature_command.extend(["--SiftExtraction.domain_size_pooling", "1"])

    if rescue_mode:
        feature_start, feature_end = 50.0, 52.5
        feature_label = "Camera rescue"
        feature_detail = "Re-extracting stronger SIFT landmarks"
    else:
        feature_start, feature_end = 12.0, 24.0
        feature_label = "Extracting features"
        feature_detail = "Finding SIFT landmarks in sampled frames"

    progress.update(feature_start, feature_label, feature_detail, force=True)
    run_streaming(
        feature_command,
        console,
        on_line=make_stage_parser(progress, "Camera rescue" if rescue_mode else "Feature extraction", feature_start, feature_end),
        project_dir=project_dir,
        on_heartbeat=getattr(progress, "heartbeat", None),
    )

    matcher_command = [
        colmap,
        matcher_name,
        "--database_path",
        str(database_path.resolve()),
    ]
    if overlap_flag:
        matcher_command.extend([overlap_flag, str(sequential_overlap)])
    if quadratic_flag:
        matcher_command.extend([quadratic_flag, "1" if quadratic_overlap else "0"])
    if guided_flag:
        matcher_command.extend([guided_flag, "1" if guided_matching else "0"])

    if rescue_mode:
        match_start, match_end = 52.5, 55.0
        match_label = "Camera rescue"
        match_detail = (
            "Exhaustively matching uploaded photos"
            if use_exhaustive
            else "Guided matching with wider temporal overlap"
        )
    else:
        match_start, match_end = 24.0, 36.0
        match_label = "Matching photos" if photo_mode else "Matching frames"
        match_detail = (
            "Comparing every uploaded photo pair"
            if use_exhaustive
            else ("Matching uploaded photos with broad overlap" if photo_mode else "Comparing nearby frames in video order")
        )

    progress.update(match_start, match_label, match_detail, force=True)
    run_streaming(
        matcher_command,
        console,
        on_line=make_stage_parser(
            progress,
            "Camera rescue" if rescue_mode else ("Exhaustive photo matching" if use_exhaustive else "Sequential matching"),
            match_start,
            match_end,
        ),
        project_dir=project_dir,
        on_heartbeat=getattr(progress, "heartbeat", None),
    )

    if rescue_mode:
        solve_start, solve_end = 55.0, 57.8
        solve_label = "Camera rescue"
        solve_detail = "Re-solving cameras from stronger correspondences"
    else:
        solve_start, solve_end = 36.0, 50.0
        solve_label = "Solving cameras"
        solve_detail = "Reconstructing camera positions and sparse geometry"

    progress.update(solve_start, solve_label, solve_detail, force=True)
    run_streaming(
        [
            colmap,
            "mapper",
            "--database_path",
            str(database_path.resolve()),
            "--image_path",
            str(images_dir.resolve()),
            "--output_path",
            str(sparse_dir.resolve()),
        ],
        console,
        on_line=make_stage_parser(progress, "Camera rescue" if rescue_mode else "Sparse reconstruction", solve_start, solve_end),
        project_dir=project_dir,
        on_heartbeat=getattr(progress, "heartbeat", None),
    )

    models = [p for p in sparse_dir.iterdir() if p.is_dir()]
    if not models:
        raise PipelineError("COLMAP finished but did not create a sparse reconstruction under sparse/.")

    progress.update(
        57.9 if rescue_mode else 50,
        "Camera rescue" if rescue_mode else "Camera solve complete",
        f"Sparse model ready · {models[0].name}",
        force=True,
    )

def build_training_command(
    project_dir,
    iterations,
    means_lr,
    batch_size,
    max_points,
    max_gaussians,
    train_max_side,
    ssim_weight,
    refine_enabled,
    antialiased,
    checkpoint_every,
    safe_mode_level=0,
    runtime_base_seconds=0.0,
):
    """Build Splat Studio's own RobotFlow gsplat-mlx multiview training command."""
    project_dir = Path(project_dir).resolve()
    training_dir = project_dir / "training"
    cache_dir = project_dir / ".splat_studio_train_cache"
    output_ply = project_dir / "splat.ply"
    output_splat = project_dir / "splat.splat"

    safe_mode_level = max(0, int(safe_mode_level))
    effective_side = int(train_max_side)
    effective_antialiased = bool(antialiased)
    effective_refine = bool(refine_enabled)
    effective_cache_limit_mb = 128
    chunk_steps = 100
    cache_clear_every = 1
    force_resume = False

    if safe_mode_level >= 1:
        # Metal Safe 1 keeps the checkpoint/model intact but reduces the
        # number of transient Metal objects produced by each raster step.
        effective_side = min(effective_side, 256)
        effective_antialiased = False
        effective_cache_limit_mb = 32
        chunk_steps = 50
        force_resume = True
    if safe_mode_level >= 2:
        # Metal Safe 2 is intentionally conservative: no free-buffer cache,
        # smaller rasterisation and no further topology growth.
        effective_side = min(effective_side, 192)
        effective_antialiased = False
        effective_refine = False
        effective_cache_limit_mb = 0
        chunk_steps = 25
        force_resume = True

    command = [
        sys.executable,
        str(TRAINER_SCRIPT),
        "--data-dir", str(project_dir),
        "--output-ply", str(output_ply),
        "--output-splat", str(output_splat),
        "--cache-dir", str(cache_dir),
        "--steps", str(int(iterations)),
        "--batch-size", str(int(batch_size)),
        "--max-points", str(int(max_points)),
        "--max-gaussians", str(int(max_gaussians)),
        "--train-max-side", str(int(effective_side)),
        "--means-lr", str(float(means_lr)),
        "--ssim-weight", str(float(ssim_weight)),
        "--rasterize-mode", "antialiased" if effective_antialiased else "classic",
        "--log-interval", "25",
        "--checkpoint-every", str(int(checkpoint_every)),
        "--process-chunk-steps", str(int(chunk_steps)),
        "--cache-clear-every", str(int(cache_clear_every)),
        "--mlx-cache-limit-mb", str(int(effective_cache_limit_mb)),
        "--runtime-base-seconds", str(float(max(0.0, runtime_base_seconds))),
        "--no-compile-training",
        "--resume",
        "--refine" if effective_refine else "--no-refine",
    ]
    if force_resume:
        command.append("--force-resume")
    return command, training_dir, cache_dir, output_ply, output_splat


def make_training_parser(progress, requested_iterations, project_dir=None):
    start, end = TRAIN_RANGE

    def parse_training(line):
        lower = line.lower()
        ratio = None
        detail = line

        # Splat Studio trainer: TRAIN step=125/2500 loss=... gaussians=...
        match = re.search(r"\btrain\s+step=(\d+)\s*/\s*(\d+)", lower)
        if match:
            current, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                ratio = current / total
                gaussians = re.search(r"\bgaussians=(\d+)", lower)
                loss = re.search(r"\bloss=([0-9.e+-]+)", lower)
                native_eta = re.search(r"\beta=([0-9.]+)s", lower)
                if native_eta and hasattr(progress, "note_native_eta"):
                    try:
                        progress.note_native_eta(float(native_eta.group(1)))
                    except (TypeError, ValueError, OverflowError):
                        pass
                if loss:
                    try:
                        st.session_state.run_model_loss = float(loss.group(1))
                    except ValueError:
                        pass
                if gaussians:
                    st.session_state.run_model_gaussians = int(gaussians.group(1))
                st.session_state.run_training_step = current
                detail = f"Training step {current:,} / {total:,}"

        if ratio is None:
            match = re.search(r"\brefine\s+step=(\d+)", lower)
            if match and requested_iterations > 0:
                current = int(match.group(1))
                ratio = current / requested_iterations
                after = re.search(r"\bafter=(\d+)", lower)
                detail = f"Adaptive Gaussian refinement at step {current:,}"
                if after:
                    st.session_state.run_model_gaussians = int(after.group(1))
                    detail += f" · model now {int(after.group(1)):,} splats"

        if ratio is None:
            resume = re.search(r"\bresume\s+step=(\d+)\s*/\s*(\d+)", lower)
            if resume:
                current, total = int(resume.group(1)), int(resume.group(2))
                if total > 0:
                    ratio = current / total
                    detail = f"Resuming safely from recovery checkpoint · step {current:,} / {total:,}"
                    if project_dir:
                        persist_current_runtime(project_dir, status="running", phase="training", last_checkpoint_step=current, total_steps=total, last_checkpoint_stage=10, resumed=True)
                        refresh_header_status()

        if ratio is None:
            checkpoint = re.search(r"\bcheckpoint\s+step=(\d+)", lower)
            if checkpoint and requested_iterations > 0:
                current = int(checkpoint.group(1))
                ratio = current / requested_iterations
                detail = f"Recovery checkpoint saved · step {current:,}"
                if project_dir:
                    persist_current_runtime(
                        project_dir,
                        status="running",
                        phase="training",
                        last_checkpoint_step=current,
                        total_steps=requested_iterations,
                        last_checkpoint_stage=10,
                        last_checkpoint_at=time.time(),
                    )
                    refresh_header_status()

        if ratio is None:
            chunk = re.search(r"\bchunk_complete\s+step=(\d+)\s*/\s*(\d+)", lower)
            if chunk:
                current, total = int(chunk.group(1)), int(chunk.group(2))
                if total > 0:
                    ratio = current / total
                    detail = f"Metal worker recycled safely at training step {current:,} / {total:,}"

        explicit_percent = None
        explicit_label = None
        if ratio is None and lower.startswith("stage "):
            if "dataset" in lower:
                explicit_percent = 60.0
                explicit_label = "Preparing training data"
                detail = "Preparing registered COLMAP views"
            elif "undistort" in lower:
                explicit_percent = 62.0
                explicit_label = "Undistorting training views"
                detail = "Undistorting camera frames"
            elif "init" in lower:
                explicit_percent = 66.0
                explicit_label = "Initialising Gaussians"
                detail = "Initialising Gaussians from sparse geometry"
            elif "export" in lower:
                explicit_percent = 96.0
                explicit_label = "Exporting result"
                detail = "Exporting PLY and .splat"

        if explicit_percent is not None:
            progress.update(explicit_percent, explicit_label, detail)
        elif ratio is not None:
            ratio = min(1.0, max(0.0, ratio))
            progress.update(start + (end - start) * ratio, "Optimising splats", detail)

    return parse_training



def make_native_metal_training_parser(progress, total_steps, project_dir):
    step_pattern = re.compile(r"\bstep=(\d+)\b", re.IGNORECASE)
    gaussian_pattern = re.compile(r"\bgaussians=(\d+)\b", re.IGNORECASE)
    checkpoint_pattern = re.compile(
        r"^CHECKPOINT\s+step=(\d+)\s+total=(\d+)\s+gaussians=(\d+)\s+elapsed=([0-9.]+)s.*?path=(.+)$",
        re.IGNORECASE,
    )
    resume_pattern = re.compile(
        r"^RESUME\s+step=(\d+)\s+total=(\d+)\s+gaussians=(\d+)\s+elapsed=([0-9.]+)s.*?path=(.+)$",
        re.IGNORECASE,
    )
    preview_pattern = re.compile(
        r"^PREVIEW\s+step=(\d+)\s+path=(.+?)\s+width=(\d+)\s+height=(\d+)$",
        re.IGNORECASE,
    )

    def parse_native_training(line):
        text = str(line).strip()
        lower = text.lower()

        checkpoint_match = checkpoint_pattern.search(text)
        if checkpoint_match:
            step = int(checkpoint_match.group(1))
            total = int(checkpoint_match.group(2))
            gaussians = int(checkpoint_match.group(3))
            runtime_seconds = float(checkpoint_match.group(4))
            st.session_state.run_training_step = step
            st.session_state.run_training_total = total
            st.session_state.run_model_gaussians = gaussians
            save_project_state(
                project_dir,
                status="running",
                phase="training",
                last_checkpoint_step=step,
                total_steps=total,
                last_checkpoint_stage=10,
                checkpoint_runtime_seconds=runtime_seconds,
            )
            refresh_header_status()
            progress.render(force=True)
            return

        resume_match = resume_pattern.search(text)
        if resume_match:
            step = int(resume_match.group(1))
            total = int(resume_match.group(2))
            gaussians = int(resume_match.group(3))
            st.session_state.run_training_step = step
            st.session_state.run_training_total = total
            st.session_state.run_model_gaussians = gaussians
            progress.update(
                70.0 + 26.0 * (step / max(1, total)),
                "Optimising splats",
                f"Resumed native Metal snapshot · {step:,}/{total:,}",
                force=True,
            )
            refresh_header_status()
            return

        preview_match = preview_pattern.search(text)
        if preview_match:
            step = int(preview_match.group(1))
            preview_path = Path(preview_match.group(2)).expanduser()
            st.session_state.run_preview_path = str(preview_path)
            st.session_state.run_preview_step = step
            st.session_state.run_preview_kind = "Training preview"
            st.session_state.run_preview_updated_ts = time.time()
            progress.render(force=True)
            return

        gmatch = gaussian_pattern.search(text)
        if gmatch:
            try:
                st.session_state.run_model_gaussians = int(gmatch.group(1))
            except (TypeError, ValueError):
                pass

        if "loading colmap scene" in lower:
            progress.update(61, "Preparing training data", "Loading undistorted COLMAP cameras")
            return
        if "preparing colmap sparse points" in lower:
            progress.update(65, "Initialising Gaussians", "Loading reconstructed SfM points and colours")
            return
        if "initializing knn log scales" in lower:
            progress.update(67, "Initialising Gaussians", "Calculating Gaussian scales from neighbouring SfM points")
            return
        if "initializing sh model" in lower:
            progress.update(69, "Initialising Gaussians", "Creating spherical-harmonic Gaussian model")
            return
        if "entering training loop" in lower:
            progress.update(70, "Optimising splats", f"Native Metal training · 0/{int(total_steps):,}")
            return

        match = step_pattern.search(text)
        if match and lower.startswith("step="):
            try:
                step = max(0, min(int(total_steps), int(match.group(1))))
                ratio = step / max(1, int(total_steps))
                progress.update(
                    70.0 + 26.0 * ratio,
                    "Optimising splats",
                    f"Native Metal training · {step:,}/{int(total_steps):,}",
                )
                st.session_state.run_training_step = step
                st.session_state.run_training_total = int(total_steps)
                loss_match = re.search(r"\bloss=([0-9.eE+-]+)", text)
                if loss_match:
                    st.session_state.run_model_loss = float(loss_match.group(1))
            except (TypeError, ValueError):
                pass
            return

        if "running final evaluation" in lower:
            progress.update(96.5, "Exporting result", "Evaluating final Gaussian model")
            return
        if "360 points multi-view training ok" in lower:
            progress.update(99, "Finalising result", "Native SPZ export complete")

    return parse_native_training



def native_metal_backend_status():
    missing = []
    if not GSPLAT_METAL_ROOT.exists():
        missing.append(str(GSPLAT_METAL_ROOT))
    if not GSPLAT_METAL_PYTHON.exists():
        missing.append(str(GSPLAT_METAL_PYTHON))
    if not GSPLAT_METAL_TRAINER.exists():
        missing.append(str(GSPLAT_METAL_TRAINER))
    required_helpers = [
        GSPLAT_METAL_TEST_DIR / "colmap_360_dataset.py",
        GSPLAT_METAL_TEST_DIR / "scanner_dataset_utils.py",
        GSPLAT_METAL_TEST_DIR / "scanner_points_training_utils.py",
    ]
    for helper in required_helpers:
        if not helper.exists():
            missing.append(str(helper))
    return {
        "ready": not missing,
        "missing": missing,
        "root": GSPLAT_METAL_ROOT,
        "python": GSPLAT_METAL_PYTHON,
        "trainer": GSPLAT_METAL_TRAINER,
    }


def _native_best_sparse_model(project_dir):
    metrics = analyze_sparse_model(project_dir)
    model_name = str(metrics.get("model") or "")
    candidate = Path(project_dir) / "sparse" / model_name if model_name else None
    if candidate and candidate.is_dir():
        return candidate

    sparse = Path(project_dir) / "sparse"
    models = [item for item in sparse.iterdir() if item.is_dir()] if sparse.exists() else []
    if not models:
        raise PipelineError("No COLMAP sparse camera model is available for native Metal training.")
    return sorted(models, key=lambda p: p.name)[0]


def _native_image_size(image_path):
    code = (
        "from PIL import Image; import sys; "
        "im=Image.open(sys.argv[1]); print(str(im.width)+' '+str(im.height))"
    )
    result = subprocess.run(
        [str(GSPLAT_METAL_PYTHON), "-c", code, str(image_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise PipelineError(f"Unable to inspect training image size: {result.stderr.strip()}")
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        raise PipelineError(f"Unexpected image-size response: {result.stdout.strip()}")
    return int(parts[0]), int(parts[1])


def prepare_native_metal_dataset(project_dir, train_max_side, progress, console):
    project_dir = Path(project_dir).resolve()

    # Reuse the expensive undistortion created by older Splat Studio builds.
    legacy = project_dir / ".splat_studio_train_cache" / "undistorted"
    if (legacy / "images").is_dir() and (legacy / "sparse").exists():
        dataset_dir = legacy
        console.write("[studio] Reusing existing undistorted COLMAP training dataset.", force=True)
    else:
        dataset_dir = project_dir / NATIVE_TRAINING_DIRNAME / "dataset"
        marker = dataset_dir / ".splat_studio_native_dataset.json"
        best_model = _native_best_sparse_model(project_dir)

        valid = False
        if marker.exists():
            try:
                data = json.loads(marker.read_text(encoding="utf-8"))
                valid = (
                    data.get("model") == str(best_model.resolve())
                    and (dataset_dir / "images").is_dir()
                    and (dataset_dir / "sparse").exists()
                )
            except (OSError, json.JSONDecodeError):
                valid = False

        if not valid:
            colmap = shutil.which("colmap")
            if not colmap:
                raise PipelineError("COLMAP is required to prepare undistorted images for native Metal training.")
            if dataset_dir.exists():
                remove_path_robustly(dataset_dir)
            dataset_dir.mkdir(parents=True, exist_ok=True)

            progress.update(60, "Undistorting training views", "Preparing native Metal COLMAP dataset")
            console.write(f"[studio] Undistorting best COLMAP model: {best_model}", force=True)
            run_streaming(
                [
                    colmap,
                    "image_undistorter",
                    "--image_path", str((project_dir / "images").resolve()),
                    "--input_path", str(best_model.resolve()),
                    "--output_path", str(dataset_dir.resolve()),
                    "--output_type", "COLMAP",
                ],
                console,
                project_dir=project_dir,
                on_heartbeat=getattr(progress, "heartbeat", None),
            )
            marker.write_text(
                json.dumps({"model": str(best_model.resolve()), "created_at": time.time()}, indent=2),
                encoding="utf-8",
            )

    image_root = dataset_dir / "images"
    images = sorted(
        p for p in image_root.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    if not images:
        raise PipelineError(f"No undistorted training images were found under {image_root}.")

    raw_w, raw_h = _native_image_size(images[0])
    max_side = max(64, int(train_max_side))
    scale = min(1.0, max_side / max(raw_w, raw_h))
    width = max(32, int(round(raw_w * scale)))
    height = max(32, int(round(raw_h * scale)))

    console.write(
        f"[studio] Native dataset: {len(images):,} images · source {raw_w}x{raw_h} · training {width}x{height}",
        force=True,
    )
    return dataset_dir, width, height


def write_native_training_meta(project_dir, native_output_dir):
    summary_path = Path(native_output_dir) / "training_summary.json"
    meta = {
        "backend": "a1091150/gsplat-mlx native C++/Metal",
        "format": "SPZ",
    }

    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            meta.update(summary)
            meta["gaussians"] = int(summary.get("exported_gaussians") or 0)
            frames = summary.get("frame_summaries") or []
            if frames:
                meta["first_view_psnr_db"] = float(frames[0].get("final_psnr") or 0.0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    (Path(project_dir) / "splat.training.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return meta

@st.cache_data(ttl=120, show_spinner=False)
def ai_runtime_status():
    if not GSPLAT_METAL_PYTHON.exists():
        return {"ready": False, "mps": False, "detail": "Splat Studio Python runtime is missing"}
    probe = (
        "import torch, transformers; "
        "print('READY=1'); "
        "print('MPS='+('1' if torch.backends.mps.is_available() else '0')); "
        "print('TORCH='+torch.__version__); print('TRANSFORMERS='+transformers.__version__)"
    )
    try:
        result = subprocess.run([str(GSPLAT_METAL_PYTHON), "-c", probe], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ready": False, "mps": False, "detail": str(exc)}
    if result.returncode != 0:
        return {"ready": False, "mps": False, "detail": (result.stderr or result.stdout).strip()[-500:]}
    text = result.stdout
    return {
        "ready": "READY=1" in text,
        "mps": "MPS=1" in text,
        "detail": text.replace("\n", " · ").strip(),
    }


def make_ai_vision_parser(progress, project_dir):
    progress_pattern = re.compile(r"^AI_PROGRESS\s+stage=(\w+)\s+current=(\d+)\s+total=(\d+)$", re.IGNORECASE)
    preview_pattern = re.compile(r"^AI_PREVIEW\s+kind=(\w+)\s+path=(.+?)\s+width=(\d+)\s+height=(\d+)$", re.IGNORECASE)
    points_pattern = re.compile(r"^AI_DEPTH_POINTS\s+count=(\d+)\s+aligned_frames=(\d+)", re.IGNORECASE)

    def parse_ai(line):
        text = str(line).strip()
        match = progress_pattern.search(text)
        if match:
            stage = match.group(1).lower()
            current, total = int(match.group(2)), max(1, int(match.group(3)))
            ratio = current / total
            if stage == "mask":
                progress.update(62.2 + ratio * 1.8, "AI scene masks", f"Semantic masking · {current:,}/{total:,}")
            else:
                progress.update(64.0 + ratio * 2.0, "AI depth assistance", f"Depth Anything V2 · {current:,}/{total:,}")
            return
        match = preview_pattern.search(text)
        if match:
            path = Path(match.group(2)).expanduser()
            st.session_state.run_preview_path = str(path)
            st.session_state.run_preview_step = 0
            st.session_state.run_preview_kind = f"AI {match.group(1).title()} preview"
            st.session_state.run_preview_updated_ts = time.time()
            progress.render(force=True)
            return
        match = points_pattern.search(text)
        if match:
            points = int(match.group(1))
            st.session_state.run_ai_depth_points = points
            save_project_state(project_dir, ai_depth_points=points)
            return

    return parse_ai


def prepare_ai_assistance(dataset_dir, project_dir, config, progress, console):
    enabled = bool(config.get("ai_depth")) or bool(config.get("ai_mask_sky")) or bool(config.get("ai_mask_dynamic"))
    if not enabled:
        return None, None

    runtime = ai_runtime_status()
    if not runtime.get("ready"):
        raise PipelineError(
            "AI preprocessing is enabled but its local runtime is not installed. "
            "Run 'Setup AI Vision.command' from the SplatStudio folder once, then retry."
        )

    output_dir = Path(project_dir) / AI_OUTPUT_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(GSPLAT_METAL_PYTHON), str(AI_VISION_SCRIPT),
        "--dataset", str(Path(dataset_dir).resolve()),
        "--backend-test-dir", str(GSPLAT_METAL_TEST_DIR.resolve()),
        "--output-dir", str(output_dir.resolve()),
        "--max-depth-points", str(int(config.get("ai_depth_points", 40000))),
        "--depth-max-frames", "64",
        "--mask-dilate", "7",
        "--depth-model", AI_DEPTH_MODEL,
        "--segment-model", AI_SEGMENT_MODEL,
        "--depth-assist" if config.get("ai_depth") else "--no-depth-assist",
        "--mask-sky" if config.get("ai_mask_sky") else "--no-mask-sky",
        "--mask-dynamic" if config.get("ai_mask_dynamic") else "--no-mask-dynamic",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(AI_MODELS_ROOT.resolve())
    env["TRANSFORMERS_CACHE"] = str(AI_MODELS_ROOT.resolve())
    progress.update(62, "AI preprocessing", "Starting local vision models", force=True)
    console.write(
        f"[studio] AI preprocessing · depth={'on' if config.get('ai_depth') else 'off'} · "
        f"sky mask={'on' if config.get('ai_mask_sky') else 'off'} · people/vehicle mask={'on' if config.get('ai_mask_dynamic') else 'off'}",
        force=True,
    )
    try:
        run_streaming(
            command,
            console,
            on_line=make_ai_vision_parser(progress, project_dir),
            env=env,
            project_dir=project_dir,
            on_heartbeat=getattr(progress, "heartbeat", None),
        )
    except PipelineError as exc:
        # AI is an enhancement, not a prerequisite for native 3DGS. Do not
        # sacrifice a good COLMAP reconstruction because an optional local
        # vision model failed.
        if st.session_state.get("run_status") == "cancelled":
            raise
        message = str(exc)
        console.write(
            "[studio] Optional AI preprocessing failed; continuing with COLMAP geometry "
            "and native Metal training without AI assistance.",
            force=True,
        )
        console.write(f"[studio] AI warning: {message[-1200:]}", force=True)
        save_project_state(
            project_dir,
            ai_enabled=False,
            ai_error=message[-4000:],
            ai_fallback_used=True,
        )
        progress.update(
            66,
            "AI preprocessing skipped",
            "Continuing safely without optional AI assistance",
            force=True,
        )
        return None, None

    manifest_path = output_dir / "ai_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    mask_dir = Path(manifest["mask_dir"]) if manifest.get("mask_dir") else None
    extra_points = Path(manifest["extra_points_npz"]) if manifest.get("extra_points_npz") else None
    save_project_state(
        project_dir,
        ai_enabled=True,
        ai_manifest=str(manifest_path),
        ai_mask_dir=str(mask_dir) if mask_dir else None,
        ai_extra_points=str(extra_points) if extra_points else None,
    )
    return mask_dir, extra_points

def run_training(
    project_dir,
    iterations,
    means_lr,
    batch_size,
    max_points,
    max_gaussians,
    train_max_side,
    ssim_weight,
    refine_enabled,
    antialiased,
    checkpoint_every,
    preview_every,
    auto_resume,
    ai_depth,
    ai_depth_points,
    ai_mask_sky,
    ai_mask_dynamic,
    progress,
    console,
):
    project_dir = Path(project_dir).resolve()
    backend = native_metal_backend_status()
    if not backend["ready"]:
        raise PipelineError(
            "Native Metal trainer is not ready. Missing: " + ", ".join(backend["missing"])
        )

    progress.update(58, "Preparing training data", "Preparing native C++/Metal Gaussian backend", force=True)
    console.write("[studio] Backend: a1091150/gsplat-mlx native C++ / Metal", force=True)
    console.write(f"[studio] Backend repo: {GSPLAT_METAL_ROOT}")
    console.write(f"[studio] Backend Python: {GSPLAT_METAL_PYTHON}")

    dataset_dir, base_width, base_height = prepare_native_metal_dataset(
        project_dir,
        train_max_side,
        progress,
        console,
    )

    ai_config = {
        "ai_depth": bool(ai_depth),
        "ai_depth_points": int(ai_depth_points),
        "ai_mask_sky": bool(ai_mask_sky),
        "ai_mask_dynamic": bool(ai_mask_dynamic),
    }
    ai_mask_dir, ai_extra_points = prepare_ai_assistance(dataset_dir, project_dir, ai_config, progress, console)

    native_output_dir = project_dir / "native_metal_training"
    native_output_dir.mkdir(parents=True, exist_ok=True)
    state_root = project_dir / NATIVE_TRAINING_DIRNAME
    checkpoint_root = state_root / "checkpoints"
    preview_dir = state_root / "previews"
    output_spz = project_dir / "splat.spz"
    output_model = native_output_dir / "trained_model_params.npz"
    output_spz.unlink(missing_ok=True)

    requested_points = max(0, int(max_points))
    recovery_level = int((load_project_state(project_dir).get("native_recovery_level") or 0))
    recovery_level = max(0, min(3, recovery_level))
    retry_count = 0
    max_retries = 3
    retry_delay_seconds = 120

    def profile(level):
        caps = {0: None, 1: 384, 2: 320, 3: 256}
        caches = {0: 8.0, 1: 4.0, 2: 2.0, 3: 1.0}
        cap = caps[level]
        scale = 1.0 if cap is None else min(1.0, cap / max(base_width, base_height))
        width = max(32, int(round(base_width * scale)))
        height = max(32, int(round(base_height * scale)))
        use_refine = bool(refine_enabled) if level < 2 else False
        return width, height, caches[level], use_refine

    while True:
        width, height, cache_gb, use_refine = profile(recovery_level)
        runtime_base = current_total_runtime_seconds(project_dir)
        command = [
            str(GSPLAT_METAL_PYTHON),
            str(GSPLAT_METAL_TRAINER),
            "--data", str(dataset_dir.resolve()),
            "--out-dir", str(native_output_dir.resolve()),
            "--out-spz", str(output_spz.resolve()),
            "--out-model-npz", str(output_model.resolve()),
            "--data-factor", "1",
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--max-points", str(requested_points),
            "--steps", str(int(iterations)),
            "--batch-size", str(int(batch_size)),
            "--means-lr", str(float(means_lr)),
            "--ssim-lambda", str(float(ssim_weight)),
            "--sh-degree", "3",
            "--sh-degree-interval", "1000",
            "--log-interval", "25",
            "--mlx-cache-limit-gb", str(float(cache_gb)),
            "--checkpoint-dir", str(checkpoint_root.resolve()),
            "--checkpoint-every", str(max(0, int(checkpoint_every))),
            "--checkpoint-keep", "2",
            "--runtime-base-seconds", str(float(runtime_base)),
            "--preview-dir", str(preview_dir.resolve()),
            "--preview-interval", str(max(0, int(preview_every))),
            "--preview-max-side", "300",
            "--preview-keep", "8",
            *(["--mask-dir", str(ai_mask_dir.resolve())] if ai_mask_dir else []),
            *(["--extra-points-npz", str(ai_extra_points.resolve())] if ai_extra_points else []),
            "--resume",
            "--refine-enabled" if use_refine else "--no-refine-enabled",
        ]

        console.write(
            f"[studio] Native training: {int(iterations):,} steps · {width}x{height} · "
            f"SfM points {'ALL' if requested_points == 0 else f'{requested_points:,}'} · "
            f"SH3 · SSIM {float(ssim_weight):.2f} · refine {'on' if use_refine else 'off'} · "
            f"snapshots every {int(checkpoint_every):,} · previews every {int(preview_every):,} · "
            f"recovery level {recovery_level}",
            force=True,
        )

        command_to_run = command
        if sys.platform == "darwin" and st.session_state.get("cfg_prevent_sleep") and shutil.which("caffeinate"):
            command_to_run = ["caffeinate", "-i"] + command

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["SPLAT_METAL_TEST_DIR"] = str(GSPLAT_METAL_TEST_DIR.resolve())
        if recovery_level >= 1:
            # Reduces work accumulated in individual Metal command buffers.
            env["MLX_MAX_OPS_PER_BUFFER"] = "1"

        checkpoint = training_checkpoint_info(project_dir)
        checkpoint_step = int((checkpoint or {}).get("step") or 0)
        progress.update(
            70.0 + 26.0 * checkpoint_step / max(1, int(iterations)),
            "Optimising splats",
            f"Native Metal training · {checkpoint_step:,}/{int(iterations):,}",
            force=True,
        )

        try:
            run_streaming(
                command_to_run,
                console,
                on_line=make_native_metal_training_parser(progress, int(iterations), project_dir),
                env=env,
                project_dir=project_dir,
                on_heartbeat=getattr(progress, "heartbeat", None),
            )
            break
        except PipelineError as exc:
            error_text = str(exc)
            lower_error = error_text.lower()
            # User-requested Stop is never auto-resumed.
            if "exited with code 130" in lower_error or st.session_state.get("run_status") != "running":
                raise
            if not auto_resume or retry_count >= max_retries:
                raise

            retry_count += 1
            recovery_level = min(3, max(recovery_level + 1, retry_count))
            checkpoint = training_checkpoint_info(project_dir)
            checkpoint_step = int((checkpoint or {}).get("step") or 0)
            snapshot_text = (
                f"snapshot {checkpoint_step:,}/{int(iterations):,}"
                if checkpoint_step
                else "no snapshot yet; training will restart from step 0"
            )
            retry_at = time.time() + retry_delay_seconds

            persist_current_runtime(
                project_dir,
                status="running",
                phase="training",
                last_error=error_text[-4000:],
                auto_resume_retry_at=retry_at,
                auto_resume_attempt=retry_count,
                native_recovery_level=recovery_level,
                last_checkpoint_step=checkpoint_step,
                total_steps=int(iterations),
                last_checkpoint_stage=10,
            )
            console.write(
                f"[studio] Training worker failed. AUTO-RESUME {retry_count}/{max_retries} scheduled in 2 minutes "
                f"from {snapshot_text}; recovery level {recovery_level} lowers GPU pressure.",
                force=True,
            )

            for remaining in range(retry_delay_seconds, 0, -1):
                if st.session_state.get("run_status") != "running":
                    raise PipelineError("Automatic resume cancelled because the run was stopped.")
                if remaining == retry_delay_seconds or remaining % 30 == 0 or remaining <= 10:
                    console.write(
                        f"[studio] Auto-resume in {remaining // 60}:{remaining % 60:02d} · "
                        f"recovery level {recovery_level} · {snapshot_text}",
                        force=remaining <= 10,
                    )
                progress.update(
                    70.0 + 26.0 * checkpoint_step / max(1, int(iterations)),
                    "Optimising splats",
                    f"Auto-resume in {remaining // 60}:{remaining % 60:02d} · Recovery {recovery_level}/3 · {snapshot_text}",
                    force=(remaining % 5 == 0 or remaining <= 5),
                )
                refresh_header_status()
                time.sleep(1.0)

            save_project_state(
                project_dir,
                auto_resume_retry_at=None,
                auto_resume_attempt=retry_count,
                native_recovery_level=recovery_level,
            )
            console.write(f"[studio] Auto-resume starting now · recovery level {recovery_level}.", force=True)
            refresh_header_status()
            continue

    if not output_spz.exists() or output_spz.stat().st_size <= 0:
        raise PipelineError(
            "Native Metal training finished without producing splat.spz. "
            "Check the Live console / pipeline log for the trainer's final output."
        )

    meta = write_native_training_meta(project_dir, native_output_dir)

    try:
        output_spz = ensure_generated_splat_upright(output_spz, project_dir, console=console)
        meta = load_training_meta(output_spz)
    except Exception as exc:
        # Training is valuable even if the optional presentation transform
        # cannot run. Review/Edit will retry this automatically later.
        console.write(
            f"[studio] Orientation correction deferred: {exc}",
            force=True,
        )
        save_project_state(project_dir, splat_orientation_error=str(exc)[-2000:])

    gaussians = int(meta.get("gaussians") or meta.get("exported_gaussians") or 0)
    if gaussians:
        st.session_state.run_model_gaussians = gaussians

    save_project_state(
        project_dir,
        training_engine="a1091150/gsplat-mlx + Splat Studio snapshots",
        training_backend="native_cpp_metal",
        native_training_resolution=[int(width), int(height)],
        native_initial_points="all" if requested_points == 0 else requested_points,
        native_recovery_level=0,
        auto_resume_retry_at=None,
        auto_resume_attempt=0,
    )
    progress.update(99, "Finalising result", f"Native SPZ · {human_size(output_spz.stat().st_size)}", force=True)
    return output_spz




def remove_path_robustly(target, attempts=8):
    """Remove generated data while tolerating short macOS filesystem races."""
    target = Path(target)
    for attempt in range(attempts):
        try:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno not in (errno.ENOTEMPTY, errno.EBUSY, errno.EACCES) or attempt == attempts - 1:
                raise
            time.sleep(0.15 * (attempt + 1))


def clean_project(project_dir):
    """Clear generated reconstruction data, preserving the source and imported splats."""
    project_dir = Path(project_dir)
    stop_stale_pipeline_process(project_dir)
    time.sleep(0.15)

    for name in ("images", "sparse", "dense", "stereo", "training", ".splat_studio_train_cache", ".splat_studio_ai", ".splat_studio_native_metal", "native_metal_training", "database.db", "database.db-shm", "database.db-wal"):
        target = project_dir / name
        try:
            remove_path_robustly(target)
        except OSError as exc:
            raise PipelineError(f"Could not clean {target}: {exc}") from exc

    imports_dir = (project_dir / "imports").resolve()
    for splat_file in project_dir.rglob("*"):
        if not splat_file.is_file() or splat_file.suffix.lower() not in {".ply", ".sog", ".spz", ".splat"}:
            continue
        try:
            resolved = splat_file.resolve()
            if imports_dir == resolved.parent or imports_dir in resolved.parents:
                continue
            remove_path_robustly(splat_file)
        except OSError:
            pass


def missing_dependencies():
    missing = [name for name in ("ffmpeg", "colmap") if not shutil.which(name)]
    if not TRAINER_SCRIPT.exists():
        missing.append(str(TRAINER_SCRIPT))
    return missing



def command_version(command):
    """Return the first version line from a command, or None when unavailable."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=8)
        text = (result.stdout or result.stderr or "").strip()
        return text.splitlines()[0] if text else None
    except (OSError, subprocess.SubprocessError):
        return None


def node_version_ok():
    """The pinned SuperSplat editor currently requires Node 20.19+."""
    version = command_version(["node", "--version"])
    if not version:
        return False, None

    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return False, version

    current = tuple(int(part) for part in match.groups())
    return current >= (20, 19, 0), version


def playcanvas_status():
    node_ok, node_version = node_version_ok()
    return {
        "git": command_version(["git", "--version"]),
        "npm": command_version(["npm", "--version"]),
        "node": node_version,
        "node_ok": node_ok,
        "editor": (SUPERSPLAT_EDITOR_DIST / "index.html").exists(),
        "viewer": (SUPERSPLAT_VIEWER_PUBLIC / "index.html").exists(),
    }


def run_capture(command, cwd=None):
    """Run a setup command and raise a readable error containing its output."""
    result = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
        raise PipelineError(f"{shlex.join([str(p) for p in command])}\n\n{output[-6000:]}")
    return ((result.stdout or "") + "\n" + (result.stderr or "")).strip()


def clone_pinned_repo(url, ref, target, replace=False):
    """Clone a stable PlayCanvas release so upstream changes do not randomly break the app."""
    if replace and target.exists():
        shutil.rmtree(target)

    if target.exists():
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    run_capture(["git", "clone", "--depth", "1", "--branch", ref, url, str(target)])


def install_playcanvas_tools(reinstall=False):
    """Install and build the local open-source SuperSplat editor and viewer."""
    status = playcanvas_status()
    missing = []

    if not status["git"]:
        missing.append("git")
    if not status["npm"]:
        missing.append("npm")
    if not status["node_ok"]:
        missing.append("Node.js 20.19+")

    if missing:
        raise PipelineError("PlayCanvas setup requires: " + ", ".join(missing))

    clone_pinned_repo(
        "https://github.com/playcanvas/supersplat.git",
        SUPERSPLAT_REF,
        SUPERSPLAT_DIR,
        replace=reinstall,
    )
    clone_pinned_repo(
        "https://github.com/playcanvas/supersplat-viewer.git",
        SUPERSPLAT_VIEWER_REF,
        SUPERSPLAT_VIEWER_DIR,
        replace=reinstall,
    )

    # npm ci uses each project's lockfile, giving a much more reproducible local build.
    run_capture(["npm", "ci"], cwd=SUPERSPLAT_DIR)
    run_capture(["npm", "run", "build"], cwd=SUPERSPLAT_DIR)

    run_capture(["npm", "ci"], cwd=SUPERSPLAT_VIEWER_DIR)
    run_capture(["npm", "run", "build"], cwd=SUPERSPLAT_VIEWER_DIR)

    if not (SUPERSPLAT_EDITOR_DIST / "index.html").exists():
        raise PipelineError("SuperSplat Editor build completed but dist/index.html was not found.")
    if not (SUPERSPLAT_VIEWER_PUBLIC / "index.html").exists():
        raise PipelineError("SuperSplat Viewer build completed but public/index.html was not found.")


class SplatAssetHandler(SimpleHTTPRequestHandler):
    """Serve the editor, viewer and generated project files from one local origin."""

    server_version = "SplatStudio/1.0"

    def log_message(self, fmt, *args):
        # Keep the Streamlit terminal clean.
        return

    @staticmethod
    def _safe_resolve(root, relative):
        root = root.resolve()
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            return None
        return candidate

    def translate_path(self, request_path):
        path = unquote(urlsplit(request_path).path)

        routes = (
            ("/editor", SUPERSPLAT_EDITOR_DIST),
            ("/viewer", SUPERSPLAT_VIEWER_PUBLIC),
            ("/projects", WORKSPACE_ROOT),
        )

        for prefix, root in routes:
            if path == prefix or path.startswith(prefix + "/"):
                relative = path[len(prefix):].lstrip("/") or "index.html"
                resolved = self._safe_resolve(root, relative)
                if resolved is not None:
                    return str(resolved)

        # Deliberately map unknown requests to a file that does not exist.
        return str((THIRD_PARTY_ROOT / "__not_found__").resolve())

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


@st.cache_resource(show_spinner=False)
def start_asset_server():
    """Start one small local HTTP server for PlayCanvas and splat files."""
    mimetypes.add_type("application/octet-stream", ".ply")
    mimetypes.add_type("application/octet-stream", ".sog")
    mimetypes.add_type("application/octet-stream", ".spz")
    mimetypes.add_type("application/octet-stream", ".splat")
    mimetypes.add_type("application/octet-stream", ".ksplat")

    server = ThreadingHTTPServer(("127.0.0.1", 0), SplatAssetHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="splat-studio-assets")
    thread.start()
    return server, server.server_address[1]



def find_splat_transform():
    candidates = [
        shutil.which("splat-transform"),
        str(SPLAT_TRANSFORM_BIN) if SPLAT_TRANSFORM_BIN.exists() else None,
        str(SUPERSPLAT_DIR / "node_modules" / ".bin" / "splat-transform"),
        str(SUPERSPLAT_VIEWER_DIR / "node_modules" / ".bin" / "splat-transform"),
    ]
    for value in candidates:
        if value and Path(value).exists():
            return str(Path(value).resolve())
    return None


def ensure_splat_transform():
    existing = find_splat_transform()
    if existing:
        return existing
    status = playcanvas_status()
    if not status.get("npm") or not status.get("node_ok"):
        raise PipelineError("SPZ preview conversion requires Node.js 20.19+ and npm. Install/repair SuperSplat tools from Settings first.")
    SPLAT_TRANSFORM_ROOT.mkdir(parents=True, exist_ok=True)
    run_capture(["npm", "install", "--prefix", str(SPLAT_TRANSFORM_ROOT), "@playcanvas/splat-transform"])
    installed = find_splat_transform()
    if not installed:
        raise PipelineError("SplatTransform installed but its CLI could not be located.")
    return installed



SPLAT_UPRIGHT_VERSION = "supersplat_x180_v1"


def _x180_vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return value
    return [float(value[0]), -float(value[1]), -float(value[2])]


def _rotate_training_view_x180(view):
    if not isinstance(view, dict):
        return view
    updated = dict(view)
    if isinstance(updated.get("position"), (list, tuple)):
        updated["position"] = _x180_vector(updated["position"])
    if isinstance(updated.get("target"), (list, tuple)):
        updated["target"] = _x180_vector(updated["target"])
    return updated


def ensure_generated_splat_upright(source_path, project_dir=None, console=None):
    """Rotate Splat Studio's generated SPZ into SuperSplat's expected upright world orientation.

    This is deliberately non-destructive for user imports: only the generated
    project-level `splat.spz` with native-training metadata is corrected.
    """
    source_path = Path(source_path).resolve()
    if source_path.suffix.lower() != ".spz" or source_path.name.lower() != "splat.spz":
        return source_path

    project_dir = Path(project_dir or source_path.parent).resolve()
    meta_path = project_dir / "splat.training.json"
    training = load_training_meta(source_path)
    project_state = load_project_state(project_dir)

    if training.get("orientation_correction") == SPLAT_UPRIGHT_VERSION:
        return source_path

    backend_text = " ".join([
        str(training.get("backend") or ""),
        str(training.get("training_backend") or ""),
        str(project_state.get("training_engine") or ""),
        str(project_state.get("training_backend") or ""),
    ]).lower()
    if not any(token in backend_text for token in ("a1091150", "native_cpp_metal", "native c++/metal")):
        # Imported/foreign SPZ: leave exactly as supplied.
        return source_path

    cli = ensure_splat_transform()
    backup_dir = project_dir / ".splat_studio" / "orientation_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / "splat.pre-upright.spz"
    if not backup_path.exists():
        shutil.copy2(source_path, backup_path)

    temporary = source_path.with_name(".splat.upright.tmp.spz")
    temporary.unlink(missing_ok=True)

    if console is not None:
        console.write("[studio] Correcting generated splat orientation for SuperSplat (180° X-axis).", force=True)

    run_capture([
        cli,
        str(source_path),
        "-r", "180,0,0",
        str(temporary),
    ])

    if not temporary.exists() or temporary.stat().st_size <= 0:
        raise PipelineError("Splat orientation correction did not produce a valid SPZ.")

    temporary.replace(source_path)

    # Keep all recorded/recommended capture views in exactly the same world
    # orientation as the transformed splat.
    if isinstance(training.get("best_view"), dict):
        training["best_view"] = _rotate_training_view_x180(training["best_view"])

    for key in ("frame_summaries", "eval_frame_summaries"):
        records = training.get(key)
        if isinstance(records, list):
            for item in records:
                if isinstance(item, dict) and isinstance(item.get("view"), dict):
                    item["view"] = _rotate_training_view_x180(item["view"])

    training["orientation_correction"] = SPLAT_UPRIGHT_VERSION
    training["orientation_rotation_degrees"] = [180.0, 0.0, 0.0]
    training["orientation_backup"] = str(backup_path)

    if meta_path.exists() or training:
        try:
            meta_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
        except OSError:
            pass

    # Any previous PLY viewer cache was made from the old orientation.
    review_cache = project_dir / ".review_preview"
    if review_cache.exists():
        try:
            remove_path_robustly(review_cache)
        except Exception:
            pass

    save_project_state(
        project_dir,
        splat_orientation=SPLAT_UPRIGHT_VERSION,
        splat_orientation_rotation=[180.0, 0.0, 0.0],
    )
    return source_path


def viewer_compatible_preview_path(source_path):
    """Convert editor-only formats such as SPZ into a lightweight PLY cache for the embedded viewer."""
    source_path = Path(source_path).resolve()
    supported = {".ply", ".sog", ".compressed.ply", ".meta.json", ".lod-meta.json"}
    lower_name = source_path.name.lower()
    if source_path.suffix.lower() in {".ply", ".sog"} or lower_name.endswith((".compressed.ply", ".meta.json", ".lod-meta.json")):
        return source_path

    if source_path.suffix.lower() not in {".spz", ".splat", ".ksplat"}:
        return source_path

    preview_dir = source_path.parent / ".review_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    target = preview_dir / f"{source_path.stem}.viewer.ply"
    if target.exists() and target.stat().st_size > 0 and target.stat().st_mtime >= source_path.stat().st_mtime:
        return target

    cli = ensure_splat_transform()
    temporary = target.with_suffix(".tmp.ply")
    temporary.unlink(missing_ok=True)
    run_capture([cli, str(source_path), str(temporary)])
    if not temporary.exists() or temporary.stat().st_size <= 0:
        raise PipelineError("SplatTransform did not produce a usable PLY preview.")
    temporary.replace(target)
    return target


def resolve_best_view(project_dir, training):
    """Return V19.0 best-view metadata, deriving it once for V17 results without retraining."""
    training = training or {}
    existing = training.get("best_view")
    if isinstance(existing, dict) and existing.get("position") and existing.get("target"):
        return existing

    summaries = list(training.get("frame_summaries") or []) + list(training.get("eval_frame_summaries") or [])
    if not summaries:
        return None
    max_visible = max((int(item.get("final_visible_gaussians") or 0) for item in summaries), default=1)
    ranked = []
    for item in summaries:
        psnr = float(item.get("final_psnr") or 0.0)
        visible = int(item.get("final_visible_gaussians") or 0)
        score = psnr + 1.5 * (visible / max(1, max_visible))
        ranked.append((score, int(item.get("frame_index") or 0), psnr, visible))
    if not ranked:
        return None
    score, frame_index, psnr, visible = max(ranked)

    dataset = training.get("dataset")
    if not dataset:
        return None
    helper = r"""import json, sys\nfrom pathlib import Path\nimport numpy as np\nsys.path.insert(0, sys.argv[1])\nfrom colmap_360_dataset import load_colmap_scene\ndata=Path(sys.argv[2])\nfactor=int(sys.argv[3]); width=int(sys.argv[4]); height=int(sys.argv[5]); test_every=int(sys.argv[6]); normalize=sys.argv[7]=='1'; frame=int(sys.argv[8])\nscene=load_colmap_scene(data, factor, width, height, test_every=test_every, normalize_world_space=normalize)\ncam=next((c for c in scene.cameras if int(c.index)==frame), None)\nif cam is None: raise SystemExit(3)\nc2w=np.linalg.inv(np.asarray(cam.viewmat,dtype=np.float32)); pos=c2w[:3,3]; center=np.median(scene.points,axis=0)\ndef axis(v): return [float(v[0]), float(v[2]), float(-v[1])]\nfy=max(float(cam.K[1,1]),1e-6); fov=float(np.degrees(2*np.arctan(float(height)/(2*fy))))\nprint(json.dumps({'image_name':str(cam.image_name),'position':axis(pos),'target':axis(center),'fov':float(np.clip(fov,25,100))}))"""
    try:
        result = subprocess.run(
            [
                str(GSPLAT_METAL_PYTHON), "-c", helper, str(GSPLAT_METAL_TEST_DIR), str(dataset),
                str(int(training.get("data_factor") or 1)), str(int(training.get("width") or 480)), str(int(training.get("height") or 270)),
                str(int(training.get("test_every") or 8)), "1" if bool(training.get("normalize_world_space", True)) else "0", str(frame_index),
            ],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            return None
        view = json.loads(result.stdout.strip().splitlines()[-1])
        view.update({"frame_index": frame_index, "psnr": psnr, "visible_gaussians": visible, "score": score, "derived_from_legacy_summary": True})
        training["best_view"] = view
        meta_path = Path(project_dir) / "splat.training.json"
        if meta_path.exists():
            try:
                meta_path.write_text(json.dumps(training, indent=2), encoding="utf-8")
            except OSError:
                pass
        return view
    except Exception:
        return None


def best_view_settings_path(project_dir, training):
    best = resolve_best_view(project_dir, training) or {}
    position = best.get("position")
    target = best.get("target")
    if not (isinstance(position, list) and len(position) == 3 and isinstance(target, list) and len(target) == 3):
        return None
    settings = {
        "version": 2,
        "tonemapping": "none",
        "highPrecisionRendering": False,
        "background": {"color": [0.015, 0.02, 0.03]},
        "postEffectSettings": {
            "sharpness": {"enabled": True, "amount": 0.12},
            "bloom": {"enabled": False, "intensity": 1, "blurLevel": 2},
            "grading": {"enabled": False, "brightness": 0, "contrast": 1, "saturation": 1, "tint": [1, 1, 1]},
            "vignette": {"enabled": False, "intensity": 0.5, "inner": 0.3, "outer": 0.75, "curvature": 1},
            "fringing": {"enabled": False, "intensity": 0.5},
        },
        "animTracks": [],
        "cameras": [{
            "initial": {
                "position": [float(v) for v in position],
                "target": [float(v) for v in target],
                "fov": float(best.get("fov") or 60.0),
            }
        }],
        "annotations": [],
        "startMode": "default",
    }
    path = Path(project_dir) / ".review_preview" / "best_view.settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return path


def best_view_source_image(project_dir, training):
    image_name = str((resolve_best_view(project_dir, training) or {}).get("image_name") or "")
    if not image_name:
        return None
    roots = [
        Path(project_dir) / ".splat_studio_train_cache" / "undistorted" / "images",
        Path(project_dir) / NATIVE_TRAINING_DIRNAME / "dataset" / "images",
        Path(project_dir) / "images",
    ]
    for root in roots:
        candidate = root / image_name
        if candidate.exists():
            return candidate
    return None

def project_web_url(file_path, base_url):
    """Return a localhost URL for a file that lives under WORKSPACE_ROOT."""
    file_path = Path(file_path).resolve()
    try:
        relative = file_path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return None

    encoded = "/".join(quote(part) for part in relative.parts)
    return f"{base_url}/projects/{encoded}"


def editor_url_for(file_path, base_url):
    content_url = project_web_url(file_path, base_url)
    if not content_url:
        return f"{base_url}/editor/index.html"
    return f"{base_url}/editor/index.html?{urlencode({'load': content_url})}"


def viewer_url_for(file_path, base_url, show_ui=True, settings_path=None, nonce=None):
    file_path = Path(file_path)
    content_url = project_web_url(file_path, base_url)
    if not content_url:
        return f"{base_url}/viewer/index.html"

    params = {"content": content_url}
    if settings_path:
        settings_url = project_web_url(settings_path, base_url)
        if settings_url:
            params["settings"] = settings_url
    if not show_ui:
        params["noui"] = "1"
    if nonce is not None:
        params["splatstudio"] = str(nonce)
    return f"{base_url}/viewer/index.html?{urlencode(params)}"


def supported_splat_files(project_dir):
    """List files that the local SuperSplat tools can open."""
    suffixes = (
        ".ply",
        ".compressed.ply",
        ".sog",
        ".splat",
        ".ksplat",
        ".spz",
        ".meta.json",
        ".lod-meta.json",
    )
    files = []
    if project_dir.exists():
        for item in project_dir.rglob("*"):
            if not item.is_file():
                continue
            try:
                relative = item.relative_to(project_dir)
                if any(part.startswith(".") for part in relative.parts[:-1]):
                    continue
            except ValueError:
                pass
            if any(item.name.lower().endswith(suffix) for suffix in suffixes):
                files.append(item)
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def save_imported_splat(uploaded_splat, project_dir):
    imports_dir = project_dir / "imports"
    imports_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", uploaded_splat.name).strip() or "imported.ply"
    destination = imports_dir / safe_name
    destination.write_bytes(uploaded_splat.getbuffer())
    return destination



def file_sha1(path, chunk_size=4 * 1024 * 1024):
    digest = hashlib.sha1()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()



PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def source_photo_files(path):
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(
        item for item in path.iterdir()
        if item.is_file()
        and item.suffix.lower() in PHOTO_EXTENSIONS
        and not item.name.startswith(".")
    )


def source_fingerprint(path):
    """Stable reconstruction signature for either one video file or a photo directory."""
    path = Path(path)
    if path.is_file():
        return file_sha1(path)

    digest = hashlib.sha1()
    files = source_photo_files(path)
    digest.update(f"photos:{len(files)}".encode("utf-8"))
    for item in files:
        stat = item.stat()
        digest.update(item.name.encode("utf-8", errors="replace"))
        digest.update(str(int(stat.st_size)).encode("ascii"))
        digest.update(str(int(stat.st_mtime_ns)).encode("ascii"))
    return digest.hexdigest()


def prepare_photo_frames(
    source_dir,
    images_dir,
    progress,
    console,
    target_frames,
    smart_select=True,
    progress_range=None,
    progress_label=None,
):
    """Normalize uploaded still photos into COLMAP-ready JPG frames."""
    from PIL import Image, ImageOps

    source_dir = Path(source_dir)
    images_dir = Path(images_dir)
    candidates = source_photo_files(source_dir)
    if not candidates:
        raise PipelineError("No supported photos were found in the uploaded photo set.")

    start, end = progress_range if progress_range is not None else EXTRACT_RANGE
    stage_label = str(progress_label or "Preparing photos")
    limit = max(2, int(target_frames or len(candidates)))

    progress.update(
        start,
        stage_label,
        f"{len(candidates):,} uploaded photos · preparing up to {min(limit, len(candidates)):,}",
        force=True,
    )

    selected = candidates
    if len(candidates) > limit:
        if smart_select:
            report_path = source_dir.parent / ".splat_studio" / "photo_curation.json"
            selected = _select_smart_frames(candidates, limit, report_path, console)
        else:
            # Preserve broad coverage even without the quality scorer.
            indexes = np.linspace(0, len(candidates) - 1, limit, dtype=int)
            selected = [candidates[int(index)] for index in indexes]

    remove_path_robustly(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    for index, photo in enumerate(selected, start=1):
        destination = images_dir / f"{index:05d}.jpg"
        try:
            with Image.open(photo) as image:
                image = ImageOps.exif_transpose(image)
                if image.mode != "RGB":
                    if "A" in image.getbands():
                        base = Image.new("RGB", image.size, (255, 255, 255))
                        alpha = image.getchannel("A")
                        base.paste(image.convert("RGB"), mask=alpha)
                        image = base
                    else:
                        image = image.convert("RGB")
                image.save(destination, format="JPEG", quality=95, subsampling=0)
        except Exception as exc:
            console.write(f"[studio] Skipping unreadable photo {photo.name}: {exc}", force=True)
            continue

        completed += 1
        ratio = completed / max(1, len(selected))
        progress.update(
            start + (end - start) * ratio,
            stage_label,
            f"{completed:,}/{len(selected):,} photos prepared",
        )

    if completed < 2:
        raise PipelineError("Fewer than two usable photos could be prepared.")

    progress.update(
        end,
        stage_label if progress_label else "Photos ready",
        f"{completed:,} photos ready for COLMAP",
        force=True,
    )
    console.write(
        f"[studio] Photo source: {len(candidates):,} uploaded → {completed:,} COLMAP frames.",
        force=True,
    )
    return completed, 0.0


def reconstruction_config_payload(source_path):
    return {
        "source_sha1": source_fingerprint(source_path),
        "source_type": "photos" if Path(source_path).is_dir() else "video",
        "target_frames": st.session_state.cfg_target_frames,
        "overlap": st.session_state.cfg_overlap,
        "max_features": st.session_state.cfg_max_features,
        "max_image_size": st.session_state.cfg_max_image_size,
        "shared_camera": st.session_state.cfg_shared_camera,
        "camera_model": st.session_state.cfg_camera_model,
        "quadratic_overlap": st.session_state.cfg_quadratic_overlap,
        "guided_matching": st.session_state.cfg_guided_matching,
        "robust_sift": st.session_state.cfg_robust_sift,
        "smart_frames": st.session_state.cfg_smart_frames,
    }


def config_hash(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def reconstruction_meta_path(project_dir):
    return Path(project_dir) / ".splat_reconstruction.json"


def sparse_model_dir(project_dir):
    sparse = Path(project_dir) / "sparse"
    if not sparse.exists():
        return None
    models = sorted([p for p in sparse.iterdir() if p.is_dir()])
    return models[0] if models else None


def can_reuse_camera_solve(project_dir, expected_hash):
    meta_path = reconstruction_meta_path(project_dir)
    if not meta_path.exists() or not (Path(project_dir) / "database.db").exists():
        return False
    if sparse_model_dir(project_dir) is None or not (Path(project_dir) / "images").exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("config_hash") == expected_hash


def save_reconstruction_meta(project_dir, payload, frame_count, sample_fps):
    meta = {
        "config_hash": config_hash(payload),
        "config": payload,
        "frame_count": frame_count,
        "sample_fps": sample_fps,
        "created_at": time.time(),
    }
    reconstruction_meta_path(project_dir).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_reconstruction_meta(project_dir):
    path = reconstruction_meta_path(project_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


_HEADER_STATUS_SLOT = None


def current_total_runtime_seconds(project_dir=None):
    """Total project wall-clock runtime across restarts/continues."""
    project_dir = Path(project_dir) if project_dir else current_project_dir()
    base = float(st.session_state.get("run_accumulated_runtime_seconds") or 0.0)
    state = load_project_state(project_dir) if project_dir else {}
    durable = float(state.get("total_runtime_seconds") or 0.0)
    started = st.session_state.get("run_started_ts")
    if started:
        endpoint = st.session_state.get("run_finished_ts") or time.time()
        segment_total = base + max(0.0, float(endpoint) - float(started))
        return max(durable, segment_total)
    return max(base, durable)


def persist_current_runtime(project_dir, **extra):
    total_runtime = current_total_runtime_seconds(project_dir)
    return save_project_state(project_dir, total_runtime_seconds=total_runtime, **extra)


def render_tip_carousel(context="run"):
    if context == "prepare":
        tips = [
            "A slow orbit with sharp frames usually beats a longer capture with motion blur.",
            "Keep the subject visible from frame to frame; COLMAP needs overlapping visual landmarks.",
            "Balanced is the safest first build. Increase detail after you know the camera solve is healthy.",
            "Stable exposure and lighting reduce false matches from moving reflections and shadows.",
        ]
    else:
        tips = [
            "Snapshots preserve the Gaussian model, Adam optimizer, refinement state, sampler position and total runtime — Continue resumes the actual training state.",
            "Unexpected native-training errors trigger a two-minute auto-resume countdown; each retry lowers GPU pressure and continues from the newest snapshot.",
            "Gaussian refinement can make individual training steps slower later in the run because the model contains more splats.",
            "The full pipeline log is written to disk even while the Live console is collapsed.",
        ]
    spans = "".join(f"<span>{html.escape(tip)}</span>" for tip in tips)
    st.markdown(f'<div class="tip-carousel">{spans}</div>', unsafe_allow_html=True)


def refresh_header_status():
    """Update the top-right status row during a blocking native pipeline run."""
    global _HEADER_STATUS_SLOT
    if _HEADER_STATUS_SLOT is not None:
        try:
            render_header_status(_HEADER_STATUS_SLOT)
        except Exception:
            pass


def project_state_path(project_dir):
    return Path(project_dir) / PROJECT_STATE_FILENAME


def load_project_state(project_dir):
    path = project_state_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_project_state(project_dir, **updates):
    """Persist durable project lifecycle separately from Streamlit session state."""
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    state = load_project_state(project_dir)
    state.update(updates)
    state["project_name"] = project_dir.name
    state["updated_at"] = time.time()
    path = project_state_path(project_dir)
    temp = path.with_suffix(".tmp")
    try:
        temp.write_text(json.dumps(state, indent=2, sort_keys=True, default=str), encoding="utf-8")
        temp.replace(path)
    except OSError:
        pass
    return state


def training_checkpoint_info(project_dir):
    """Return the latest durable native-Metal snapshot, if one exists."""
    project_dir = Path(project_dir)
    root = project_dir / NATIVE_TRAINING_DIRNAME / "checkpoints"
    pointer = root / "latest.json"
    checkpoint_dir = None

    if pointer.exists():
        try:
            pointer_data = json.loads(pointer.read_text(encoding="utf-8"))
            candidate = root / str(pointer_data.get("directory") or "")
            if candidate.is_dir():
                checkpoint_dir = candidate
        except (OSError, json.JSONDecodeError):
            checkpoint_dir = None

    if checkpoint_dir is None:
        candidates = sorted(
            (path for path in root.glob("step_*") if path.is_dir() and (path / "state.json").exists()),
            key=lambda path: path.name,
        ) if root.exists() else []
        checkpoint_dir = candidates[-1] if candidates else None

    if checkpoint_dir is None:
        return None

    state_path = checkpoint_dir / "state.json"
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    try:
        mtime = state_path.stat().st_mtime
    except OSError:
        mtime = time.time()

    return {
        "step": int(data.get("step") or 0),
        "total_steps": int(data.get("total_steps") or 0),
        "gaussians": int(data.get("gaussians") or 0),
        "project_runtime_seconds": float(data.get("elapsed_seconds") or 0.0),
        "created_at": float(data.get("created_at") or mtime),
        "mtime": float(mtime),
        "reason": str(data.get("reason") or "interval"),
        "stage_index": 10,
        "path": checkpoint_dir,
        "root": root,
    }




def generated_output_for_project(project_dir):
    project_dir = Path(project_dir)
    state = load_project_state(project_dir)
    recorded = state.get("output_path")
    candidates = [Path(recorded)] if recorded else []
    candidates.extend([project_dir / "splat.spz", project_dir / "splat.ply", project_dir / "splat.splat"])
    training_meta = project_dir / "splat.training.json"

    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            if training_meta.exists() or state.get("status") == "complete":
                return candidate
    return None


def inspect_project_progress(project_dir):
    """Infer project state from durable files first, then the state manifest."""
    project_dir = Path(project_dir)
    state = load_project_state(project_dir)
    source = latest_source_for_project(project_dir)
    output = generated_output_for_project(project_dir)

    if output:
        return {
            "status": "complete",
            "label": "Finished",
            "detail": "Gaussian output is complete and ready to review.",
            "output": output,
            "source": source,
            "state": state,
            "checkpoint": None,
        }

    checkpoint = training_checkpoint_info(project_dir)
    if checkpoint:
        step = checkpoint.get("step") or 0
        total = checkpoint.get("total_steps") or 0
        detail = (
            f"Recovery snapshot at training step {step:,} of {total:,}."
            if total
            else f"Recovery snapshot saved at training step {step:,}."
        )
        return {
            "status": "checkpoint",
            "label": "Continue",
            "detail": detail,
            "output": None,
            "source": source,
            "state": state,
            "checkpoint": checkpoint,
        }

    if source and sparse_model_dir(project_dir) is not None and (project_dir / "database.db").exists():
        return {
            "status": "camera",
            "label": "Continue",
            "detail": "Camera reconstruction is present. Splat Studio can continue into Gaussian training.",
            "output": None,
            "source": source,
            "state": state,
            "checkpoint": None,
        }

    if source and state.get("status") in {"running", "error", "cancelled", "interrupted"}:
        return {
            "status": "incomplete",
            "label": "Continue",
            "detail": "This project was started previously and can continue from its saved source and compatible stages.",
            "output": None,
            "source": source,
            "state": state,
            "checkpoint": None,
        }

    return {
        "status": "new",
        "label": "New",
        "detail": "No recoverable reconstruction progress has been found yet.",
        "output": None,
        "source": source,
        "state": state,
        "checkpoint": None,
    }


def total_system_memory_bytes():
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return int(result.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    try:
        import psutil
        return int(psutil.virtual_memory().total)
    except Exception:
        return 0


def load_run_history():
    if not RUN_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(RUN_HISTORY_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def append_run_history(record):
    history = load_run_history()
    history.append(record)
    history = history[-40:]
    try:
        RUN_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        temp = RUN_HISTORY_FILE.with_suffix(".tmp")
        temp.write_text(json.dumps(history, indent=2, default=str), encoding="utf-8")
        temp.replace(RUN_HISTORY_FILE)
    except OSError:
        pass


def config_work_score(config):
    frames = max(1, int(config.get("target_frames", 300)))
    overlap = max(1, int(config.get("overlap", 12)))
    features = max(512, int(config.get("max_features", 4096)))
    image_size = max(512, int(config.get("max_image_size", 1920)))
    iterations = max(1, int(config.get("iterations", 2500)))
    train_side = max(64, int(config.get("mlx_train_max_side", 256)))
    init_points = max(1, int(config.get("mlx_max_points", 3500)))
    max_gaussians = max(init_points, int(config.get("mlx_max_gaussians", 12000)))
    batch = max(1, int(config.get("mlx_batch_size", 1)))

    camera = (frames / 300) * (overlap / 12) ** 0.8 * (features / 4096) ** 0.7 * (image_size / 1920) ** 1.35
    training = (
        (iterations / 2500)
        * (train_side / 256) ** 2
        * batch
        * (0.45 + 0.25 * (init_points / 3500) + 0.30 * (max_gaussians / 12000))
    )
    if config.get("robust_sift"):
        camera *= 1.35
    if config.get("guided_matching"):
        camera *= 1.15
    if config.get("mlx_antialiased"):
        training *= 1.18
    if float(config.get("mlx_ssim_weight", 0.20) or 0) > 0:
        training *= 1.10
    if config.get("mlx_refine", True):
        training *= 1.12
    return max(0.05, 0.42 * camera + 0.58 * training)


def resource_estimate(config, project_dir=None, source_path=None):
    """Estimate peak resources and runtime with headroom; results are planning guidance."""
    frames = max(1, int(config.get("target_frames", 300)))
    overlap = max(1, int(config.get("overlap", 12)))
    features = max(512, int(config.get("max_features", 4096)))
    image_size = max(512, int(config.get("max_image_size", 1920)))
    steps = max(1, int(config.get("iterations", 2500)))
    side = max(64, int(config.get("mlx_train_max_side", 256)))
    max_points = max(1, int(config.get("mlx_max_points", 3500)))
    max_gaussians = max(max_points, int(config.get("mlx_max_gaussians", 12000)))
    batch = max(1, int(config.get("mlx_batch_size", 1)))
    antialiased = bool(config.get("mlx_antialiased", False))
    refine = bool(config.get("mlx_refine", True))
    ssim = float(config.get("mlx_ssim_weight", 0.20) or 0.0)

    descriptor_bytes = frames * features * 128 * 4
    colmap_gb = 0.8 + descriptor_bytes * 2.4 / (1024 ** 3) + 0.55 * (image_size / 1920) ** 2

    mlx_factor = (side / 256) ** 2 * batch
    mlx_population = 0.7 * (max_points / 3500) + 2.4 * (max_gaussians / 12000)
    mlx_gb = 1.0 + mlx_factor * mlx_population
    if antialiased:
        mlx_gb *= 1.18
    if ssim > 0:
        mlx_gb *= 1.10
    if refine:
        mlx_gb *= 1.08

    estimated_peak_gb = max(colmap_gb, mlx_gb)
    total_mem_bytes = total_system_memory_bytes()
    total_mem_gb = total_mem_bytes / (1024 ** 3) if total_mem_bytes else 0.0
    safe_mem_gb = total_mem_gb * 0.68 if total_mem_gb else 0.0
    mem_ratio = estimated_peak_gb / safe_mem_gb if safe_mem_gb else 0.0

    metal_pressure = (
        (side / 256) ** 2
        * batch
        * (max_gaussians / 12000)
        * (1.18 if antialiased else 1.0)
        * (1.10 if refine else 1.0)
    )

    if mem_ratio >= 1.0 or metal_pressure >= 3.0:
        risk, risk_label = "bad", "High risk"
    elif mem_ratio >= 0.72 or metal_pressure >= 1.75:
        risk, risk_label = "warn", "Heavy"
    else:
        risk, risk_label = "good", "Comfortable"

    extraction_s = 0.05 * frames
    features_s = 0.06 * frames * (features / 4096) * (image_size / 1920) ** 1.45
    matching_s = 0.015 * frames * overlap * (features / 4096) ** 1.2 * (image_size / 1920) ** 0.65
    mapping_s = max(25.0, 0.23 * frames + 0.006 * frames * overlap)
    if config.get("robust_sift"):
        features_s *= 1.45
    if config.get("guided_matching"):
        matching_s *= 1.25
    if config.get("quadratic_overlap", True):
        matching_s *= 1.12

    training_s = steps * 0.35 * (side / 256) ** 2 * batch * (0.72 + 0.28 * (max_points / 3500))
    if antialiased:
        training_s *= 1.18
    if ssim > 0:
        training_s *= 1.10
    if refine:
        training_s *= 1.12

    heuristic_seconds = extraction_s + features_s + matching_s + mapping_s + training_s
    current_score = config_work_score(config)
    memory_key = round(total_mem_gb, 1)

    ratios = []
    for record in load_run_history():
        if record.get("status") != "success" or record.get("machine") != platform.machine():
            continue
        if record.get("memory_gb") and abs(float(record["memory_gb"]) - memory_key) > 1.0:
            continue
        old_est = float(record.get("heuristic_seconds") or 0)
        actual = float(record.get("runtime_seconds") or 0)
        if old_est > 30 and actual > 30:
            ratios.append(actual / old_est)

    learned_runs = len(ratios)
    multiplier = max(0.45, min(3.5, statistics.median(ratios))) if ratios else 1.0
    estimated_seconds = heuristic_seconds * multiplier

    checkpoint = training_checkpoint_info(project_dir) if project_dir else None
    continue_seconds = None
    if checkpoint and checkpoint.get("step") and checkpoint.get("total_steps"):
        remaining = max(0.0, 1.0 - checkpoint["step"] / checkpoint["total_steps"])
        continue_seconds = training_s * remaining * multiplier + 20.0

    source_bytes = Path(source_path).stat().st_size if source_path and Path(source_path).exists() else 0
    frame_disk_gb = frames * 1.15 * (image_size / 1920) ** 2 / 1024
    estimated_disk_gb = source_bytes / (1024 ** 3) + frame_disk_gb * 2.1 + 0.6
    try:
        free_disk_gb = shutil.disk_usage(project_dir or WORKSPACE_ROOT).free / (1024 ** 3)
    except OSError:
        free_disk_gb = 0.0

    previous_failures = [
        float(record.get("work_score") or 0)
        for record in load_run_history()
        if record.get("resource_failure") and record.get("machine") == platform.machine()
    ]
    previous_limit_warning = bool(previous_failures and current_score >= min(previous_failures) * 0.90)

    return {
        "risk": risk,
        "risk_label": risk_label,
        "estimated_peak_gb": estimated_peak_gb,
        "system_memory_gb": total_mem_gb,
        "safe_memory_gb": safe_mem_gb,
        "memory_ratio": mem_ratio,
        "metal_pressure": metal_pressure,
        "estimated_seconds": estimated_seconds,
        "continue_seconds": continue_seconds,
        "estimated_disk_gb": estimated_disk_gb,
        "free_disk_gb": free_disk_gb,
        "learned_runs": learned_runs,
        "heuristic_seconds": heuristic_seconds,
        "work_score": current_score,
        "previous_limit_warning": previous_limit_warning,
    }


def record_run_history(status, error=None):
    started = st.session_state.get("run_started_ts")
    finished = st.session_state.get("run_finished_ts") or time.time()
    runtime = max(0.0, finished - started) if started else 0.0
    config = st.session_state.get("run_config") or {}
    estimate = resource_estimate(config, run_project_dir(), st.session_state.get("run_source_path"))
    text = str(error or "")
    append_run_history({
        "timestamp": time.time(),
        "status": status,
        "runtime_seconds": runtime,
        "heuristic_seconds": estimate["heuristic_seconds"],
        "work_score": estimate["work_score"],
        "machine": platform.machine(),
        "memory_gb": estimate["system_memory_gb"],
        "resource_failure": "metal::malloc" in text.lower() or "resource limit" in text.lower(),
        "error": text[-1200:],
        "config": config,
    })


def session_run_config(source_path):
    payload = reconstruction_config_payload(source_path)
    return {
        **payload,
        "iterations": st.session_state.cfg_iterations,
        "means_lr": st.session_state.cfg_means_lr,
        "mlx_batch_size": st.session_state.cfg_mlx_batch_size,
        "mlx_max_points": st.session_state.cfg_mlx_max_points,
        "mlx_max_gaussians": st.session_state.cfg_mlx_max_gaussians,
        "mlx_train_max_side": st.session_state.cfg_mlx_train_max_side,
        "mlx_ssim_weight": st.session_state.cfg_mlx_ssim_weight,
        "mlx_refine": st.session_state.cfg_mlx_refine,
        "mlx_antialiased": st.session_state.cfg_mlx_antialiased,
        "checkpoint_every": st.session_state.cfg_checkpoint_every,
        "preview_every": st.session_state.cfg_preview_every,
        "auto_resume": st.session_state.cfg_auto_resume,
        "reuse_camera": st.session_state.cfg_reuse_camera,
        "smart_frames": st.session_state.cfg_smart_frames,
        "ai_depth": st.session_state.cfg_ai_depth,
        "ai_depth_points": st.session_state.cfg_ai_depth_points,
        "ai_mask_sky": st.session_state.cfg_ai_mask_sky,
        "ai_mask_dynamic": st.session_state.cfg_ai_mask_dynamic,
    }


def planning_config_from_session():
    return {
        "target_frames": st.session_state.cfg_target_frames,
        "overlap": st.session_state.cfg_overlap,
        "max_features": st.session_state.cfg_max_features,
        "max_image_size": st.session_state.cfg_max_image_size,
        "shared_camera": st.session_state.cfg_shared_camera,
        "camera_model": st.session_state.cfg_camera_model,
        "quadratic_overlap": st.session_state.cfg_quadratic_overlap,
        "guided_matching": st.session_state.cfg_guided_matching,
        "robust_sift": st.session_state.cfg_robust_sift,
        "auto_camera_rescue": st.session_state.cfg_auto_camera_rescue,
        "iterations": st.session_state.cfg_iterations,
        "means_lr": st.session_state.cfg_means_lr,
        "mlx_batch_size": st.session_state.cfg_mlx_batch_size,
        "mlx_max_points": st.session_state.cfg_mlx_max_points,
        "mlx_max_gaussians": st.session_state.cfg_mlx_max_gaussians,
        "mlx_train_max_side": st.session_state.cfg_mlx_train_max_side,
        "mlx_ssim_weight": st.session_state.cfg_mlx_ssim_weight,
        "mlx_refine": st.session_state.cfg_mlx_refine,
        "mlx_antialiased": st.session_state.cfg_mlx_antialiased,
        "checkpoint_every": st.session_state.cfg_checkpoint_every,
        "preview_every": st.session_state.cfg_preview_every,
        "auto_resume": st.session_state.cfg_auto_resume,
        "reuse_camera": st.session_state.cfg_reuse_camera,
        "smart_frames": st.session_state.cfg_smart_frames,
        "ai_depth": st.session_state.cfg_ai_depth,
        "ai_depth_points": st.session_state.cfg_ai_depth_points,
        "ai_mask_sky": st.session_state.cfg_ai_mask_sky,
        "ai_mask_dynamic": st.session_state.cfg_ai_mask_dynamic,
    }


def queue_saved_run_config(config):
    """Restore saved widget settings on the next Prepare render, before widgets exist."""
    if config:
        st.session_state["_pending_saved_run_config"] = dict(config)



def apply_saved_run_config(config):
    mapping = {
        "target_frames": "cfg_target_frames",
        "overlap": "cfg_overlap",
        "max_features": "cfg_max_features",
        "max_image_size": "cfg_max_image_size",
        "shared_camera": "cfg_shared_camera",
        "camera_model": "cfg_camera_model",
        "quadratic_overlap": "cfg_quadratic_overlap",
        "guided_matching": "cfg_guided_matching",
        "robust_sift": "cfg_robust_sift",
        "auto_camera_rescue": "cfg_auto_camera_rescue",
        "iterations": "cfg_iterations",
        "means_lr": "cfg_means_lr",
        "mlx_batch_size": "cfg_mlx_batch_size",
        "mlx_max_points": "cfg_mlx_max_points",
        "mlx_max_gaussians": "cfg_mlx_max_gaussians",
        "mlx_train_max_side": "cfg_mlx_train_max_side",
        "mlx_ssim_weight": "cfg_mlx_ssim_weight",
        "mlx_refine": "cfg_mlx_refine",
        "mlx_antialiased": "cfg_mlx_antialiased",
        "checkpoint_every": "cfg_checkpoint_every",
        "preview_every": "cfg_preview_every",
        "auto_resume": "cfg_auto_resume",
        "reuse_camera": "cfg_reuse_camera",
        "smart_frames": "cfg_smart_frames",
        "ai_depth": "cfg_ai_depth",
        "ai_depth_points": "cfg_ai_depth_points",
        "ai_mask_sky": "cfg_ai_mask_sky",
        "ai_mask_dynamic": "cfg_ai_mask_dynamic",
    }
    for key, state_key in mapping.items():
        if key in config:
            st.session_state[state_key] = config[key]


def launch_project_run(project_dir, source_path, run_config, continue_mode=False):
    project_dir = Path(project_dir)
    source_path = Path(source_path)
    save_settings()

    previous_state = load_project_state(project_dir)
    accumulated_runtime = 0.0
    if continue_mode:
        checkpoint_runtime = training_checkpoint_info(project_dir)
        accumulated_runtime = max(
            float(previous_state.get("total_runtime_seconds") or 0.0),
            float((checkpoint_runtime or {}).get("project_runtime_seconds") or 0.0),
        )
        # Migration path for older projects: they stored started_at/updated_at
        # but not an explicit accumulated runtime yet.
        if accumulated_runtime <= 0 and previous_state.get("started_at"):
            end_stamp = float(previous_state.get("updated_at") or time.time())
            accumulated_runtime = max(0.0, end_stamp - float(previous_state.get("started_at")))
    st.session_state.run_accumulated_runtime_seconds = accumulated_runtime

    log_path = project_dir / "pipeline.log"
    if continue_mode and log_path.exists():
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n===== CONTINUE {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    else:
        log_path.write_text("", encoding="utf-8")

    # run_config is authoritative for this run. Widget restoration is deferred
    # until the next Prepare page render to comply with Streamlit state rules.
    st.session_state.run_config = dict(run_config)
    st.session_state.run_project_dir = str(project_dir.resolve())
    st.session_state.run_source_path = str(source_path.resolve())
    st.session_state.run_status = "running"
    st.session_state.run_launch_pending = True
    st.session_state.run_error = None
    st.session_state.run_traceback = None
    st.session_state.run_started_ts = time.time()
    st.session_state.run_finished_ts = None
    st.session_state.run_progress = 0.0
    st.session_state.run_eta_seconds = None
    st.session_state.run_eta_calc_ts = 0.0
    st.session_state.run_native_eta_seconds = None
    st.session_state.run_native_eta_updated_ts = 0.0
    st.session_state.run_label = "Preparing project"
    st.session_state.run_detail = "Continuing saved project" if continue_mode else "Starting reconstruction"
    st.session_state.run_stage_index = 1
    st.session_state.run_last_activity_ts = time.time()
    st.session_state.run_log_path = str(log_path)
    st.session_state.run_stats = {}
    st.session_state.page = "run"

    save_project_state(
        project_dir,
        status="running",
        phase="reconstruct",
        source_path=str(source_path.resolve()),
        run_config=dict(run_config),
        started_at=time.time(),
        continue_mode=bool(continue_mode),
        total_runtime_seconds=accumulated_runtime,
        last_error=None,
    )


def render_resource_planner(config, project_dir, source_path):
    estimate = resource_estimate(config, project_dir, source_path)
    meter = min(100.0, max(4.0, estimate["memory_ratio"] * 100.0 if estimate["system_memory_gb"] else estimate["metal_pressure"] * 30.0))
    runtime_text = format_duration(estimate["estimated_seconds"])
    if estimate["continue_seconds"] is not None:
        runtime_text = f"{format_duration(estimate['continue_seconds'])} to continue"
    confidence = f"calibrated from {estimate['learned_runs']} completed run(s)" if estimate["learned_runs"] else "initial planning estimate"

    st.markdown(
        f"""
        <div class="resource-card">
          <div class="resource-top">
            <div class="resource-title">Resource planner</div>
            <div class="resource-grade {estimate['risk']}">{estimate['risk_label']}</div>
          </div>
          <div class="summary-grid">
            <div class="summary-box"><div class="summary-k">Estimated runtime</div><div class="summary-v">{html.escape(runtime_text)}</div></div>
            <div class="summary-box"><div class="summary-k">Peak unified memory</div><div class="summary-v">~{estimate['estimated_peak_gb']:.1f} GB</div></div>
            <div class="summary-box"><div class="summary-k">Detected memory</div><div class="summary-v">{f"{estimate['system_memory_gb']:.0f} GB" if estimate['system_memory_gb'] else "Unknown"}</div></div>
            <div class="summary-box"><div class="summary-k">Working disk</div><div class="summary-v">~{estimate['estimated_disk_gb']:.1f} GB</div></div>
          </div>
          <div class="resource-meter"><div class="resource-fill {estimate['risk']}" style="width:{meter:.1f}%"></div></div>
          <div class="resource-note">Planning estimate only · {html.escape(confidence)} · Metal pressure index {estimate['metal_pressure']:.2f}. Splat Studio keeps headroom for macOS and unified-memory graphics work.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    warnings = []
    if estimate["previous_limit_warning"]:
        warnings.append("This Mac has previously hit a Metal resource limit at a similar or lighter workload.")
    if estimate["risk"] == "bad":
        warnings.append("These settings are likely outside the comfortable resource envelope. Reduce Training render resolution, Maximum Gaussians or Batch size.")
    elif estimate["risk"] == "warn":
        warnings.append("These settings are heavy. Keep other memory-intensive apps closed and expect slower late-stage training.")
    if estimate["free_disk_gb"] and estimate["estimated_disk_gb"] > estimate["free_disk_gb"] * 0.70:
        warnings.append(f"Estimated working data is large relative to the {estimate['free_disk_gb']:.1f} GB currently free.")

    if warnings:
        if estimate["risk"] == "bad" or estimate["previous_limit_warning"]:
            st.error(" ".join(warnings))
        else:
            st.warning(" ".join(warnings))
    else:
        st.caption("Resource estimate looks comfortable for the detected machine. Actual MLX/Metal usage still varies with scene complexity.")


def analyze_sparse_model(project_dir):
    """Best-effort summary of the strongest COLMAP sparse model."""
    sparse = Path(project_dir) / "sparse"
    colmap = shutil.which("colmap")
    if not sparse.exists() or not colmap:
        return {}

    models = [item for item in sparse.iterdir() if item.is_dir()]
    if not models:
        return {}

    patterns = {
        "registered_images": [r"Registered images\s*:\s*(\d+)", r"Images\s*:\s*(\d+)"],
        "points": [r"Points\s*:\s*(\d+)", r"3D points\s*:\s*(\d+)"],
        "observations": [r"Observations\s*:\s*(\d+)"],
        "reprojection_error": [r"Mean reprojection error\s*:\s*([\d.]+)"],
    }

    best = {}
    best_registered = -1
    for model in models:
        try:
            result = subprocess.run(
                [colmap, "model_analyzer", "--path", str(model.resolve())],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue

        text = (result.stdout or "") + "\n" + (result.stderr or "")
        metrics = {"model": model.name}
        for key, candidates in patterns.items():
            for pattern in candidates:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    metrics[key] = match.group(1)
                    break
        try:
            registered = int(metrics.get("registered_images", 0))
        except (TypeError, ValueError):
            registered = 0
        if registered > best_registered:
            best = metrics
            best_registered = registered
    return best


def _metric_int(metrics, key):
    try:
        return int(metrics.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def camera_solve_health(project_dir, frame_count):
    metrics = analyze_sparse_model(project_dir)
    registered = _metric_int(metrics, "registered_images")
    points = _metric_int(metrics, "points")
    frame_count = max(1, int(frame_count or 0))
    ratio = registered / frame_count
    if registered < 12 or ratio < 0.08 or points < 50:
        level = "bad"
        message = "Camera solve is too sparse for a useful Gaussian reconstruction."
    elif ratio < 0.35 or points < 500:
        level = "warn"
        message = "Camera solve is usable but weak; missing views or geometry may be visible."
    else:
        level = "good"
        message = "Camera coverage is healthy enough to continue into Gaussian training."
    return {
        "level": level,
        "message": message,
        "registered": registered,
        "points": points,
        "frame_count": frame_count,
        "registration_ratio": ratio,
        "metrics": metrics,
    }


def repair_colmap_connectivity(project_dir, images_dir, progress, console, sequential_overlap):
    colmap = shutil.which("colmap")
    if not colmap:
        return
    database_path = Path(project_dir) / "database.db"
    sparse_dir = Path(project_dir) / "sparse"
    matcher_help = colmap_help(colmap, "sequential_matcher")
    overlap_flag = choose_supported_option(matcher_help, ["--SequentialMatching.overlap", "--SequentialPairing.overlap"])
    quadratic_flag = choose_supported_option(matcher_help, ["--SequentialMatching.quadratic_overlap", "--SequentialPairing.quadratic_overlap"])
    guided_flag = choose_supported_option(matcher_help, ["--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"])
    recovery_overlap = max(24, min(64, int(sequential_overlap) * 2))
    command = [colmap, "sequential_matcher", "--database_path", str(database_path.resolve())]
    if overlap_flag:
        command.extend([overlap_flag, str(recovery_overlap)])
    if quadratic_flag:
        command.extend([quadratic_flag, "1"])
    if guided_flag:
        command.extend([guided_flag, "1"])
    console.write(f"[studio] Weak camera coverage; repair pass overlap={recovery_overlap}, quadratic=on, guided=on", force=True)
    progress.update(53, "Repairing camera solve", "Adding wider video-frame correspondences", force=True)
    run_streaming(command, console, on_line=make_stage_parser(progress, "Connectivity repair", 44, 49), project_dir=project_dir, on_heartbeat=getattr(progress, "heartbeat", None))
    remove_path_robustly(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    progress.update(56, "Repairing camera solve", "Re-running sparse mapping with added matches", force=True)
    run_streaming([
        colmap, "mapper",
        "--database_path", str(database_path.resolve()),
        "--image_path", str(Path(images_dir).resolve()),
        "--output_path", str(sparse_dir.resolve()),
    ], console, on_line=make_stage_parser(progress, "Sparse repair", 49, 58), project_dir=project_dir, on_heartbeat=getattr(progress, "heartbeat", None))



def update_camera_run_stats(project_dir, frame_count, sample_fps, health=None, rescue_mode=None):
    stats = dict(st.session_state.get("run_stats") or {})
    stats["frame_count"] = int(frame_count or 0)
    stats["sample_fps"] = float(sample_fps or 0.0)
    if rescue_mode is not None:
        stats["camera_rescue"] = str(rescue_mode)
    if health:
        stats.update({
            "registered_cameras": int(health.get("registered") or 0),
            "sparse_points": int(health.get("points") or 0),
            "registration_ratio": float(health.get("registration_ratio") or 0.0),
            "camera_health": str(health.get("level") or "unknown"),
        })
    st.session_state.run_stats = stats

    state_updates = {
        "frame_count": stats.get("frame_count", 0),
        "sample_fps": stats.get("sample_fps", 0.0),
        "camera_rescue": stats.get("camera_rescue"),
    }
    if health:
        state_updates.update({
            "registered_cameras": stats.get("registered_cameras", 0),
            "sparse_points": stats.get("sparse_points", 0),
            "registration_ratio": stats.get("registration_ratio", 0.0),
            "camera_health": stats.get("camera_health", "unknown"),
        })
    save_project_state(project_dir, **state_updates)


def repair_colmap_exhaustive(project_dir, images_dir, progress, console):
    """Last-resort camera connectivity pass for modest frame counts."""
    colmap = shutil.which("colmap")
    if not colmap:
        return

    database_path = Path(project_dir) / "database.db"
    sparse_dir = Path(project_dir) / "sparse"
    matcher_help = colmap_help(colmap, "exhaustive_matcher")
    guided_flag = choose_supported_option(
        matcher_help,
        ["--FeatureMatching.guided_matching", "--SiftMatching.guided_matching"],
    )

    command = [colmap, "exhaustive_matcher", "--database_path", str(database_path.resolve())]
    if guided_flag:
        command.extend([guided_flag, "1"])

    console.write(
        "[studio] Camera rescue final pass: exhaustive geometric matching across the selected frames.",
        force=True,
    )
    progress.update(55.0, "Camera rescue", "Exhaustively linking difficult camera views", force=True)
    run_streaming(
        command,
        console,
        on_line=make_stage_parser(progress, "Exhaustive repair", 55.0, 56.5),
        project_dir=project_dir,
        on_heartbeat=getattr(progress, "heartbeat", None),
    )

    remove_path_robustly(sparse_dir)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    progress.update(56.5, "Camera rescue", "Mapping exhaustive correspondences", force=True)
    run_streaming(
        [
            colmap,
            "mapper",
            "--database_path", str(database_path.resolve()),
            "--image_path", str(Path(images_dir).resolve()),
            "--output_path", str(sparse_dir.resolve()),
        ],
        console,
        on_line=make_stage_parser(progress, "Exhaustive repair", 56.5, 57.9),
        project_dir=project_dir,
        on_heartbeat=getattr(progress, "heartbeat", None),
    )


def automatic_camera_rescue(source_path, project_dir, config, progress, console, previous_frame_count):
    """Rebuild a failed camera solve with denser/stronger source coverage and robust matching."""
    source_path = Path(source_path)
    photo_mode = source_path.is_dir()

    original_target = max(2, int(config.get("target_frames", previous_frame_count or 140)))
    if photo_mode:
        available_photos = len(source_photo_files(source_path))
        rescue_target = min(1000, max(previous_frame_count, available_photos))
    else:
        rescue_target = min(
            700,
            max(
                240,
                original_target + 100,
                int(round(original_target * 1.70)),
            ),
        )

    rescue_overlap = min(48, max(20, int(config.get("overlap", 12)) * 2))
    rescue_features = max(6144, int(config.get("max_features", 4096)))
    rescue_image_size = max(1920, int(config.get("max_image_size", 1920)))

    console.write(
        (
            f"[studio] Automatic photo camera rescue: using up to {rescue_target:,} uploaded photos · "
            f"SIFT {rescue_features:,} · exhaustive matching where practical."
            if photo_mode
            else
            f"[studio] Automatic camera rescue: {previous_frame_count:,} → target {rescue_target:,} frames · "
            f"overlap {rescue_overlap} · SIFT {rescue_features:,} · guided + robust matching."
        ),
        force=True,
    )
    progress.update(
        50.0,
        "Camera rescue",
        (
            f"Rebuilding from the complete photo set · up to {rescue_target:,} photos"
            if photo_mode
            else f"Rebuilding camera solve with denser coverage · target {rescue_target:,} frames"
        ),
        force=True,
    )

    clean_project(project_dir)
    images_dir = Path(project_dir) / "images"

    if photo_mode:
        frame_count, sample_fps = prepare_photo_frames(
            source_path,
            images_dir,
            progress,
            console,
            rescue_target,
            False,
            progress_range=(50.0, 52.0),
            progress_label="Camera rescue",
        )
    else:
        frame_count, sample_fps = extract_frames(
            source_path,
            images_dir,
            progress,
            console,
            rescue_target,
            bool(config.get("smart_frames", True)),
            progress_range=(50.0, 52.0),
            progress_label="Camera rescue",
        )

    update_camera_run_stats(
        project_dir,
        frame_count,
        sample_fps,
        rescue_mode=("photo_full_set" if photo_mode else f"dense_resample_{rescue_target}"),
    )

    if frame_count < 2:
        raise PipelineError("Automatic camera rescue could not prepare enough usable source images.")

    run_colmap(
        project_dir,
        images_dir,
        progress,
        console,
        rescue_overlap,
        rescue_features,
        rescue_image_size,
        bool(config.get("shared_camera", True)),
        str(config.get("camera_model", "SIMPLE_RADIAL")),
        True,
        True,
        True,
        rescue_mode=True,
        photo_mode=photo_mode,
    )

    health = camera_solve_health(project_dir, frame_count)
    update_camera_run_stats(
        project_dir,
        frame_count,
        sample_fps,
        health,
        rescue_mode=("photo_robust" if photo_mode else "dense_resample+guided+robust"),
    )
    console.write(
        f"[studio] Camera rescue result: {health['registered']}/{frame_count} registered "
        f"({health['registration_ratio']:.1%}), {health['points']:,} sparse points.",
        force=True,
    )

    # Video rescue can gain another connectivity pass. Small photo sets have
    # already used exhaustive matching above, so repeating it is pointless.
    if health["level"] == "bad" and frame_count <= 320 and not photo_mode:
        repair_colmap_exhaustive(project_dir, images_dir, progress, console)
        health = camera_solve_health(project_dir, frame_count)
        update_camera_run_stats(
            project_dir,
            frame_count,
            sample_fps,
            health,
            rescue_mode="dense_resample+guided+robust+exhaustive",
        )
        console.write(
            f"[studio] Exhaustive rescue result: {health['registered']}/{frame_count} registered "
            f"({health['registration_ratio']:.1%}), {health['points']:,} sparse points.",
            force=True,
        )

    return frame_count, sample_fps, health



def latest_source_for_project(project_dir):
    project_dir = Path(project_dir)
    candidates = list(project_dir.glob("input.*"))
    photo_dir = project_dir / "source_photos"
    if len(source_photo_files(photo_dir)) >= 2:
        candidates.append(photo_dir)
    candidates = [item for item in candidates if item.exists()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None




def project_disk_size(project_dir):
    total = 0
    try:
        for path in Path(project_dir).rglob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def project_library():
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    entries = []
    for path in sorted((p for p in WORKSPACE_ROOT.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True):
        progress = inspect_project_progress(path)
        state = progress.get("state") or load_project_state(path)
        entries.append({
            "path": path,
            "name": path.name,
            "status": progress.get("status") or "new",
            "detail": progress.get("detail") or "",
            "output": progress.get("output"),
            "size": project_disk_size(path),
            "modified": path.stat().st_mtime,
            "state": state,
        })
    return entries


def project_live_process(project_dir):
    pid_file = _active_process_file(project_dir)
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def trash_project(project_dir):
    project_dir = Path(project_dir).resolve()
    try:
        project_dir.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise PipelineError("Refusing to delete a folder outside the Splat Studio projects directory.") from exc
    if project_live_process(project_dir):
        raise PipelineError("This project still has an active reconstruction process. Stop it before deleting the project.")
    if sys.platform == "darwin":
        trash = Path.home() / ".Trash"
        trash.mkdir(parents=True, exist_ok=True)
        destination = trash / f"SplatStudio-{project_dir.name}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(project_dir), str(destination))
        return destination
    remove_path_robustly(project_dir)
    return None


def render_projects():
    st.markdown('<div class="eyebrow">Project library</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Projects</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Open, review, reveal or safely remove local reconstruction projects.</div>', unsafe_allow_html=True)

    entries = project_library()
    if not entries:
        st.info("No projects yet. Create one from Prepare.")
        if st.button("Create a project", type="primary", width="stretch"):
            st.session_state.page = "prepare"
            st.rerun()
        return

    labels = []
    by_label = {}
    for entry in entries:
        status_label = {"complete": "Finished", "checkpoint": "Snapshot", "camera": "Camera solve", "incomplete": "In progress", "new": "New"}.get(entry["status"], entry["status"].title())
        label = f"{entry['name']}  ·  {status_label}  ·  {human_size(entry['size'])}"
        labels.append(label)
        by_label[label] = entry

    selected_label = st.selectbox("Project", labels, key="project_library_choice")
    entry = by_label[selected_label]
    path = entry["path"]
    status_label = {"complete": "Finished", "checkpoint": "Snapshot available", "camera": "Camera solve saved", "incomplete": "In progress", "new": "New"}.get(entry["status"], entry["status"].title())

    state_class = "good" if entry["status"] == "complete" else ("warn" if entry["status"] in {"checkpoint", "camera", "incomplete"} else "")
    st.markdown(
        f'<div class="project-state {state_class}">'
        f'<div class="project-state-title">{html.escape(entry["name"])} · {html.escape(status_label)}</div>'
        f'<div class="project-state-sub">{html.escape(str(path))}<br>{human_size(entry["size"])} · modified {format_duration(max(0, time.time()-entry["modified"]))} ago</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        if st.button("Open", icon=":material/folder_open:", width="stretch"):
            st.session_state.project_name = entry["name"]
            st.session_state.source_path = str(latest_source_for_project(path)) if latest_source_for_project(path) else None
            st.session_state.page = "prepare"
            st.rerun()
    with c2:
        output = entry.get("output")
        if st.button("Review", icon=":material/rate_review:", width="stretch", disabled=not bool(output)):
            st.session_state.project_name = entry["name"]
            st.session_state.active_splat = str(output)
            st.session_state.last_output = str(output)
            st.session_state.page = "review"
            st.rerun()
    with c3:
        if sys.platform == "darwin" and st.button("Reveal", icon=":material/folder:", width="stretch"):
            subprocess.Popen(["open", str(path)])
    with c4:
        if st.button("Delete project", icon=":material/delete:", width="stretch", disabled=project_live_process(path)):
            st.session_state.project_delete_confirm = str(path)

    if st.session_state.get("project_delete_confirm") == str(path):
        st.warning(f"Delete {entry['name']}? On macOS it will be moved to Trash, including its source, camera solve, snapshots and splats.")
        d1, d2 = st.columns(2)
        with d1:
            if st.button(f"Confirm delete · {entry['name']}", type="primary", icon=":material/delete_forever:", width="stretch"):
                destination = trash_project(path)
                st.session_state.project_delete_confirm = None
                if current_project_dir().resolve() == path.resolve():
                    st.session_state.project_name = "new-project"
                    st.session_state.source_path = None
                    st.session_state.active_splat = None
                    st.session_state.last_output = None
                if destination:
                    st.success(f"Moved to Trash: {destination.name}")
                st.rerun()
        with d2:
            if st.button("Cancel", width="stretch"):
                st.session_state.project_delete_confirm = None
                st.rerun()

def current_project_dir():
    return WORKSPACE_ROOT / slugify(st.session_state.project_name)


def format_duration(seconds):
    return RunProgress._format_duration(seconds)


def reconstruction_profiles():
    common = {
        "cfg_mlx_max_points": 0,
        "cfg_mlx_batch_size": 1,
        "cfg_mlx_refine": True,
        "cfg_mlx_antialiased": False,
        "cfg_checkpoint_every": 250,
        "cfg_preview_every": 250,
        "cfg_auto_resume": True,
        "cfg_smart_frames": True,
        "cfg_ai_depth": False,
        "cfg_ai_depth_points": 40000,
        "cfg_ai_mask_sky": False,
        "cfg_ai_mask_dynamic": False,
        "cfg_quadratic_overlap": True,
        "cfg_guided_matching": False,
        "cfg_robust_sift": False,
        "cfg_auto_camera_rescue": True,
    }
    profiles = {
        "Quick Preview": ({
            "cfg_target_frames": 180, "cfg_overlap": 12, "cfg_max_features": 4096, "cfg_max_image_size": 1600,
            "cfg_iterations": 1000, "cfg_mlx_max_gaussians": 8000, "cfg_mlx_train_max_side": 320,
        }, "Fast sanity check · safer 180-frame camera solve · automatic rescue enabled."),
        "Fast": ({
            "cfg_target_frames": 220, "cfg_overlap": 10, "cfg_max_features": 4096, "cfg_max_image_size": 1600,
            "cfg_iterations": 2000, "cfg_mlx_max_gaussians": 10000, "cfg_mlx_train_max_side": 384,
        }, "Quick production build · smart frames · AI models off."),
        "Balanced": ({
            "cfg_target_frames": 300, "cfg_overlap": 12, "cfg_max_features": 4096, "cfg_max_image_size": 1920,
            "cfg_iterations": 4000, "cfg_mlx_max_gaussians": 16000, "cfg_mlx_train_max_side": 480,
        }, "Recommended general-purpose profile · strong quality/speed balance."),
        "Balanced AI": ({
            "cfg_target_frames": 300, "cfg_overlap": 12, "cfg_max_features": 4096, "cfg_max_image_size": 1920,
            "cfg_iterations": 4000, "cfg_mlx_max_gaussians": 18000, "cfg_mlx_train_max_side": 480,
            "cfg_ai_depth": True, "cfg_ai_depth_points": 40000, "cfg_ai_mask_sky": True,
        }, "Balanced + Depth Anything geometry + automatic sky masking."),
        "Outdoor AI": ({
            "cfg_target_frames": 360, "cfg_overlap": 16, "cfg_max_features": 6144, "cfg_max_image_size": 1920,
            "cfg_iterations": 4500, "cfg_mlx_max_gaussians": 22000, "cfg_mlx_train_max_side": 512,
            "cfg_ai_depth": True, "cfg_ai_depth_points": 60000, "cfg_ai_mask_sky": True, "cfg_ai_mask_dynamic": True,
            "cfg_guided_matching": True,
        }, "Buildings/outdoor captures · depth assist + sky + people/vehicle masking."),
        "Indoor AI": ({
            "cfg_target_frames": 360, "cfg_overlap": 16, "cfg_max_features": 6144, "cfg_max_image_size": 1920,
            "cfg_iterations": 4500, "cfg_mlx_max_gaussians": 22000, "cfg_mlx_train_max_side": 512,
            "cfg_ai_depth": True, "cfg_ai_depth_points": 60000, "cfg_ai_mask_dynamic": True,
            "cfg_guided_matching": True,
        }, "Rooms/warehouses · depth assist + moving-object masking, without sky removal."),
        "Object Scan": ({
            "cfg_target_frames": 260, "cfg_overlap": 18, "cfg_max_features": 6144, "cfg_max_image_size": 1920,
            "cfg_iterations": 5000, "cfg_mlx_max_gaussians": 24000, "cfg_mlx_train_max_side": 512,
            "cfg_ai_depth": True, "cfg_ai_depth_points": 30000, "cfg_robust_sift": True,
        }, "Close orbit around a single object · dense overlap and depth guidance."),
        "High Detail": ({
            "cfg_target_frames": 500, "cfg_overlap": 18, "cfg_max_features": 6144, "cfg_max_image_size": 2560,
            "cfg_iterations": 6500, "cfg_mlx_max_gaussians": 30000, "cfg_mlx_train_max_side": 640,
            "cfg_guided_matching": True, "cfg_robust_sift": True,
        }, "High-detail native build · AI models off · slower camera solve and training."),
        "High Detail AI": ({
            "cfg_target_frames": 500, "cfg_overlap": 20, "cfg_max_features": 8192, "cfg_max_image_size": 2560,
            "cfg_iterations": 7000, "cfg_mlx_max_gaussians": 36000, "cfg_mlx_train_max_side": 640,
            "cfg_ai_depth": True, "cfg_ai_depth_points": 80000, "cfg_ai_mask_sky": True, "cfg_ai_mask_dynamic": True,
            "cfg_guided_matching": True, "cfg_robust_sift": True,
        }, "Maximum practical AI-assisted detail for difficult or important captures."),
        "Maximum": ({
            "cfg_target_frames": 700, "cfg_overlap": 24, "cfg_max_features": 12288, "cfg_max_image_size": 3200,
            "cfg_iterations": 9000, "cfg_mlx_max_gaussians": 50000, "cfg_mlx_train_max_side": 768,
            "cfg_ai_depth": True, "cfg_ai_depth_points": 100000, "cfg_ai_mask_sky": True, "cfg_ai_mask_dynamic": True,
            "cfg_guided_matching": True, "cfg_robust_sift": True,
        }, "Heavy profile · maximum coverage/features/AI guidance. Use when quality matters more than time."),
    }
    result = {}
    for name, (overrides, description) in profiles.items():
        settings = dict(common)
        settings.update(overrides)
        result[name] = {"settings": settings, "description": description}
    return result


def apply_preset(name):
    profiles = reconstruction_profiles()
    if name not in profiles:
        return
    for key, value in profiles[name]["settings"].items():
        st.session_state[key] = value
    st.session_state.cfg_preset = name
    save_settings()


def workload_summary():
    frames = st.session_state.cfg_target_frames
    overlap = st.session_state.cfg_overlap
    features = st.session_state.cfg_max_features
    image_size = st.session_state.cfg_max_image_size
    iterations = st.session_state.cfg_iterations
    train_side = st.session_state.cfg_mlx_train_max_side
    init_points = max(1, st.session_state.cfg_mlx_max_points or 10000)

    match_pairs = int(frames * overlap)
    training_factor = (iterations / 2500) * (train_side / 256) ** 2 * (init_points / 3500)
    score = (
        0.24 * (frames / 300)
        + 0.31 * (frames / 300) * (overlap / 12) * (features / 4096) * (image_size / 1920) ** 1.5
        + 0.45 * training_factor
    )
    if st.session_state.cfg_robust_sift:
        score *= 1.35
    if st.session_state.cfg_guided_matching:
        score *= 1.15

    if score < 0.72:
        label = "Light"
    elif score < 1.35:
        label = "Balanced"
    elif score < 2.25:
        label = "Heavy"
    else:
        label = "Very heavy"
    return label, score, match_pairs


def render_stepper(active):
    steps = ["Prepare", "Reconstruct", "Review", "Edit", "View"]
    active_index = steps.index(active) if active in steps else 0
    items = []
    for index, step in enumerate(steps):
        state = "active" if index == active_index else "done" if index < active_index else ""
        items.append(f'<div class="wiz-step {state}"><span class="wiz-num">0{index + 1}</span>{step}</div>')
    st.markdown('<div class="wizard">' + ''.join(items) + '</div>', unsafe_allow_html=True)


def available_navigation():
    pages = ["Prepare", "Projects"]
    if st.session_state.run_status != "idle":
        pages.append("Reconstruct")
    active = st.session_state.get("active_splat")
    if active and Path(active).exists():
        pages.extend(["Review", "Edit", "View"])
    return pages


def navigate_to(label):
    page_map = {
        "Prepare": "prepare",
        "Projects": "projects",
        "Reconstruct": "run",
        "Review": "review",
        "Edit": "edit",
        "View": "view",
    }
    st.session_state.page = page_map[label]


def on_navigation_change():
    selected = st.session_state.get("nav_choice")
    if selected:
        navigate_to(selected)


def header_project_status():
    """Return CSS class + concise durable project/checkpoint status for the header."""
    try:
        project_dir = current_project_dir()
        progress = inspect_project_progress(project_dir)
        checkpoint = progress.get("checkpoint")
        state = progress.get("state") or load_project_state(project_dir)

        if progress.get("status") == "complete":
            return "finished", f"Saved · Stage {PIPELINE_STAGE_TOTAL}/{PIPELINE_STAGE_TOTAL} · Review ready"

        retry_at = state.get("auto_resume_retry_at")
        if retry_at:
            remaining = max(0, int(float(retry_at) - time.time()))
            if remaining > 0:
                retry_text = f"{remaining // 60}:{remaining % 60:02d}"
                if checkpoint and int(checkpoint.get("total_steps") or 0):
                    return "warning", (
                        f"Auto-resume · {retry_text} · Snapshot "
                        f"{int(checkpoint.get('step') or 0):,}/{int(checkpoint.get('total_steps') or 0):,}"
                    )
                return "warning", f"Auto-resume · {retry_text} · recovery pending"

        if checkpoint:
            step = int(checkpoint.get("step") or 0)
            total = int(checkpoint.get("total_steps") or 0)
            stage = int(checkpoint.get("stage_index") or state.get("last_checkpoint_stage") or 10)
            age_seconds = max(0.0, time.time() - float(checkpoint.get("mtime") or time.time()))
            age_text = format_duration(age_seconds)
            runtime_seconds = float(checkpoint.get("project_runtime_seconds") or 0.0)
            runtime_text = format_duration(runtime_seconds) if runtime_seconds > 0 else None
            runtime_suffix = f" · total {runtime_text}" if runtime_text else ""
            if total:
                return "", f"Snapshot · Stage {stage}/{PIPELINE_STAGE_TOTAL} · {step:,}/{total:,} · {age_text} ago{runtime_suffix}"
            return "", f"Snapshot · Stage {stage}/{PIPELINE_STAGE_TOTAL} · step {step:,} · {age_text} ago{runtime_suffix}"

        if progress.get("status") == "camera":
            return "", f"Saved · Stage 5/{PIPELINE_STAGE_TOTAL} · camera solve"

        if progress.get("status") in {"incomplete", "checkpoint"}:
            stage = int(state.get("last_checkpoint_stage") or state.get("last_stage") or 1)
            return "", f"Progress saved · Stage {stage}/{PIPELINE_STAGE_TOTAL}"

        return "neutral", "No snapshot yet"
    except Exception:
        return "neutral", "Snapshot status unavailable"


def render_header_status(slot):
    checkpoint_class, checkpoint_text = header_project_status()
    checkpoint_class_attr = f" {checkpoint_class}" if checkpoint_class else ""
    slot.markdown(
        f"""
        <div class="header-status-row">
            <span class="studio-local">Local compute</span>
            <span class="studio-brand-tag">COLMAP</span>
            <span class="studio-brand-tag">MLX</span>
            <span class="studio-brand-tag">SuperSplat</span>
            <span class="checkpoint-chip{checkpoint_class_attr}">{html.escape(checkpoint_text)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stop_and_close_splat_studio():
    """Safely stop the active reconstruction, persist its checkpoint state, then close the local app."""
    if st.session_state.get("run_status") == "running":
        cancel_active_run()
    close_splat_studio()


def render_header():
    left, right = st.columns([4.15, 2.65], vertical_alignment="center")

    with left:
        icon_col, identity_col = st.columns([0.48, 5.4], vertical_alignment="center", gap="small")

        with icon_col:
            if APP_ICON_PATH.exists():
                st.image(str(APP_ICON_PATH), width=78)
            else:
                st.markdown(
                    '<div style="width:72px;height:72px;border:1px solid var(--line-strong);border-radius:14px;background:var(--panel2)"></div>',
                    unsafe_allow_html=True,
                )

        with identity_col:
            st.markdown(
                """
                <div class="studio-brand-title">Splat Studio</div>
                <div class="studio-brand-sub">Gaussian Reconstruction &amp; Editing Workstation</div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        global _HEADER_STATUS_SLOT
        _HEADER_STATUS_SLOT = st.empty()
        render_header_status(_HEADER_STATUS_SLOT)

        navigate_col, settings_col, close_col = st.columns([1.1, 1.25, 0.95], gap="small")

        with navigate_col:
            navigate_popover = st.popover("Navigate", icon=":material/menu:", width="stretch")

        with settings_col:
            settings_popover = st.popover("Settings", icon=":material/tune:", width="stretch")

        with close_col:
            with st.container(key="header_danger_control"):
                danger_label = "Stop" if st.session_state.run_status == "running" else "Close"
                danger_icon = ":material/stop:" if st.session_state.run_status == "running" else ":material/power_settings_new:"
                close_popover = st.popover(danger_label, icon=danger_icon, width="stretch")

        with navigate_popover:
            st.markdown("##### Go to")
            st.caption("The numbered strip shows workflow progress. These controls jump to pages that are currently available.")

            navigation_targets = [
                ("Setup & parameters", "Prepare"),
                ("Manage projects", "Projects"),
                ("Run status / report", "Reconstruct"),
                ("Review result", "Review"),
                ("Edit in SuperSplat", "Edit"),
                ("3D viewer", "View"),
            ]
            available = set(available_navigation())
            for button_label, page_label in navigation_targets:
                if page_label in available:
                    if st.button(button_label, key=f"nav_menu_{page_label}", width="stretch"):
                        navigate_to(page_label)
                        st.rerun()

        with settings_popover:
            st.markdown("##### Appearance")
            with st.container(key="theme_segment_control"):
                st.segmented_control(
                    "Theme",
                    ["Dark", "Light", "Custom"],
                    key="theme_mode",
                    on_change=save_settings,
                    width="stretch",
                    help="Dark and Light use tuned palettes. Custom lets you choose the entire application background colour.",
                )

            if st.session_state.theme_mode == "Dark":
                dark_choices = ["Midnight", "Graphite", "Deep Blue"]
                if st.session_state.theme_background not in dark_choices:
                    st.session_state.theme_background = "Midnight"
                st.selectbox(
                    "Background",
                    dark_choices,
                    key="theme_background",
                    on_change=save_settings,
                    help="Changes the base dark surface while retaining readable controls and panels.",
                )
            elif st.session_state.theme_mode == "Light":
                light_choices = ["Cloud", "Paper", "Warm"]
                if st.session_state.theme_background not in light_choices:
                    st.session_state.theme_background = "Cloud"
                st.selectbox(
                    "Background",
                    light_choices,
                    key="theme_background",
                    on_change=save_settings,
                    help="Choose a cool, neutral or warmer light application background.",
                )
            else:
                st.color_picker(
                    "Background colour",
                    key="theme_custom_background",
                    on_change=save_settings,
                    help="Choose almost any background. Splat Studio automatically switches text and controls between light and dark contrast.",
                )

            st.color_picker(
                "Accent",
                key="theme_accent",
                on_change=save_settings,
                help="Changes highlights, progress bars, active navigation and primary actions.",
            )
            st.slider("Interface size", 90, 115, key="theme_scale", format="%d%%", on_change=save_settings)
            st.slider("Console text", 10, 16, key="theme_console_size", format="%d px", on_change=save_settings)
            st.slider("Console history", 120, 500, step=20, key="theme_console_lines", on_change=save_settings, help="How many recent terminal lines stay visible in the live console. The full log is always kept on disk.")
            st.toggle("Reduce motion", key="theme_reduce_motion", on_change=save_settings, help="Disables page and progress animations.")
            st.caption(f"Saved automatically · {SETTINGS_FILE}")
            if st.button("Reset saved preferences", width="stretch", help="Restore the default reconstruction and appearance settings. Project files are not removed."):
                reset_saved_settings()
                st.rerun()

            st.divider()
            st.markdown("##### Local tools")
            status = playcanvas_status()
            st.caption("SuperSplat Editor: " + ("ready" if status["editor"] else "not installed"))
            st.caption("SuperSplat Viewer: " + ("ready" if status["viewer"] else "not installed"))
            install_text = "Repair SuperSplat tools" if status["editor"] and status["viewer"] else "Install SuperSplat tools"
            if st.button(install_text, width="stretch"):
                with st.status("Building local SuperSplat tools...", expanded=True) as status_box:
                    try:
                        install_playcanvas_tools(reinstall=True)
                        status_box.update(label="SuperSplat tools ready", state="complete", expanded=False)
                    except PipelineError as exc:
                        status_box.update(label="SuperSplat setup failed", state="error", expanded=True)
                        st.error(str(exc))

        with close_popover:
            if st.session_state.run_status == "running":
                st.markdown("##### Stop reconstruction")
                checkpoint = training_checkpoint_info(run_project_dir())
                if checkpoint:
                    step = int(checkpoint.get("step") or 0)
                    total = int(checkpoint.get("total_steps") or 0)
                    checkpoint_text = f"Snapshot {step:,}/{total:,}" if total else f"Snapshot step {step:,}"
                    st.caption(f"{checkpoint_text} is currently recoverable. Stop requests another safe snapshot boundary before terminating the worker.")
                else:
                    st.caption("Stops the active reconstruction safely. If training has started, Splat Studio requests a recovery snapshot before terminating the worker.")

                if st.button("Stop", icon=":material/stop:", key="header_stop_action", width="stretch"):
                    cancel_active_run()
                    st.rerun()

                if st.button("Stop & Close", icon=":material/power_settings_new:", key="header_stop_close_action", width="stretch"):
                    st.success("Stopping reconstruction and closing Splat Studio…")
                    stop_and_close_splat_studio()
            else:
                st.markdown("##### Close Splat Studio")
                st.caption("Stops the local Streamlit server. Saved projects, snapshots and preferences remain on disk.")
                if st.button("Close Application", icon=":material/power_settings_new:", key="header_close_action", width="stretch"):
                    st.success("Closing Splat Studio…")
                    close_splat_studio()

    label_map = {"prepare": "Prepare", "projects": "Prepare", "run": "Reconstruct", "review": "Review", "edit": "Edit", "view": "View"}
    current_label = label_map.get(st.session_state.page, "Prepare")
    render_stepper(current_label)
    if st.session_state.run_status != "running":
        st.caption("Workflow progress · use Navigate above to jump between available pages.")



def render_source_summary(path):
    path = Path(path)

    if path.is_dir():
        photos = source_photo_files(path)
        total_size = sum(item.stat().st_size for item in photos)
        target = int(st.session_state.cfg_target_frames)
        used = min(len(photos), target)
        right = (
            f"{len(photos):,} photos · all used"
            if len(photos) <= target
            else f"{len(photos):,} photos · up to {used:,} selected"
        )
        st.markdown(
            f"""<div class="file-strip"><div style="min-width:0"><div class="file-name">Uploaded photo set</div><div class="file-meta">{human_size(total_size)} · {html.escape(right)}</div></div><div class="ready">Ready</div></div>""",
            unsafe_allow_html=True,
        )
        return

    duration = probe_duration(path)
    target = st.session_state.cfg_target_frames
    sample_fps = min(8.0, max(0.10, target / duration)) if duration else None
    right = f"{format_duration(duration)} · ~{sample_fps:.2f} fps sampling" if duration and sample_fps else path.suffix[1:].upper()
    st.markdown(
        f"""<div class="file-strip"><div style="min-width:0"><div class="file-name">{html.escape(path.name)}</div><div class="file-meta">{human_size(path.stat().st_size)} · {html.escape(right)}</div></div><div class="ready">Ready</div></div>""",
        unsafe_allow_html=True,
    )



def render_prepare():
    pending_config = st.session_state.pop("_pending_saved_run_config", None)
    if pending_config:
        apply_saved_run_config(pending_config)

    st.markdown('<div class="eyebrow">Reconstruction setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Prepare</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Choose your source, tune the camera solve and decide how much time to trade for reconstruction detail.</div>', unsafe_allow_html=True)

    source_col, project_col = st.columns([1.45, 1], gap="large")
    with source_col:
        with st.container(border=True):
            st.markdown(
                '<div class="card-title">Source</div>'
                '<div class="card-sub">Use a video orbit or upload a set of overlapping still photos.</div>',
                unsafe_allow_html=True,
            )

            st.segmented_control(
                "Source type",
                ["Video", "Photos"],
                key="source_mode",
                width="stretch",
                help="Video is sampled automatically. Photos are sent directly to COLMAP after EXIF rotation is normalized.",
            )

            project_dir = current_project_dir()
            project_dir.mkdir(parents=True, exist_ok=True)
            source_path = latest_source_for_project(project_dir)

            if st.session_state.source_mode == "Video":
                uploaded = st.file_uploader(
                    "Choose video",
                    type=["mp4", "mov", "m4v"],
                    key="source_uploader",
                    help="Splat Studio samples frames across the video instead of feeding every frame into COLMAP.",
                )

                if uploaded is not None:
                    suffix = Path(uploaded.name).suffix.lower() or ".mp4"
                    destination = project_dir / f"input{suffix}"
                    data = uploaded.getbuffer()
                    upload_sig = f"video:{uploaded.name}:{len(data)}:{hashlib.sha1(data).hexdigest()}"

                    if st.session_state.get("source_upload_sig") != upload_sig or not destination.exists():
                        for old in project_dir.glob("input.*"):
                            if old != destination:
                                old.unlink(missing_ok=True)
                        photo_dir = project_dir / "source_photos"
                        if photo_dir.exists():
                            remove_path_robustly(photo_dir)
                        destination.write_bytes(data)
                        st.session_state.source_upload_sig = upload_sig
                    source_path = destination

            else:
                photo_uploads = st.file_uploader(
                    "Choose photos",
                    type=["jpg", "jpeg", "png", "webp", "tif", "tiff"],
                    accept_multiple_files=True,
                    key="photo_source_uploader",
                    help="Select overlapping photos of the same scene. 30–200 well-spaced images is a good range; all photos remain local.",
                )

                if photo_uploads:
                    digest = hashlib.sha1()
                    digest.update(f"count:{len(photo_uploads)}".encode("ascii"))
                    for uploaded_photo in photo_uploads:
                        buffer = uploaded_photo.getbuffer()
                        digest.update(uploaded_photo.name.encode("utf-8", errors="replace"))
                        digest.update(str(len(buffer)).encode("ascii"))
                        digest.update(buffer)
                    upload_sig = f"photos:{digest.hexdigest()}"

                    photo_dir = project_dir / "source_photos"
                    if st.session_state.get("source_upload_sig") != upload_sig or len(source_photo_files(photo_dir)) != len(photo_uploads):
                        if photo_dir.exists():
                            remove_path_robustly(photo_dir)
                        photo_dir.mkdir(parents=True, exist_ok=True)

                        for index, uploaded_photo in enumerate(photo_uploads, start=1):
                            original = Path(uploaded_photo.name)
                            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", original.stem).strip("._")[:90] or "photo"
                            suffix = original.suffix.lower()
                            destination = photo_dir / f"{index:04d}_{safe_stem}{suffix}"
                            destination.write_bytes(uploaded_photo.getbuffer())

                        for old in project_dir.glob("input.*"):
                            old.unlink(missing_ok=True)

                        # Make directory freshness deterministic for latest_source_for_project().
                        os.utime(photo_dir, None)
                        st.session_state.source_upload_sig = upload_sig

                    source_path = photo_dir

                st.caption(
                    "Still-photo mode automatically fixes EXIF orientation. "
                    "If you upload more photos than the profile's Maximum frames, Smart frame curation selects the strongest coverage."
                )

            if source_path and source_path.exists():
                st.session_state.source_path = str(source_path)
                render_source_summary(source_path)

                with st.expander("Preview source"):
                    if source_path.is_dir():
                        previews = source_photo_files(source_path)[:8]
                        if previews:
                            cols = st.columns(min(4, len(previews)))
                            for index, photo in enumerate(previews):
                                with cols[index % len(cols)]:
                                    st.image(str(photo), caption=photo.name, width="stretch")
                            if len(source_photo_files(source_path)) > len(previews):
                                st.caption(f"Showing {len(previews)} of {len(source_photo_files(source_path)):,} uploaded photos.")
                    else:
                        st.video(str(source_path))
            else:
                st.session_state.source_path = None
                st.caption(
                    "No source photos loaded yet."
                    if st.session_state.source_mode == "Photos"
                    else "No source loaded yet."
                )

    with project_col:
        with st.container(border=True):
            st.markdown('<div class="card-title">Project</div><div class="card-sub">Files stay under the local projects folder.</div>', unsafe_allow_html=True)
            st.text_input(
                "Project name",
                key="project_name",
                help="Used as the project folder name. Existing projects can be reopened by entering the same name.",
            )
            project_dir = current_project_dir()
            project_progress = inspect_project_progress(project_dir)
            st.caption(str(project_dir))

            if project_progress["status"] == "complete":
                output = project_progress["output"]
                st.markdown(
                    '<div class="project-state good"><div class="project-state-title">Finished · ready for Review</div>'
                    '<div class="project-state-sub">A completed Gaussian output exists. This project no longer shows Continue because there is nothing left to resume.</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Review finished project", type="primary", icon=":material/rate_review:", width="stretch"):
                    st.session_state.active_splat = str(output)
                    st.session_state.last_output = str(output)
                    st.session_state.page = "review"
                    st.rerun()
            elif project_progress["status"] in {"checkpoint", "camera", "incomplete"}:
                checkpoint = project_progress.get("checkpoint")
                extra = f" · {checkpoint['gaussians']:,} Gaussians saved" if checkpoint and checkpoint.get("gaussians") else ""
                st.markdown(
                    f'<div class="project-state warn"><div class="project-state-title">In progress · Continue available</div>'
                    f'<div class="project-state-sub">{html.escape(project_progress["detail"] + extra)}</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Continue project", type="primary", icon=":material/resume:", width="stretch", key="project_continue_top"):
                    saved_config = (project_progress.get("state") or {}).get("run_config") or {}
                    if saved_config:
                        queue_saved_run_config(saved_config)
                    source_for_continue = project_progress.get("source") or latest_source_for_project(project_dir)
                    if source_for_continue:
                        launch_project_run(project_dir, source_for_continue, saved_config or session_run_config(source_for_continue), continue_mode=True)
                        st.rerun()
            else:
                st.markdown(
                    '<div class="project-state"><div class="project-state-title">New project</div>'
                    '<div class="project-state-sub">Upload a video or photo set and choose a reconstruction profile to begin.</div></div>',
                    unsafe_allow_html=True,
                )

            if st.button("Manage all projects", icon=":material/folder_managed:", width="stretch"):
                st.session_state.page = "projects"
                st.rerun()

            imported = st.file_uploader(
                "Or open an existing splat",
                type=["ply", "sog", "splat", "ksplat", "spz"],
                key="existing_splat_uploader",
                help="Skip reconstruction and go straight to Review, Edit or View.",
            )
            if imported is not None:
                imported_path = save_imported_splat(imported, project_dir)
                st.session_state.active_splat = str(imported_path)
                st.session_state.page = "review"
                st.rerun()

    st.markdown("### Reconstruction profile")
    profiles = reconstruction_profiles()
    profile_names = list(profiles)
    for row_start in (0, 5):
        cols = st.columns(5, gap="small")
        for col, name in zip(cols, profile_names[row_start:row_start + 5]):
            with col:
                st.button(
                    name,
                    on_click=apply_preset,
                    args=(name,),
                    key=f"profile_{slugify(name)}",
                    width="stretch",
                    type="primary" if st.session_state.cfg_preset == name else "secondary",
                    help=profiles[name]["description"],
                )
    selected_profile = profiles.get(st.session_state.cfg_preset)
    if selected_profile:
        ai_bits = []
        psettings = selected_profile["settings"]
        if psettings.get("cfg_ai_depth"):
            ai_bits.append(f"Depth +{int(psettings.get('cfg_ai_depth_points', 0)):,} pts")
        if psettings.get("cfg_ai_mask_sky"):
            ai_bits.append("Sky mask")
        if psettings.get("cfg_ai_mask_dynamic"):
            ai_bits.append("People/vehicle mask")
        ai_text = " · ".join(ai_bits) if ai_bits else "AI vision off"
        st.markdown(
            f'<div class="project-state good"><div class="project-state-title">{html.escape(st.session_state.cfg_preset)}</div>'
            f'<div class="project-state-sub">{html.escape(selected_profile["description"])}<br>'
            f'{int(psettings["cfg_target_frames"]):,} frames · {int(psettings["cfg_iterations"]):,} steps · '
            f'{int(psettings["cfg_mlx_train_max_side"])}px training · {html.escape(ai_text)}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Camera reconstruction</div><div class="card-sub">These mainly control COLMAP speed, memory use and camera-solve robustness.</div>', unsafe_allow_html=True)
            st.slider(
                "Maximum frames",
                80,
                1000,
                step=20,
                key="cfg_target_frames",
                help="Video: maximum sampled frames. Photos: maximum uploaded images used before smart curation. More views add coverage and detail but increase COLMAP and training work.",
            )
            st.slider(
                "Frame matching overlap",
                4,
                40,
                step=1,
                key="cfg_overlap",
                help="How many nearby frames each frame is compared against. Higher values can rescue fast camera movement or weak overlap, but matching work rises quickly.",
            )
            st.select_slider(
                "SIFT features per frame",
                options=[1024, 2048, 3072, 4096, 6144, 8192, 12288],
                key="cfg_max_features",
                help="Maximum visual landmarks retained per image. Higher values help textured or complex scenes but cost memory and matching time.",
            )
            st.select_slider(
                "Feature image resolution",
                options=[1024, 1280, 1600, 1920, 2560, 3200],
                key="cfg_max_image_size",
                help="Maximum image dimension COLMAP uses for feature extraction. Lower values are much faster; higher values can preserve small details and distant texture.",
            )

    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Gaussian training</div><div class="card-sub">These mainly affect the final splat optimisation after camera poses are solved.</div>', unsafe_allow_html=True)
            st.slider(
                "Training iterations",
                1000,
                30000,
                step=1000,
                key="cfg_iterations",
                help="How long the Gaussian optimiser trains. Too few can look under-developed; more iterations improve convergence but eventually give diminishing returns.",
            )
            st.selectbox(
                "Camera model",
                ["SIMPLE_RADIAL", "PINHOLE", "OPENCV"],
                key="cfg_camera_model",
                help="Lens model used by COLMAP. SIMPLE_RADIAL is a strong general starting point for ordinary phone/camera video; more complex models estimate more lens parameters.",
            )
            st.toggle(
                "Shared camera intrinsics",
                key="cfg_shared_camera",
                help="Treats every frame as coming from the same physical camera/lens. Recommended for a normal single-camera video and generally makes calibration more stable.",
            )
            st.toggle(
                "Reuse compatible camera solve",
                key="cfg_reuse_camera",
                help="If the source and COLMAP settings are unchanged, reuse extracted frames/database/sparse poses and jump straight to Gaussian training. Changing reconstruction settings automatically forces a rebuild.",
            )

    with st.container(border=True):
        st.markdown('<div class="card-title">AI-assisted preprocessing</div><div class="card-sub">Optional local vision models can clean the input and add geometry guidance before native Metal training.</div>', unsafe_allow_html=True)
        runtime = ai_runtime_status()
        if runtime.get("ready"):
            device_text = "Apple GPU · MPS" if runtime.get("mps") else "CPU fallback"
            st.markdown(f'<div class="project-state good"><div class="project-state-title">AI runtime ready · {html.escape(device_text)}</div><div class="project-state-sub">Models run locally. Downloads are stored under SplatStudio/models/huggingface.</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="project-state warn"><div class="project-state-title">AI runtime not installed</div><div class="project-state-sub">Run Setup AI Vision.command once. Smart frame curation still works without the neural models.</div></div>', unsafe_allow_html=True)

        ai1, ai2 = st.columns(2, gap="large")
        with ai1:
            st.toggle(
                "Smart frame curation",
                key="cfg_smart_frames",
                help="Oversamples the video, then keeps sharp, well-exposed, temporally distributed frames while rejecting near-duplicates. Recommended.",
            )
            st.toggle(
                "Depth Anything V2 geometry assist",
                key="cfg_ai_depth",
                disabled=not runtime.get("ready"),
                help="Experimental. Estimates relative depth locally, aligns it to COLMAP sparse geometry and seeds extra Gaussians in weakly reconstructed regions.",
            )
            st.select_slider(
                "AI depth point budget",
                options=[10000, 20000, 40000, 60000, 100000],
                key="cfg_ai_depth_points",
                format_func=lambda value: f"Up to {value:,} extra points",
                disabled=not st.session_state.cfg_ai_depth,
                help="Maximum depth-assisted Gaussian seed points added to the real COLMAP sparse cloud.",
            )
        with ai2:
            st.toggle(
                "AI mask sky",
                key="cfg_ai_mask_sky",
                disabled=not runtime.get("ready"),
                help="Semantic segmentation marks sky pixels as ignored during Gaussian training. Original images are never modified.",
            )
            st.toggle(
                "AI mask people & vehicles",
                key="cfg_ai_mask_dynamic",
                disabled=not runtime.get("ready"),
                help="Ignores recognized people and common vehicle classes during training to reduce ghosting from moving objects. Original images are preserved.",
            )
            st.caption("Masking is non-destructive · only the training loss ignores masked pixels.")
            st.caption("AI previews use the same small ≤300 px live-preview panel as training.")

    with st.expander("Advanced quality and performance"):
        a, b = st.columns(2, gap="large")
        with a:
            st.toggle(
                "Quadratic frame matching",
                key="cfg_quadratic_overlap",
                help="Also links frames at increasing distances through the sequence. It improves longer-range connectivity for many videos at a modest matching cost.",
            )
            st.toggle(
                "Guided matching",
                key="cfg_guided_matching",
                help="Asks COLMAP to refine matches using geometric verification. It can increase useful correspondences in difficult scenes but is slower.",
            )
            st.toggle(
                "Robust DSP/Affine SIFT",
                key="cfg_robust_sift",
                help="Enables affine-shape estimation and domain-size pooling when your COLMAP build supports them. More robust to viewpoint/scale changes, but noticeably slower.",
            )
            st.toggle(
                "Automatic camera rescue",
                key="cfg_auto_camera_rescue",
                help="Recommended. If the first COLMAP solve is unusable, Splat Studio automatically resamples more overlapping views, enables guided/robust matching and can try exhaustive matching before giving up.",
            )
        with b:
            st.number_input(
                "Position learning rate",
                min_value=0.00001,
                max_value=0.01,
                step=0.00001,
                format="%.5f",
                key="cfg_means_lr",
                help="Learning rate for Gaussian positions (means). Splat Studio uses 0.00016 as the default position learning rate. Higher can move points faster but can make optimisation less stable.",
            )
            st.select_slider(
                "Training render resolution",
                options=[256, 320, 384, 480, 512, 640, 768],
                key="cfg_mlx_train_max_side",
                format_func=lambda value: f"{value}px max side",
                help="Maximum image dimension used by the native C++/Metal trainer. 480px is the Balanced target and is roughly quarter-resolution for 1080p footage.",
            )
            st.selectbox(
                "Training batch size",
                [1, 2, 4],
                key="cfg_mlx_batch_size",
                help="Number of registered camera views optimized together. 1 is safest; larger batches can improve view consistency but consume considerably more unified memory.",
            )
            st.select_slider(
                "Initial SfM points",
                options=[0, 5000, 10000, 20000, 30000, 50000],
                key="cfg_mlx_max_points",
                format_func=lambda value: "All reconstructed points" if value == 0 else f"{value:,} points",
                help="Use the complete COLMAP sparse cloud for normal native training. Limits are mainly useful for quick diagnostic runs.",
            )
            st.caption("Gaussian population · native adaptive split/prune manages the count automatically; Splat Studio no longer imposes an artificial cap.")
            st.slider(
                "SSIM detail weight",
                min_value=0.0,
                max_value=0.4,
                step=0.05,
                key="cfg_mlx_ssim_weight",
                help="Balances plain pixel error with structural similarity. 0.20 is a strong general-purpose value; higher values push local structure/detail harder and add some compute.",
            )
            st.toggle(
                "Adaptive split / prune",
                key="cfg_mlx_refine",
                help="Periodically splits high-gradient Gaussians and prunes near-transparent ones. This can add missing detail while controlling wasted splats; topology changes also add training work.",
            )
            st.caption("Renderer · native MLX C++/Metal kernels with spherical harmonics (SH degree 3).")
            st.select_slider(
                "Training snapshot interval",
                options=[100, 250, 500, 1000],
                key="cfg_checkpoint_every",
                format_func=lambda value: f"Every {value} steps",
                help="Saves Gaussian parameters, complete Adam state, refinement state, frame sampler state, runtime and training step. The latest two snapshots are retained.",
            )
            st.select_slider(
                "Live preview interval",
                options=[100, 250, 500, 1000],
                key="cfg_preview_every",
                format_func=lambda value: f"Every {value} steps",
                help="Renders one fixed camera at up to 300×300 px. 250 steps gives 16 previews during a 4,000-step Balanced run.",
            )
            st.toggle(
                "Automatic error recovery",
                key="cfg_auto_resume",
                help="If native training fails unexpectedly, wait two minutes and resume automatically from the newest snapshot. Up to three retries progressively lower render resolution/cache pressure; later retries pause refinement.",
            )
            st.caption("Automatic recovery · 2-minute visible countdown · up to 3 retries · progressively safer GPU settings.")
            st.toggle(
                "Keep Mac awake while running",
                key="cfg_prevent_sleep",
                help="On macOS, wraps native reconstruction commands with caffeinate so an unattended run is not interrupted by idle sleep.",
            )

    workload, score, pairs = workload_summary()
    project_dir = current_project_dir()
    source_path = Path(st.session_state.source_path) if st.session_state.get("source_path") else None
    duration = probe_duration(source_path) if source_path and source_path.exists() else None
    expected_fps = min(8.0, max(0.10, st.session_state.cfg_target_frames / duration)) if duration else None

    st.markdown(
        f"""<div class="summary-grid">
        <div class="summary-box"><div class="summary-k">Profile</div><div class="summary-v">{workload}</div></div>
        <div class="summary-box"><div class="summary-k">Relative work</div><div class="summary-v">×{score:.1f}</div></div>
        <div class="summary-box"><div class="summary-k">Nearby pairs</div><div class="summary-v">~{pairs:,}</div></div>
        <div class="summary-box"><div class="summary-k">Video sample</div><div class="summary-v">{f'{expected_fps:.2f} fps' if expected_fps else 'Waiting'}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    render_resource_planner(planning_config_from_session(), project_dir, source_path)

    deps = missing_dependencies()
    pc = playcanvas_status()
    if deps:
        st.error("Missing required pipeline dependency: " + ", ".join(deps))
    elif not (pc["editor"] and pc["viewer"]):
        st.info("Reconstruction is ready. SuperSplat is optional during Build and can be installed later from Settings for Edit/View.")
    else:
        st.markdown('<div class="tip"><strong>Ready.</strong> The pipeline is configured for sequential video matching and Splat Studio multiview MLX training. If a first test is slow, reduce Maximum frames or Feature image resolution before reducing training iterations.</div>', unsafe_allow_html=True)

    render_tip_carousel("prepare")

    start_disabled = bool(deps) or not source_path or not source_path.exists()
    final_progress = inspect_project_progress(project_dir)

    if final_progress["status"] == "complete":
        if st.button("Review completed result", type="primary", icon=":material/rate_review:", width="stretch"):
            st.session_state.active_splat = str(final_progress["output"])
            st.session_state.last_output = str(final_progress["output"])
            st.session_state.page = "review"
            st.rerun()
    elif final_progress["status"] in {"checkpoint", "camera", "incomplete"}:
        if st.button("Continue project", type="primary", icon=":material/resume:", width="stretch", disabled=start_disabled):
            saved_config = (final_progress.get("state") or {}).get("run_config") or {}
            if saved_config:
                queue_saved_run_config(saved_config)
            launch_project_run(project_dir, source_path, saved_config or session_run_config(source_path), continue_mode=True)
            st.rerun()
    else:
        if st.button("Start reconstruction", type="primary", icon=":material/play_arrow:", width="stretch", disabled=start_disabled):
            launch_project_run(project_dir, source_path, session_run_config(source_path), continue_mode=False)
            st.rerun()


def run_project_dir():
    stored = st.session_state.get("run_project_dir")
    return Path(stored) if stored else current_project_dir()


def cancel_active_run():
    project_dir = run_project_dir()
    stop_stale_pipeline_process(project_dir)
    st.session_state.run_launch_pending = False
    st.session_state.run_status = "cancelled"
    st.session_state.run_error = "Reconstruction cancelled by user."
    st.session_state.run_finished_ts = time.time()
    checkpoint = training_checkpoint_info(project_dir)
    persist_current_runtime(
        project_dir,
        status="cancelled",
        phase="training" if checkpoint else "reconstruct",
        last_error=st.session_state.run_error,
        last_checkpoint_step=(checkpoint or {}).get("step", 0),
        total_steps=(checkpoint or {}).get("total_steps", 0),
    )
    st.session_state.page = "run"


def diagnostics_payload():
    try:
        uname = os.uname()
        platform_text = f"{uname.sysname} {uname.release} ({uname.machine})"
        machine_text = str(uname.machine)
    except Exception:
        platform_text = str(sys.platform)
        machine_text = "unknown"

    return {
        "app": APP_NAME,
        "timestamp": time.time(),
        "platform": platform_text,
        "machine": machine_text,
        "python": sys.version,
        "streamlit": getattr(st, "__version__", "unknown"),
        "ffmpeg": command_version(["ffmpeg", "-version"]),
        "colmap": command_version(["colmap", "-h"]),
        "trainer": str(GSPLAT_METAL_TRAINER.resolve()),
        "project_dir": str(run_project_dir().resolve()),
        "source": st.session_state.get("run_source_path"),
        "run_status": st.session_state.get("run_status"),
        "run_error": st.session_state.get("run_error"),
        "run_config": st.session_state.get("run_config"),
        "run_stats": st.session_state.get("run_stats"),
        "pipeline_stage": st.session_state.get("run_stage_index"),
        "pipeline_stage_total": PIPELINE_STAGE_TOTAL,
        "total_runtime_seconds": current_total_runtime_seconds(run_project_dir()),
        "model_gaussians": st.session_state.get("run_model_gaussians"),
        "model_loss": st.session_state.get("run_model_loss"),
        "pipeline_stage_progress_percent": pipeline_stage_progress(
            int(st.session_state.get("run_stage_index") or 1),
            float(st.session_state.get("run_progress", 0.0)),
        ),
    }


def load_log_text():
    log = st.session_state.get("run_log_path")
    if not log or not Path(log).exists():
        return "No pipeline log is available."
    try:
        return Path(log).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "Unable to read pipeline log."


def render_run():
    status = st.session_state.run_status
    project_dir = run_project_dir()

    if status == "running" and not st.session_state.get("run_launch_pending"):
        # The original pipeline run never rerenders this function while it is active.
        # Reaching here again therefore means the app was refreshed/rerun, so stop any orphan safely.
        stop_stale_pipeline_process(project_dir)
        st.session_state.run_status = "interrupted"
        st.session_state.run_error = "The reconstruction was interrupted by an app rerun or browser refresh. Any native child process was stopped safely."
        st.session_state.run_finished_ts = time.time()
        checkpoint = training_checkpoint_info(project_dir)
        persist_current_runtime(
            project_dir,
            status="interrupted",
            phase="training" if checkpoint else "reconstruct",
            last_error=st.session_state.run_error,
            last_checkpoint_step=(checkpoint or {}).get("step", 0),
            total_steps=(checkpoint or {}).get("total_steps", 0),
        )
        status = "interrupted"

    if status == "running":
        st.markdown('<div class="eyebrow">Reconstruction in progress</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="page-title">{html.escape(st.session_state.project_name)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="run-live"><span class="run-spinner"></span><div><div class="run-live-title">Splat Studio is processing</div><div class="run-live-sub">Local FFmpeg, COLMAP or MLX work is active. Use the red Stop control in the header if needed.</div></div></div>', unsafe_allow_html=True)

        progress_slot = st.empty()
        m1, m2, m3, m4 = st.columns(4)
        elapsed_slot = m1.empty()
        eta_slot = m2.empty()
        model_slot = m3.empty()
        activity_slot = m4.empty()

        st.caption("ETA learns silently for the first five minutes, then appears and refreshes every minute using current stage timing.")
        render_tip_carousel("run")

        st.markdown("#### Live training preview")
        preview_col, preview_info_col = st.columns([1.0, 2.1], gap="large")
        with preview_col:
            with st.container(border=True):
                preview_slot = st.empty()
                preview_meta_slot = st.empty()
        with preview_info_col:
            st.caption(
                "A single fixed camera is rendered only at the configured preview interval. "
                "The render itself is capped at 300 px on its longest side so monitoring does not become the workload."
            )

        progress = RunProgress(
            progress_slot,
            elapsed_slot,
            eta_slot,
            model_slot,
            activity_slot,
            preview_slot,
            preview_meta_slot,
        )

        with st.expander("Live console", expanded=False):
            console_slot = st.empty()
            console = Console(
                console_slot,
                max_lines=st.session_state.theme_console_lines,
                log_path=st.session_state.get("run_log_path"),
                activity_callback=progress.touch_activity,
            )
            console.flush()
            st.caption("The complete pipeline log continues to be written even while this panel is collapsed.")

        return progress, console

    # Completed, failed, interrupted or cancelled: render a compact run report from the persisted log.
    title_map = {
        "success": ("Run complete", "The reconstruction pipeline completed successfully."),
        "error": ("Run failed", st.session_state.get("run_error") or "The pipeline returned an error."),
        "interrupted": ("Run interrupted", st.session_state.get("run_error") or "The run was interrupted."),
        "cancelled": ("Run cancelled", "No native reconstruction process is still running."),
    }
    title, subtitle = title_map.get(status, ("Reconstruction", "No run has been started yet."))
    st.markdown('<div class="eyebrow">Run report</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{html.escape(subtitle)}</div>', unsafe_allow_html=True)

    started = st.session_state.get("run_started_ts")
    finished = st.session_state.get("run_finished_ts") or time.time()
    if started:
        a, b, c = st.columns(3)
        total_runtime = float(st.session_state.get("run_accumulated_runtime_seconds") or 0.0) + max(0.0, finished - started)
        a.metric("Total runtime", format_duration(total_runtime))
        stage_no = int(st.session_state.get("run_stage_index") or pipeline_stage_for(st.session_state.get("run_label", ""), st.session_state.get("run_detail", "")))
        overall = float(st.session_state.get("run_progress", 0))
        stage_pct = pipeline_stage_progress(stage_no, overall)
        b.metric("Last pipeline stage", f"{stage_no} / {PIPELINE_STAGE_TOTAL}")
        b.caption(f"{st.session_state.get('run_label', '')} · {stage_pct:.1f}% through this stage")
        stats = st.session_state.get("run_stats") or {}
        persisted_state = load_project_state(project_dir)
        frame_value = stats.get("frame_count") or persisted_state.get("frame_count") or "—"
        c.metric("Frames", frame_value)

    if status == "success":
        st.markdown('<div class="success-banner">Output created and ready for review.</div>', unsafe_allow_html=True)
    elif status in {"error", "interrupted", "cancelled"}:
        st.markdown(f'<div class="error-banner">{html.escape(st.session_state.get("run_error") or subtitle)}</div>', unsafe_allow_html=True)
        checkpoint = training_checkpoint_info(project_dir)
        if checkpoint:
            step = int(checkpoint.get("step") or 0)
            total = int(checkpoint.get("total_steps") or 0)
            st.info(
                f"Native recovery snapshot available · {step:,}/{total:,}. "
                "Continue resumes the Gaussian model, Adam optimizer, refinement state, frame sampler and accumulated runtime."
            )

    log_text = load_log_text()
    diagnostics_json = json.dumps(diagnostics_payload(), indent=2, default=str)

    if status in {"error", "interrupted", "cancelled"}:
        checkpoint_info = training_checkpoint_info(project_dir)
        source_value = st.session_state.get("run_source_path") or latest_source_for_project(project_dir)
        config = st.session_state.get("run_config") or (load_project_state(project_dir).get("run_config") or {})

        st.markdown("#### What would you like to do?")
        action_columns = st.columns(4, gap="small")

        with action_columns[0]:
            if checkpoint_info and source_value:
                step = int(checkpoint_info.get("step") or 0)
                total = int(checkpoint_info.get("total_steps") or config.get("iterations") or 0)
                checkpoint_help = (
                    f"Resume the saved native Gaussian model and complete optimizer/refinement state from step {step:,} of {total:,}."
                    if total
                    else f"Resume the saved native Gaussian model and complete optimizer/refinement state from step {step:,}."
                )
                if st.button(
                    "Continue from last snapshot",
                    type="primary",
                    icon=":material/resume:",
                    key="failure_continue_checkpoint",
                    width="stretch",
                    help=checkpoint_help,
                ):
                    launch_project_run(project_dir, source_value, config, continue_mode=True)
                    st.rerun()
            else:
                st.button(
                    "Continue from last snapshot",
                    icon=":material/resume:",
                    key="failure_continue_unavailable",
                    width="stretch",
                    disabled=True,
                    help="No compatible native training snapshot exists yet.",
                )

        with action_columns[1]:
            if st.button(
                "Restart Project Again",
                icon=":material/restart_alt:",
                key="failure_restart_project",
                width="stretch",
                help="Start the same project again from the beginning using the same captured settings. Existing reconstruction/training output will be rebuilt.",
            ):
                if source_value:
                    restart_config = dict(config)
                    restart_config["reuse_camera"] = False
                    launch_project_run(project_dir, source_value, restart_config, continue_mode=False)
                    st.rerun()

        with action_columns[2]:
            if st.button(
                "Back to Prepare Page",
                icon=":material/tune:",
                key="failure_back_prepare",
                width="stretch",
                help="Return to settings without starting anything.",
            ):
                st.session_state.page = "prepare"
                st.rerun()

        with action_columns[3]:
            with st.popover("Logs / Debug", icon=":material/bug_report:", width="stretch"):
                st.caption("Recent pipeline output")
                st.code("\n".join(log_text.splitlines()[-st.session_state.theme_console_lines:]), language="text")

                if st.session_state.get("run_traceback"):
                    st.markdown("##### Technical error")
                    st.code(st.session_state.run_traceback, language="text")

                d1, d2 = st.columns(2)
                with d1:
                    st.download_button(
                        "Full log",
                        data=log_text,
                        file_name=f"{slugify(st.session_state.project_name)}-pipeline.log",
                        mime="text/plain",
                        width="stretch",
                    )
                with d2:
                    st.download_button(
                        "Diagnostics",
                        data=diagnostics_json,
                        file_name=f"{slugify(st.session_state.project_name)}-diagnostics.json",
                        mime="application/json",
                        width="stretch",
                    )

        if checkpoint_info:
            step = int(checkpoint_info.get("step") or 0)
            total = int(checkpoint_info.get("total_steps") or config.get("iterations") or 0)
            saved_text = f"step {step:,}/{total:,}" if total else f"step {step:,}"
            st.success(f"Recovery available · latest training snapshot is {saved_text}.")

        return None, None

    # Successful runs keep logs available without cluttering the main review path.
    with st.expander("Pipeline log", expanded=False):
        st.code("\n".join(log_text.splitlines()[-st.session_state.theme_console_lines:]), language="text")
        d1, d2 = st.columns(2)
        with d1:
            st.download_button("Download full log", data=log_text, file_name=f"{slugify(st.session_state.project_name)}-pipeline.log", mime="text/plain", width="stretch")
        with d2:
            st.download_button("Download diagnostics", data=diagnostics_json, file_name=f"{slugify(st.session_state.project_name)}-diagnostics.json", mime="application/json", width="stretch")

    left, right = st.columns(2)
    with left:
        if st.button("Back to Prepare Page", width="stretch"):
            st.session_state.page = "prepare"
            st.rerun()
    with right:
        if status == "success" and st.session_state.get("active_splat"):
            if st.button("Review result", type="primary", width="stretch"):
                st.session_state.page = "review"
                st.rerun()

    return None, None


def active_splat_path():
    value = st.session_state.get("active_splat") or st.session_state.get("last_output")
    if not value:
        return None
    path = Path(value)
    return path if path.exists() else None


def load_training_meta(path):
    path = Path(path)
    candidate = path.with_suffix(".training.json") if path.suffix.lower() == ".ply" else path.with_name("splat.training.json")
    if not candidate.exists():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def render_review(local_asset_base):
    path = active_splat_path()
    if not path:
        st.info("No result is available yet.")
        if st.button("Return to Prepare"):
            st.session_state.page = "prepare"
            st.rerun()
        return

    project_dir = current_project_dir()
    if path.suffix.lower() == ".spz" and path.name.lower() == "splat.spz":
        try:
            with st.spinner("Checking splat orientation…"):
                path = ensure_generated_splat_upright(path, project_dir)
                st.session_state.active_splat = str(path)
        except Exception as exc:
            st.warning(f"Automatic upright correction could not be applied yet: {exc}")

    reconstruction_meta = load_reconstruction_meta(project_dir)
    frame_count = int(reconstruction_meta.get("frame_count") or st.session_state.get("run_stats", {}).get("frame_count") or 0)
    health = camera_solve_health(project_dir, frame_count) if frame_count else {"level":"warn","message":"Camera coverage information is unavailable.","registered":0,"points":0,"frame_count":0,"registration_ratio":0.0,"metrics":{}}
    metrics = health.get("metrics", {})
    training = load_training_meta(path)
    gaussian_count = int(training.get("gaussians") or 0)

    st.markdown('<div class="eyebrow">Reconstruction result</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Review</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Inspect the result, then continue to cleanup/editing or rebuild if camera coverage is weak.</div>', unsafe_allow_html=True)

    health_class = {"good":"health-good","warn":"health-warn","bad":"health-bad"}.get(health["level"], "health-warn")
    registered_text = f"{health['registered']}/{health['frame_count']} frames ({health['registration_ratio']:.1%})" if health["frame_count"] else f"{health['registered']} cameras"
    gaussian_text = f" · {gaussian_count:,} trained Gaussians" if gaussian_count else ""
    guide_html = f"<div class='review-guide {health_class}'><strong>{html.escape(health['message'])}</strong><br><span>Camera coverage: {registered_text} · {health['points']:,} sparse points{gaussian_text}. Drag the preview to orbit, scroll to zoom, then choose Edit when the object looks coherent.</span></div>"
    st.markdown(guide_html, unsafe_allow_html=True)

    a1, a2, a3, a4 = st.columns([1.35, 1, 1, 1.25], gap="small")
    with a1:
        if health["level"] == "bad":
            if st.button("Rebuild with Balanced", type="primary", icon=":material/refresh:", width="stretch"):
                apply_preset("Balanced"); save_settings(); st.session_state.page = "prepare"; st.rerun()
        else:
            if st.button("Edit in SuperSplat", type="primary", icon=":material/edit:", width="stretch"):
                st.session_state.page = "edit"; st.rerun()
    with a2:
        if st.button("Open viewer", icon=":material/visibility:", width="stretch"):
            st.session_state.page = "view"; st.rerun()
    with a3:
        if sys.platform == "darwin" and st.button("Reveal in Finder", icon=":material/folder_open:", width="stretch"):
            subprocess.Popen(["open", "-R", str(path.resolve())])
    with a4:
        if st.button("Rebuild / settings", icon=":material/tune:", width="stretch"):
            st.session_state.page = "prepare"; st.rerun()

    st.markdown('<div class="review-hint">Preview loads automatically. If it is empty or shows only a few dots, check camera coverage and Gaussian count before editing.</div>', unsafe_allow_html=True)

    splat_choices = supported_splat_files(project_dir)
    if len(splat_choices) > 1:
        labels = [str(item.relative_to(project_dir)) if project_dir in item.parents else item.name for item in splat_choices]
        default_index = splat_choices.index(path) if path in splat_choices else 0
        selected = st.selectbox("Active result", labels, index=default_index, help="Switch between generated or imported splats stored in this project.")
        selected_path = splat_choices[labels.index(selected)]
        if selected_path != path:
            st.session_state.active_splat = str(selected_path); path = selected_path; training = load_training_meta(path)

    left, right = st.columns([1.75, 0.72], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="card-title">Interactive preview</div>', unsafe_allow_html=True)
            best_view = resolve_best_view(project_dir, training) or {}
            best_image = best_view_source_image(project_dir, training)
            if best_view:
                st.caption(
                    f"Recommended capture view · {best_view.get('image_name', 'camera')} · "
                    f"PSNR {float(best_view.get('psnr') or 0):.1f} dB · "
                    f"{int(best_view.get('visible_gaussians') or 0):,} visible splats"
                )
            else:
                st.caption("Drag to orbit · scroll to zoom · double-click a useful point to change the orbit centre.")

            pc = playcanvas_status()
            if pc["viewer"]:
                try:
                    if path.suffix.lower() in {".spz", ".splat", ".ksplat"}:
                        with st.spinner("Preparing lightweight interactive preview…"):
                            preview_path = viewer_compatible_preview_path(path)
                    else:
                        preview_path = path
                    settings_path = best_view_settings_path(project_dir, training)
                    nonce = int(st.session_state.get("review_view_nonce") or 0)
                    st.iframe(
                        viewer_url_for(preview_path, local_asset_base, show_ui=True, settings_path=settings_path, nonce=nonce),
                        height=700,
                        width="stretch",
                    )
                    if best_view:
                        reset_col, image_col = st.columns([0.72, 1.28], vertical_alignment="center")
                        with reset_col:
                            if st.button("Reset to recommended view", icon=":material/center_focus_strong:", width="stretch"):
                                st.session_state.review_view_nonce = nonce + 1
                                st.rerun()
                        with image_col:
                            if best_image and best_image.exists():
                                st.caption(f"Reference frame: {best_view.get('image_name', '')}")
                except Exception as exc:
                    st.warning(f"Lightweight preview could not be prepared: {exc}")
                    st.caption("Falling back to the SuperSplat editor renderer, which can open SPZ directly.")
                    st.iframe(editor_url_for(path, local_asset_base), height=700, width="stretch")
            elif pc["editor"]:
                st.caption("Viewer build unavailable; using the SuperSplat editor renderer for preview.")
                st.iframe(editor_url_for(path, local_asset_base), height=700, width="stretch")
            else:
                st.info("Install the local SuperSplat tools from Settings to enable the integrated 3D preview.")
    with right:
        with st.container(border=True):
            st.markdown('<div class="card-title">Result health</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-path">{html.escape(str(path.resolve()))}</div>', unsafe_allow_html=True)
            x, y = st.columns(2); x.metric("Size", human_size(path.stat().st_size)); y.metric("Format", path.suffix[1:].upper())
            run_started = st.session_state.get("run_started_ts"); run_finished = st.session_state.get("run_finished_ts")
            if run_started and run_finished: st.metric("Build time", format_duration(run_finished - run_started))
            if frame_count: st.metric("Sampled frames", f"{frame_count:,}")
            if health.get("registered"): st.metric("Registered cameras", f"{health['registered']:,}")
            if health.get("points"): st.metric("Sparse points", f"{health['points']:,}")
            if gaussian_count: st.metric("Final Gaussians", f"{gaussian_count:,}")
            if training.get("first_view_psnr_db") is not None: st.metric("First-view PSNR", f"{float(training['first_view_psnr_db']):.1f} dB")
            if metrics.get("reprojection_error"): st.metric("Reprojection error", metrics["reprojection_error"])

    with st.expander("Run configuration"):
        st.json(st.session_state.get("run_config") or reconstruction_meta.get("config", {}))


def render_edit(local_asset_base):
    path = active_splat_path()
    project_dir = current_project_dir()
    if path and path.suffix.lower() == ".spz" and path.name.lower() == "splat.spz":
        try:
            path = ensure_generated_splat_upright(path, project_dir)
            st.session_state.active_splat = str(path)
        except Exception as exc:
            st.warning(f"Automatic upright correction could not be applied yet: {exc}")

    st.markdown('<div class="eyebrow">Local SuperSplat workspace</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Edit</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Clean floaters, crop, transform, colour-correct and prepare the splat using the locally built SuperSplat editor.</div>', unsafe_allow_html=True)

    pc = playcanvas_status()
    if not pc["editor"]:
        st.info("SuperSplat Editor is not installed. Open Settings and install the local tools.")
        return

    editor_url = editor_url_for(path, local_asset_base) if path else f"{local_asset_base}/editor/index.html"
    c1, c2, c3 = st.columns([1, 1, 4])
    with c1:
        if st.button("Review", width="stretch"):
            st.session_state.page = "review"
            st.rerun()
    with c2:
        if st.button("View", width="stretch"):
            st.session_state.page = "view"
            st.rerun()
    with c3:
        st.link_button("Open editor fullscreen", editor_url, width="stretch")
    st.iframe(editor_url, height=900, width="stretch")

    with st.expander("Bring an exported edit back into this project"):
        st.caption("If SuperSplat exports a new file through the browser, import it here to make that edited file the active Review/View result.")
        edited_upload = st.file_uploader(
            "Import edited splat",
            type=["ply", "sog", "splat", "ksplat", "spz"],
            key="edited_splat_uploader",
        )
        if edited_upload is not None:
            imported_path = save_imported_splat(edited_upload, current_project_dir())
            st.session_state.active_splat = str(imported_path)
            st.session_state.page = "review"
            st.rerun()


def render_view(local_asset_base):
    path = active_splat_path()
    st.markdown('<div class="eyebrow">Final inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">View</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Inspect the active splat with the lightweight local viewer and open a distraction-free fullscreen view when needed.</div>', unsafe_allow_html=True)

    pc = playcanvas_status()
    if not path:
        st.info("No active splat is available.")
        return
    if not pc["viewer"] and not pc["editor"]:
        st.info("SuperSplat tools are not installed. Open Settings and install the local tools.")
        return

    project_dir = current_project_dir()

    if path.suffix.lower() == ".spz" and path.name.lower() == "splat.spz":
        try:
            with st.spinner("Checking splat orientation…"):
                path = ensure_generated_splat_upright(path, project_dir)
                st.session_state.active_splat = str(path)
        except Exception as exc:
            st.warning(f"Automatic upright correction could not be applied yet: {exc}")

    controls = st.toggle(
        "Viewer controls",
        value=True,
        help="Show SuperSplat's built-in navigation and viewer controls.",
    )

    viewer_url = None
    viewer_error = None
    if pc["viewer"]:
        try:
            with st.spinner("Preparing viewer…"):
                preview_path = viewer_compatible_preview_path(path)
                training = load_training_meta(path)
                settings_path = best_view_settings_path(project_dir, training)
                nonce = int(st.session_state.get("view_page_nonce") or 0)
                viewer_url = viewer_url_for(
                    preview_path,
                    local_asset_base,
                    show_ui=controls,
                    settings_path=settings_path,
                    nonce=nonce,
                )
        except Exception as exc:
            viewer_error = str(exc)

    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 3.8])
    with c1:
        if st.button("Review", width="stretch"):
            st.session_state.page = "review"
            st.rerun()
    with c2:
        if st.button("Edit", width="stretch"):
            st.session_state.page = "edit"
            st.rerun()
    with c3:
        if viewer_url and st.button("Reset view", icon=":material/center_focus_strong:", width="stretch"):
            st.session_state.view_page_nonce = int(st.session_state.get("view_page_nonce") or 0) + 1
            st.rerun()
    with c4:
        if viewer_url:
            st.link_button("Open viewer fullscreen", viewer_url, width="stretch")
        elif pc["editor"]:
            st.link_button("Open editor fullscreen", editor_url_for(path, local_asset_base), width="stretch")

    if viewer_url:
        st.iframe(viewer_url, height=900, width="stretch")
    elif pc["editor"]:
        if viewer_error:
            st.warning(f"Lightweight viewer could not be prepared: {viewer_error}")
        st.caption("Falling back to the SuperSplat editor renderer.")
        st.iframe(editor_url_for(path, local_asset_base), height=900, width="stretch")
    else:
        st.warning(viewer_error or "The local viewer could not be prepared.")



def execute_pipeline(progress, console):
    project_dir = run_project_dir()
    source_value = st.session_state.get("run_source_path") or st.session_state.get("source_path")
    if not source_value or not Path(source_value).exists():
        raise PipelineError("Source is missing. Return to Prepare and select the video or photos again.")

    source_path = Path(source_value)
    photo_mode = source_path.is_dir()
    source_kind = "photos" if photo_mode else "video"
    config = st.session_state.get("run_config") or {}

    payload = {
        "source_sha1": source_fingerprint(source_path),
        "source_type": source_kind,
        "target_frames": int(config.get("target_frames", 300)),
        "overlap": int(config.get("overlap", 12)),
        "max_features": int(config.get("max_features", 4096)),
        "max_image_size": int(config.get("max_image_size", 1920)),
        "shared_camera": bool(config.get("shared_camera", True)),
        "camera_model": str(config.get("camera_model", "SIMPLE_RADIAL")),
        "quadratic_overlap": bool(config.get("quadratic_overlap", True)),
        "guided_matching": bool(config.get("guided_matching", False)),
        "robust_sift": bool(config.get("robust_sift", False)),
        "smart_frames": bool(config.get("smart_frames", True)),
    }

    expected_hash = config_hash(payload)
    reuse = bool(config.get("reuse_camera", True)) and can_reuse_camera_solve(project_dir, expected_hash)
    checkpoint = training_checkpoint_info(project_dir)
    if checkpoint and (project_dir / "images").exists() and sparse_model_dir(project_dir) is not None:
        reuse = True

    save_project_state(
        project_dir,
        status="running",
        phase="reconstruct",
        run_config=config,
        source_path=str(source_path.resolve()),
        source_type=source_kind,
    )

    console.write(f"[studio] Project: {project_dir.resolve()}", force=True)
    console.write(f"[studio] Source ({source_kind}): {source_path.resolve()}")
    console.write(f"[studio] Max frames: {config.get('target_frames')} · overlap: {config.get('overlap')} · SIFT: {config.get('max_features')}")

    if reuse:
        meta = load_reconstruction_meta(project_dir)
        frame_count = int(meta.get("frame_count") or len(list((project_dir / "images").glob("*.jpg"))))
        sample_fps = float(meta.get("sample_fps") or 0)
        reused_health = camera_solve_health(project_dir, frame_count)
        if reused_health["level"] == "bad":
            console.write("[studio] Existing camera solve is too weak to reuse; rebuilding automatically.", force=True)
            reuse = False
        else:
            console.write("[studio] Compatible camera solve found — reusing COLMAP output.")
            update_camera_run_stats(project_dir, frame_count, sample_fps, reused_health)
            progress.update(50, "Camera solve reused", f"{frame_count:,} source images already solved", force=True)

    if not reuse:
        progress.update(2, "Preparing project", "Clearing stale reconstruction data", force=True)
        clean_project(project_dir)
        images_dir = project_dir / "images"

        if photo_mode:
            frame_count, sample_fps = prepare_photo_frames(
                source_path,
                images_dir,
                progress,
                console,
                int(config.get("target_frames", 300)),
                bool(config.get("smart_frames", True)),
            )
        else:
            frame_count, sample_fps = extract_frames(
                source_path,
                images_dir,
                progress,
                console,
                int(config.get("target_frames", 300)),
                bool(config.get("smart_frames", True)),
            )

        update_camera_run_stats(project_dir, frame_count, sample_fps)
        if frame_count < 2:
            raise PipelineError("Source preparation produced fewer than two usable images.")

        run_colmap(
            project_dir,
            images_dir,
            progress,
            console,
            int(config.get("overlap", 12)),
            int(config.get("max_features", 4096)),
            int(config.get("max_image_size", 1920)),
            bool(config.get("shared_camera", True)),
            str(config.get("camera_model", "SIMPLE_RADIAL")),
            bool(config.get("quadratic_overlap", True)),
            bool(config.get("guided_matching", False)),
            bool(config.get("robust_sift", False)),
            photo_mode=photo_mode,
        )

        health = camera_solve_health(project_dir, frame_count)
        update_camera_run_stats(project_dir, frame_count, sample_fps, health)
        console.write(
            f"[studio] Camera solve: {health['registered']}/{frame_count} registered "
            f"({health['registration_ratio']:.1%}), {health['points']:,} sparse points.",
            force=True,
        )

        if health["level"] != "good":
            repair_colmap_connectivity(
                project_dir,
                images_dir,
                progress,
                console,
                int(config.get("overlap", 12)),
            )
            health = camera_solve_health(project_dir, frame_count)
            update_camera_run_stats(project_dir, frame_count, sample_fps, health, rescue_mode="wider_sequential")
            console.write(
                f"[studio] Camera repair: {health['registered']}/{frame_count} registered "
                f"({health['registration_ratio']:.1%}), {health['points']:,} sparse points.",
                force=True,
            )

        if health["level"] == "bad" and bool(config.get("auto_camera_rescue", True)):
            frame_count, sample_fps, health = automatic_camera_rescue(
                source_path,
                project_dir,
                config,
                progress,
                console,
                frame_count,
            )
            images_dir = project_dir / "images"

        if health["level"] == "bad":
            rescue_note = (
                " Automatic camera rescue was attempted but the source still did not contain enough connected visual geometry."
                if bool(config.get("auto_camera_rescue", True))
                else ""
            )
            raise PipelineError(
                "Camera reconstruction is too weak to train a useful Gaussian splat. "
                f"COLMAP registered only {health['registered']} of {frame_count} source images ({health['registration_ratio']:.1%}) "
                f"with {health['points']} sparse points.{rescue_note} "
                "Use more overlapping views, avoid large viewpoint jumps, and keep textured scene features visible between shots."
            )

        save_reconstruction_meta(project_dir, payload, frame_count, sample_fps)

    progress.update(
        58,
        "Validating cameras",
        f"Checking registration coverage and sparse geometry across {frame_count:,} source images",
        force=True,
    )
    health = camera_solve_health(project_dir, frame_count)

    final_stats = dict(st.session_state.get("run_stats") or {})
    final_stats.update({
        "frame_count": frame_count,
        "sample_fps": sample_fps,
        "source_type": source_kind,
        "reused_camera_solve": reuse,
        "registered_cameras": health.get("registered", 0),
        "sparse_points": health.get("points", 0),
        "registration_ratio": health.get("registration_ratio", 0.0),
        "camera_health": health.get("level", "unknown"),
    })
    st.session_state.run_stats = final_stats

    save_project_state(
        project_dir,
        status="running",
        phase="training",
        source_type=source_kind,
        run_config=config,
        frame_count=frame_count,
        registered_cameras=health.get("registered", 0),
        sparse_points=health.get("points", 0),
    )

    output_file = run_training(
        project_dir,
        int(config.get("iterations", 2500)),
        float(config.get("means_lr", 0.00016)),
        int(config.get("mlx_batch_size", 1)),
        int(config.get("mlx_max_points", 3500)),
        int(config.get("mlx_max_gaussians", 12000)),
        int(config.get("mlx_train_max_side", 256)),
        float(config.get("mlx_ssim_weight", 0.20)),
        bool(config.get("mlx_refine", True)),
        bool(config.get("mlx_antialiased", False)),
        int(config.get("checkpoint_every", 250)),
        int(config.get("preview_every", 250)),
        bool(config.get("auto_resume", True)),
        bool(config.get("ai_depth", False)),
        int(config.get("ai_depth_points", 40000)),
        bool(config.get("ai_mask_sky", False)),
        bool(config.get("ai_mask_dynamic", False)),
        progress,
        console,
    )

    if not output_file:
        raise PipelineError("Training finished, but no SPZ output was detected. Check the pipeline log for the native trainer output.")

    st.session_state.last_output = str(output_file)
    st.session_state.active_splat = str(output_file)
    persist_current_runtime(
        project_dir,
        status="complete",
        phase="complete",
        output_path=str(output_file.resolve()),
        completed_at=time.time(),
        run_config=config,
        run_stats=st.session_state.run_stats,
        last_error=None,
    )
    progress.update(100, "Complete", output_file.name, force=True)
    return output_file



# ---------------- persistent preferences ----------------
SETTINGS_SCHEMA_VERSION = 14

THEME_SETTING_KEYS = (
    "theme_mode",
    "theme_accent",
    "theme_background",
    "theme_custom_background",
    "theme_scale",
    "theme_console_size",
    "theme_console_lines",
    "theme_reduce_motion",
)

PERSISTED_SETTING_KEYS = (
    "project_name",
    "cfg_preset",
    "cfg_target_frames",
    "cfg_overlap",
    "cfg_max_features",
    "cfg_max_image_size",
    "cfg_iterations",
    "cfg_means_lr",
    "cfg_mlx_batch_size",
    "cfg_mlx_max_points",
    "cfg_mlx_max_gaussians",
    "cfg_mlx_train_max_side",
    "cfg_mlx_ssim_weight",
    "cfg_mlx_antialiased",
    "cfg_mlx_refine",
    "cfg_checkpoint_every",
    "cfg_preview_every",
    "cfg_auto_resume",
    "cfg_smart_frames",
    "cfg_ai_depth",
    "cfg_ai_depth_points",
    "cfg_ai_mask_sky",
    "cfg_ai_mask_dynamic",
    "cfg_camera_model",
    "cfg_shared_camera",
    "cfg_reuse_camera",
    "cfg_quadratic_overlap",
    "cfg_guided_matching",
    "cfg_robust_sift",
    "cfg_auto_camera_rescue",
    "cfg_prevent_sleep",
    "theme_mode",
    "theme_accent",
    "theme_background",
    "theme_custom_background",
    "theme_scale",
    "theme_console_size",
    "theme_console_lines",
    "theme_reduce_motion",
)


def load_saved_settings():
    """Load local preferences and safely migrate settings from older trainer builds."""
    if not SETTINGS_FILE.exists():
        return {}

    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}

        schema = int(data.get("_schema_version", 0) or 0)
        if schema < SETTINGS_SCHEMA_VERSION:
            # V19.0 migrates the old RobotFlow training controls to native Metal defaults.
            data["cfg_mlx_max_points"] = 0
            data["cfg_iterations"] = 4000
            data["cfg_mlx_train_max_side"] = 480
            data["cfg_mlx_batch_size"] = 1
            data["cfg_mlx_ssim_weight"] = 0.20
            data["cfg_mlx_refine"] = True
            data["cfg_checkpoint_every"] = 250
            data["cfg_preview_every"] = 250
            data["cfg_auto_resume"] = True
            data.setdefault("cfg_smart_frames", True)
            data.setdefault("cfg_ai_depth", False)
            data.setdefault("cfg_ai_depth_points", 40000)
            data.setdefault("cfg_ai_mask_sky", False)
            data.setdefault("cfg_ai_mask_dynamic", False)
            data.setdefault("cfg_auto_camera_rescue", True)
            data.pop("cfg_mlx_cache_gb", None)
        return data
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}


def save_settings():
    """Persist appearance and reconstruction preferences between app launches."""
    data = {
        key: st.session_state.get(key)
        for key in PERSISTED_SETTING_KEYS
        if key in st.session_state
    }
    data["_schema_version"] = SETTINGS_SCHEMA_VERSION

    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SETTINGS_FILE.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(SETTINGS_FILE)
    except OSError:
        # Settings persistence should never break reconstruction.
        pass


def reset_saved_settings():
    """Remove persistent preferences and restore defaults on the next rerun."""
    try:
        SETTINGS_FILE.unlink(missing_ok=True)
    except OSError:
        pass

    for key in PERSISTED_SETTING_KEYS:
        st.session_state.pop(key, None)


def _shutdown_worker():
    """Give Streamlit enough time to send the final response before exiting."""
    time.sleep(0.8)
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except OSError:
        os._exit(0)


def close_splat_studio():
    """Save preferences then stop the local Streamlit process."""
    save_settings()

    project_dir = current_project_dir()
    stop_stale_pipeline_process(project_dir)

    thread = threading.Thread(
        target=_shutdown_worker,
        daemon=True,
        name="splat-studio-shutdown",
    )
    thread.start()


# ---------------- application state ----------------
def init_state():
    defaults = {
        "page": "prepare",
        "nav_choice": "Prepare",
        "nav_page_snapshot": "prepare",
        "run_status": "idle",
        "run_launch_pending": False,
        "run_error": None,
        "run_traceback": None,
        "run_config": {},
        "run_stats": {},
        "run_accumulated_runtime_seconds": 0.0,
        "run_model_gaussians": None,
        "run_model_loss": None,
        "run_training_step": 0,
        "run_preview_path": None,
        "review_view_nonce": 0,
        "project_delete_confirm": None,
        "run_preview_step": 0,
        "run_preview_updated_ts": 0.0,
        "run_preview_kind": None,
        "run_ai_depth_points": 0,
        "run_stage_index": 1,
        "run_label": "Preparing",
        "run_detail": "Starting reconstruction",
        "run_eta_seconds": None,
        "run_eta_calc_ts": 0.0,
        "run_native_eta_seconds": None,
        "run_native_eta_updated_ts": 0.0,
        "run_project_dir": None,
        "run_source_path": None,
        "last_output": None,
        "active_splat": None,
        "source_path": None,
        "source_mode": "Video",
        "project_name": "my-scene",
        "cfg_preset": "Balanced",
        "cfg_target_frames": 300,
        "cfg_overlap": 12,
        "cfg_max_features": 4096,
        "cfg_max_image_size": 1920,
        "cfg_iterations": 4000,
        "cfg_means_lr": 0.00016,
        "cfg_mlx_batch_size": 1,
        "cfg_mlx_max_points": 0,
        "cfg_mlx_max_gaussians": 12000,
        "cfg_mlx_train_max_side": 480,
        "cfg_mlx_ssim_weight": 0.20,
        "cfg_mlx_refine": True,
        "cfg_checkpoint_every": 250,
        "cfg_preview_every": 250,
        "cfg_auto_resume": True,
        "cfg_smart_frames": True,
        "cfg_ai_depth": False,
        "cfg_ai_depth_points": 40000,
        "cfg_ai_mask_sky": False,
        "cfg_ai_mask_dynamic": False,
        "cfg_mlx_antialiased": False,
        "cfg_camera_model": "SIMPLE_RADIAL",
        "cfg_shared_camera": True,
        "cfg_reuse_camera": True,
        "cfg_quadratic_overlap": True,
        "cfg_guided_matching": False,
        "cfg_robust_sift": False,
        "cfg_auto_camera_rescue": True,
        "cfg_prevent_sleep": True,
        "theme_mode": "Dark",
        "theme_accent": "#67D7FF",
        "theme_background": "Midnight",
        "theme_custom_background": "#10141C",
        "theme_scale": 100,
        "theme_console_size": 12,
        "theme_console_lines": 220,
        "theme_reduce_motion": False,
    }

    saved = load_saved_settings()
    for key, value in defaults.items():
        if key in PERSISTED_SETTING_KEYS and key in saved:
            st.session_state.setdefault(key, saved[key])
        else:
            st.session_state.setdefault(key, value)

    # Appearance is application-global, not page-local. Re-apply the saved
    # theme at the start of every rerun so Retry/navigation/native-process
    # transitions cannot silently fall back to Streamlit/default colours.
    for key in THEME_SETTING_KEYS:
        if key in saved:
            st.session_state[key] = saved[key]


init_state()
render_theme_css()

# If an existing project already has an input video, restore it before rendering.
project_dir = current_project_dir()
project_dir.mkdir(parents=True, exist_ok=True)
if not st.session_state.get("source_path") or not Path(st.session_state.source_path).exists():
    existing_source = latest_source_for_project(project_dir)
    if existing_source:
        st.session_state.source_path = str(existing_source)

# Restore durable project status after app/browser/Mac restarts.
startup_progress = inspect_project_progress(project_dir)
if startup_progress["status"] == "complete" and startup_progress.get("output"):
    st.session_state.active_splat = str(startup_progress["output"])
    st.session_state.last_output = str(startup_progress["output"])
elif startup_progress["status"] in {"checkpoint", "camera", "incomplete"}:
    saved_config = (startup_progress.get("state") or {}).get("run_config") or {}
    if saved_config and not st.session_state.get("run_config"):
        st.session_state.run_config = saved_config

# Tiny local HTTP server for the self-hosted SuperSplat editor/viewer and project files.
_asset_server, _asset_port = start_asset_server()
LOCAL_ASSET_BASE = f"http://127.0.0.1:{_asset_port}"

st.markdown('<div class="app-shell">', unsafe_allow_html=True)
render_header()

run_progress = None
run_console = None

if st.session_state.page == "prepare":
    render_prepare()
elif st.session_state.page == "projects":
    render_projects()
elif st.session_state.page == "run":
    run_progress, run_console = render_run()
elif st.session_state.page == "review":
    render_review(LOCAL_ASSET_BASE)
elif st.session_state.page == "edit":
    render_edit(LOCAL_ASSET_BASE)
elif st.session_state.page == "view":
    render_view(LOCAL_ASSET_BASE)
else:
    st.session_state.page = "prepare"
    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Persist changed appearance/reconstruction settings automatically.
save_settings()

# Launch exactly once after Start/Retry. A browser refresh does not silently restart the pipeline.
if (
    st.session_state.page == "run"
    and st.session_state.run_status == "running"
    and st.session_state.get("run_launch_pending")
    and run_progress is not None
    and run_console is not None
):
    st.session_state.run_launch_pending = False
    try:
        output = execute_pipeline(run_progress, run_console)
        st.session_state.run_status = "success"
        st.session_state.run_finished_ts = time.time()
        st.session_state.run_error = None
        record_run_history("success")
        st.session_state.page = "review"
        st.rerun()
    except Exception as exc:
        # A Cancel click causes a rerun that deliberately kills this process group.
        # Do not let the old script race the newer cancelled state and relabel it as an error.
        if st.session_state.get("run_status") != "cancelled":
            tb = traceback.format_exc()
            run_console.write(f"[error] {exc}", force=True)
            run_console.write(tb, force=True)
            st.session_state.run_status = "error"
            st.session_state.run_finished_ts = time.time()
            st.session_state.run_error = str(exc)
            st.session_state.run_traceback = tb
            checkpoint = training_checkpoint_info(run_project_dir())
            persist_current_runtime(
                run_project_dir(),
                status="error",
                phase="training" if checkpoint else "reconstruct",
                last_error=str(exc),
                last_checkpoint_step=(checkpoint or {}).get("step", 0),
                total_steps=(checkpoint or {}).get("total_steps", 0),
                run_config=st.session_state.get("run_config") or {},
            )
            record_run_history("error", str(exc))
            st.session_state.page = "run"
            st.rerun()
