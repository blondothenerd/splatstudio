#!/usr/bin/env python3
from pathlib import Path
import platform, shutil, subprocess, sys
root = Path(__file__).resolve().parents[1]
items = {
    "macOS": platform.system() == "Darwin",
    "Apple Silicon": platform.machine() == "arm64",
    "runtime Python": (root / "runtime/gsplat_core/bin/python").exists(),
    "native backend": (root / "backend/gsplat-metal").exists(),
    "FFmpeg": bool(shutil.which("ffmpeg")),
    "COLMAP": bool(shutil.which("colmap")),
    "Node": bool(shutil.which("node")),
    "npm": bool(shutil.which("npm")),
    "Git": bool(shutil.which("git")),
}
for name, ok in items.items(): print(f"{'OK' if ok else 'MISSING':7} {name}")
sys.exit(0 if all(items.values()) else 1)
