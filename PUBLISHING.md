# Publishing Splat Studio to GitHub

## Description

Local macOS workstation for creating, reviewing and editing 3D Gaussian Splats
from videos or photo sets using COLMAP, MLX, native Metal acceleration,
AI-assisted preprocessing and SuperSplat.

## Suggested topics

`gaussian-splatting`, `3dgs`, `apple-silicon`, `metal`, `mlx`, `colmap`,
`streamlit`, `supersplat`, `photogrammetry`, `computer-vision`,
`3d-reconstruction`, `macos`, `depth-estimation`, `ai`

## Before the first push

Check for local/private paths or data:

```bash
git status
git grep -n "/Users/" || true
git grep -ni "blondothenerd" || true
```

Do not commit:

- runtime environments
- compiled backend copies
- AI model caches
- projects
- videos/photos
- generated splats
- local settings/logs
- credentials

Then:

```bash
git init -b main
git add .
git status
git commit -m "Initial public release of Splat Studio"
```

With GitHub CLI:

```bash
gh repo create SplatStudio --public --source=. --remote=origin --push
```

The repository uses The Unlicense for Splat Studio's original source.
Third-party notices must remain intact.
