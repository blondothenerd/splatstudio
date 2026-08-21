# Splat Studio

<p align="center">
  <img src="SplatStudio_icon.png" alt="Splat Studio icon" width="128">
</p>

<p align="center">
  <strong>Local 3D Gaussian Splat reconstruction for Apple Silicon.</strong>
</p>

<p align="center">Created by <strong>blondothenerd</strong>.</p>

<p align="center">
  <img src="docs/images/image1.png" alt="Splat Studio interface" width="100%">
</p>


Splat Studio is a local macOS workstation for creating, reviewing and editing
**3D Gaussian Splats** from videos or still-photo sets.

It combines **COLMAP camera reconstruction**, native **MLX / C++ / Metal**
Gaussian Splatting, optional **AI-assisted preprocessing**, resumable training,
live previews and local **SuperSplat** editing in one interface.

---

## Highlights

- Video or still-photo reconstruction
- Smart frame/photo curation
- COLMAP sparse reconstruction and automatic camera rescue
- Native Apple Silicon MLX / C++ / Metal Gaussian training
- Full-scene initialization from COLMAP sparse points
- Optional Depth Anything V2 assistance
- Optional semantic masking
- Training snapshots and resume
- Unattended automatic retry after training errors
- Lightweight live training previews
- Multiple reconstruction profiles
- SuperSplat review, editing and fullscreen viewing
- Recommended camera viewpoints
- Automatic orientation correction
- Local project management
- Dark, Light and Custom themes
- Local-only project storage

---

## Requirements

Splat Studio currently targets:

- **macOS**
- **Apple Silicon** (M-series)
- the full **Xcode** application
- Apple's **Metal Toolchain**

The installer handles the other normal dependencies for you.

> Windows, Linux and Intel Macs are not currently supported by the packaged
> native Metal workflow.

---

# 🚀 Easy installation

For most users, **you do not need to manually install Python, COLMAP, FFmpeg,
Node.js, MLX or the native Gaussian backend.**

### 1. Download Splat Studio

Either use Git:

```bash
git clone https://github.com/blondothenerd/splatstudio.git
cd splatstudio
```

or on GitHub choose:

**Code → Download ZIP**

and extract the folder somewhere you want to keep it.

### 2. Double-click the installer

Run:

```text
Install Splat Studio.command
```

The installer automatically checks and, where possible, installs or repairs:

- Homebrew
- Git
- FFmpeg
- COLMAP
- CMake
- Node.js / npm
- a private Miniforge runtime
- Python 3.11
- Splat Studio Python dependencies
- MLX
- the native C++ / Metal Gaussian backend
- SPZ support
- SuperSplat Editor
- SuperSplat Viewer
- SplatTransform
- optional AI dependencies and models, if selected

Network/build steps are retried automatically. A recoverable error does **not**
immediately abort the entire installation.

At the end, the installer performs a final health check and lists anything that
is still missing. It also writes:

```text
.splat_studio/install_report.txt
```

If something could not be installed, fix that item and simply run
`Install Splat Studio.command` again. The installer is designed to be safely
rerun.

### What about Xcode?

If full Xcode is already installed, Splat Studio will automatically:

- select the full Xcode developer directory
- complete Xcode first-launch setup where possible
- install/check the optional Metal Toolchain

If full Xcode is **not** installed, the installer opens Apple's Xcode page,
continues installing whatever else it can, and reports Xcode as missing at the
end.

Finish installing/opening Xcode, then rerun the same Splat Studio installer.

### Homebrew

If Homebrew is missing, Splat Studio automatically attempts to install it using
Homebrew's official installer. If macOS requires a password/confirmation, the
official installer may ask for it.

## 🛡️ macOS blocked the installer?

On some Macs, the first time you open one of Splat Studio's `.command` files,
macOS may show a message similar to:

> **“Install Splat Studio.command” was blocked to protect your Mac.**  
> Apple could not verify that the file is free of malware.

This is expected for a downloaded script that has not been signed/notarised as
a commercial Mac application.

### Easiest fix

1. Try to open `Install Splat Studio.command` normally.
2. When macOS blocks it, open **System Settings**.
3. Go to **Privacy & Security**.
4. Scroll down to the **Security** section.
5. You should see a message saying the Splat Studio `.command` file was blocked.
6. Click **Open Anyway**.
7. Enter your Mac password / use Touch ID if macOS asks.
8. Click **Open** when macOS asks one final time.

The installer should then start in Terminal.

> You normally only need to approve a particular downloaded script once.

### Alternative: right-click → Open

You can also try:

1. Find the `.command` file in Finder.
2. **Right-click / Control-click** it.
3. Choose **Open**.
4. Confirm **Open** again if macOS offers the option.

### If the file says it cannot be executed

ZIP downloads can occasionally lose the executable flag.

Open Terminal, drag the Splat Studio folder into Terminal if needed, then run:

```bash
cd /path/to/splatstudio
chmod +x *.command
```

Then double-click:

```text
Install Splat Studio.command
```

again.

### Important

Only bypass the macOS warning if you intentionally downloaded Splat Studio from
the official repository:

```text
https://github.com/blondothenerd/splatstudio
```

You can inspect every `.command` file in the repository before running it;
they are plain-text shell scripts.

### 3. Launch

When installation is complete, double-click:

```text
Launch Splat Studio.command
```

The Splat Studio interface opens in your browser.

---

# 🍎 Optional: create a Splat Studio.app

After Splat Studio has been installed, you can create a normal macOS app-style
launcher.

Double-click:

```text
Create Splat Studio App.command
```

It creates:

```text
Splat Studio.app
```

inside the Splat Studio folder.

The app creator automatically builds the macOS app icon from:

```text
SplatStudio_icon.png
```

and embeds it into the generated app.

You can then drag:

```text
Splat Studio.app
```

into:

```text
/Applications
```

if you want Splat Studio to appear alongside your other Mac applications.

### What the generated app does

When opened, the generated app:

- starts `SplatStudio.py` directly
- uses Splat Studio's private Python runtime
- restores Homebrew paths so COLMAP, FFmpeg, Node.js and related tools remain available
- restores the Xcode developer environment required by the Metal backend
- waits for the Streamlit server to become genuinely healthy
- opens the browser only after Splat Studio has successfully started
- offers to run `Install Splat Studio.command` if the local runtime is missing

If startup fails, the app does **not** simply open a dead localhost page.

Instead, it writes a launch log to:

```text
.splat_studio/app-launch.log
```

and offers to open that log for troubleshooting.

### Important

The generated `.app` is a **local launcher**. The Python runtime, backend,
projects, models and Splat Studio source remain inside the main Splat Studio
folder.

The app creator records the location of **your local** Splat Studio installation
inside the app when it is created. This local path is generated on the user's
Mac and is **not** stored in the public GitHub repository.

This means you can move `Splat Studio.app` itself into `/Applications`, while
the main Splat Studio folder can remain somewhere such as your Applications,
Documents or development folder.

If you later move or rename the main Splat Studio folder, simply run:

```text
Create Splat Studio App.command
```

again to regenerate the launcher with the new location.

The generated `.app` is ignored by Git and should not be committed to the
repository.

---

# Advanced / manual installation

The automatic installer is recommended. This section is mainly for
troubleshooting or users who want to understand/control each dependency.

## 1. Xcode and Metal

Install the full Xcode application from Apple and open it at least once.

Then:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch -checkForNewerComponents
xcodebuild -downloadComponent metalToolchain
```

Verify:

```bash
xcode-select --print-path
xcrun --find metal
xcodebuild -version
```

## 2. Homebrew

Install Homebrew using its official installer:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

On Apple Silicon, make Homebrew available to the current shell:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Install the system tools:

```bash
brew install git ffmpeg colmap cmake node
```

Verify:

```bash
git --version
ffmpeg -version
colmap -h
cmake --version
node --version
npm --version
```

## 3. Splat Studio runtime

The normal installer creates a private Miniforge/Python environment under:

```text
runtime/
```

rather than modifying your system Python.

If you are troubleshooting the runtime, it is usually easier to delete the
local `runtime/` folder and rerun:

```bash
./"Install Splat Studio.command"
```

The installer will rebuild it.

## 4. Native backend

The installer manages the native backend under:

```text
backend/gsplat-metal/
```

and pins it to a known-compatible upstream revision.

If that folder becomes damaged, remove only the generated backend folder and
rerun the installer:

```bash
rm -rf backend/gsplat-metal
./"Install Splat Studio.command"
```

Do not remove your `projects/` folder.

## 5. SuperSplat tools

The local editor/viewer are installed under:

```text
.splat_studio/third_party/
```

They can also be installed or repaired later from **Settings → Local tools**
inside Splat Studio.

---

## Optional AI setup

The main installer asks whether to install optional AI support.

It can also be installed later with:

```text
Setup AI Vision.command
```

AI models are cached locally under:

```text
models/huggingface/
```

AI support is **not required** to reconstruct a Gaussian Splat.

The currently configured optional NVIDIA SegFormer model has separate
non-commercial research/evaluation licensing terms. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Video and photo sources

Splat Studio supports:

- MP4
- MOV
- M4V
- JPG / JPEG
- PNG
- WebP
- TIFF

Video sources are sampled automatically.

Photo sets are normalized for EXIF orientation and can use exhaustive COLMAP
matching for smaller unordered image sets.

---

## Native Metal training

The Gaussian training engine uses the native `a1091150/gsplat-mlx` backend.

Training initializes from reconstructed COLMAP sparse geometry and performs
adaptive Gaussian optimization using MLX and native Metal acceleration.

---

## Snapshots and recovery

Splat Studio can periodically save recovery snapshots containing training state.

Features include:

- configurable snapshot intervals
- resume from the latest compatible snapshot
- total runtime tracking
- safe Stop behavior
- automatic retry after worker errors
- lower-pressure retry modes where appropriate

---

## Live previews

Training can periodically render a small preview image.

Preview size and frequency are intentionally limited so previews do not
substantially interfere with training performance.

---

## SuperSplat

Splat Studio can:

- preview results
- open the local SuperSplat editor
- launch a fullscreen viewer
- calculate a recommended source-camera viewpoint
- correct generated scene orientation
- prepare viewer-compatible cached assets

---

## Project management

Projects can be:

- opened
- reviewed
- revealed in Finder
- restarted
- deleted/moved to Trash

---

## Local file layout

The public repository contains source and installation files only.

These directories are generated locally and should **not** be committed:

```text
splatstudio/
├── backend/
├── runtime/
├── models/
├── projects/
└── .splat_studio/
```

They contain compiled dependencies, model caches, projects, settings and logs.

---

## Privacy

Splat Studio is designed as a **local reconstruction workstation**.

Source videos, photographs, reconstruction data and generated splats remain on
the user's computer during the normal workflow.

Optional AI models are downloaded from their upstream providers, but inference
runs locally after installation.

---

## Capture tips

### Video

- move smoothly
- keep the scene in view
- maintain strong overlap between neighboring views
- avoid heavy motion blur
- avoid sudden viewpoint jumps

### Photos

- capture many overlapping viewpoints
- move gradually around the scene
- keep common textured features visible
- avoid large gaps between adjacent positions
- avoid excessive duplicate images

---

## Troubleshooting

### Installation finished with missing items

Read:

```text
.splat_studio/install_report.txt
```

Fix the item listed there, then run:

```text
Install Splat Studio.command
```

again.

### Too few COLMAP cameras register

Try:

- a stronger reconstruction profile
- more overlapping views
- steadier footage
- sharper source images
- more scene texture
- automatic camera rescue

Splat Studio intentionally refuses to train when the camera reconstruction is
too weak to produce a useful result.

### Training error

If a compatible snapshot exists, Splat Studio can resume from the latest saved
state. Automatic recovery can also retry with a lower-pressure configuration.

### Viewer is blank

The installer now attempts to build the local SuperSplat Editor, Viewer and
SplatTransform automatically.

They can also be repaired later from **Settings → Local tools**.

---

## Updating

If you cloned with Git:

```bash
git pull
```

Then rerun:

```text
Install Splat Studio.command
```

if a release changes dependencies.

Projects, models and local settings live outside tracked source files and should
not be affected by a normal source update.

---

## License

Splat Studio's original source code is released under **The Unlicense**.

See [`LICENSE`](LICENSE).

Third-party software and model weights retain their own licenses. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

---

## Credits

Splat Studio builds on excellent projects including:

- COLMAP
- MLX
- `a1091150/gsplat-mlx`
- PlayCanvas SuperSplat
- FFmpeg
- Streamlit
- Depth Anything V2
- Hugging Face Transformers

Please support and cite upstream projects where appropriate.

---

## Contributing

Issues and pull requests are welcome.

See [`CONTRIBUTING.md`](CONTRIBUTING.md).
