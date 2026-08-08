# Finetuning the GOAT detector (run on a machine with storage + GPU)

The main box lacks disk for the HM3D-Semantics annotations, so we train the
detector elsewhere and copy back a single ~6 MB weight file.

**Deliverable to send back:** `checkpoints/yolo_goat.pt`

The pipeline: render HM3D **train** scenes (RGB + semantic masks) → YOLO labels
for the 36 GOAT categories → finetune YOLOv8n → export `yolo_goat.pt`.

---

## 0. Running this on Windows (WSL2) — added 2026-08-10

habitat-sim has **no Windows build** (no `win-64` package on any channel), so the
render step cannot run natively. It does run in WSL2, but WSL has no NVIDIA
EGL/GLX driver (`/usr/lib/wsl/lib` ships no `libEGL_nvidia.so`), so habitat's
default GPU path aborts with:

```
Platform::WindowlessEglApplication::tryCreateContext(): unable to find CUDA
device 0 among 1 EGL devices in total
```

Hardware GL in WSL comes from Mesa's **d3d12** gallium driver talking to
`/dev/dxg`. Working setup:

```bash
sudo apt-get install -y git-lfs build-essential mesa-utils
git lfs install                      # else GLBs arrive as 133-byte LFS pointers

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export GALLIUM_DRIVER=d3d12          # else Mesa falls back to llvmpipe (CPU)
export EGL_PLATFORM=surfaceless      # GBM/device platforms fail: no /dev/dri
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA   # else it picks the Intel iGPU
export HABITAT_GPU_DEVICE_ID=-1      # skip Magnum's EGL<->CUDA device match
```

`HABITAT_GPU_DEVICE_ID` is read by `HabitatEnv` and `make_yolo_dataset.py`;
it defaults to 0, so native-Linux machines are unaffected. Verified: RGB +
semantic at 512x512, **~128 fps** on an RTX 3050 Laptop (4 GB).

Gotchas that cost real time here:

- **git-lfs**: without it every `.glb` is a text pointer and habitat fails with
  `Utility::Json: unexpected v` (the file starts `version https://git-lfs...`).
- **numpy**: torch/ultralytics pull numpy 2.x, which breaks habitat-sim's
  quaternion extension (`_ARRAY_API not found`). Pin `numpy==1.26.4`, and use
  `opencv-python-headless==4.10.0.84` + `pillow==10.4.0`, which are the last
  numpy-1.x-compatible releases.
- **requirements.txt is not installable as written** in this env: `numpy==1.24.4`
  conflicts with habitat-sim, and `ultralytics==8.1.34` calls `torch.load()`
  without `weights_only=False`, which torch>=2.6 rejects. Use ultralytics 8.4.x.
- **scikit-fmm is missing from requirements.txt** but `src/planning/fmm.py`
  needs it; without it the planner silently degrades to A*.
- **WSL DNS** (`nameserver 10.255.255.254`) can stop resolving mid-session,
  which looks exactly like an auth failure (`curl rc=6`). Put `1.1.1.1` in
  `/etc/resolv.conf` and set `generateResolvConf = false` in `/etc/wsl.conf`.

**VRAM measured on 4 GB (RTX 3050), imgsz 512:** batch 24 peaks at 2247 MiB, so
batch 24-32 is safe; the `--batch 16` default below is conservative.

---

## 1. Environment

```bash
conda create -n goat python=3.9 -y && conda activate goat
conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
pip install ultralytics opencv-python numpy
git clone https://github.com/N1CKX-MU/goat-lite.git && cd goat-lite   # scripts + src/perception/goat_classes.py must be present
```

## 2. Download HM3D train + semantics (needs Matterport/HM3D access token)

HM3D is license-gated — request access and get your API token from the
Matterport/HM3D instructions, then use habitat-sim's downloader. Exact uid
strings vary by version; list them first:

```bash
python -m habitat_sim.utils.datasets_download --list        # find the right uids
python -m habitat_sim.utils.datasets_download \
    --username <TOKEN_USER> --password <TOKEN_PASS> \
    --uids hm3d_train_v0.2 hm3d_semantic_annots_v0.2 \
    --data-path data
```

Verify each scene dir under the train root has **both** `*.basis.glb` and
`*.semantic.glb` (plus `*.semantic.txt`). If `*.semantic.glb` is missing the
render step produces no labels.

## 3. Render the YOLO dataset

```bash
python scripts/make_yolo_dataset.py \
    --hm3d-root data/scene_datasets/hm3d/train \
    --out data/yolo_goat \
    --frames-per-scene 60 --resolution 512
```

Produces `data/yolo_goat/{images,labels}/{train,val}` and `data/yolo_goat/data.yaml`.
Sanity-check a few label files are non-empty and class ids are in `0..35`.

## 4. Finetune

```bash
python scripts/finetune_yolo.py \
    --data data/yolo_goat/data.yaml \
    --epochs 60 --imgsz 512 --batch 16 --device 0
```

Exports the best checkpoint to `checkpoints/yolo_goat.pt`.

## 5. Send back

Copy **`checkpoints/yolo_goat.pt`** to the main machine's `checkpoints/` dir.
The detector auto-loads it (`default_weights()` in `src/perception/detector.py`);
no code change needed. Its class names are the GOAT categories, so detections
feed category matching and `goal_in_view` directly.

---

## GOAT categories (class id order, from `src/perception/goat_classes.py`)

```
0 boiler          9 flowerpot        18 mirror          27 printer
1 book           10 footrest         19 nightstand      28 radiator
2 calendar       11 freezer          20 parapet         29 refrigerator
3 carpet         12 glass            21 photo           30 rug
4 christmas tree 13 handrail         22 photo mount     31 shower glass
5 decorative plant 14 hanger         23 piano           32 stair
6 dresser        15 hanging clothes  24 picture         33 statue
7 exercise bike  16 island          25 pillow          34 vase
8 flower vase    17 microwave        26 plant           35 window glass
```

These are the goal categories in HM3D **val_unseen** (what we're evaluated on).
Training images come from **train** scenes so val stays unseen. If a category is
rare/absent in train, the detector may underperform on it — note which for the
report's error analysis.

---

## Prompt for your AI assistant (paste this into Claude Code / Cursor / etc.)

Copy everything in the box below into your AI coding assistant, running it from
an empty working directory on your machine. It will do the whole job.

```
You are helping me finetune a YOLOv8 object detector for a robotics project
(GOAT-Bench object navigation). I have a laptop with a GPU and plenty of disk.
Do the entire pipeline end to end and stop only when you need my input.

GOAL: produce the file `checkpoints/yolo_goat.pt` inside the cloned repo, then
tell me to send that single file back to my friend.

Steps to execute:

1. Set up the environment:
   - conda create -n goat python=3.9 -y && conda activate goat
   - conda install habitat-sim withbullet -c conda-forge -c aihabitat -y
   - pip install ultralytics opencv-python numpy
   - git clone https://github.com/N1CKX-MU/goat-lite.git && cd goat-lite
   Confirm scripts/make_yolo_dataset.py, scripts/finetune_yolo.py, and
   src/perception/goat_classes.py exist.

2. Download HM3D TRAIN scenes + HM3D-Semantics annotations. HM3D is
   license-gated. FIRST run:
     python -m habitat_sim.utils.datasets_download --list
   and find the current uids for the HM3D train scenes and the HM3D semantic
   annotations. Then ASK ME for my HM3D/Matterport API username+password (do
   NOT proceed without them; do not invent credentials). Then run the download,
   e.g.:
     python -m habitat_sim.utils.datasets_download \
       --username <ASK_ME> --password <ASK_ME> \
       --uids <train_uid> <semantic_annots_uid> --data-path data
   VERIFY: each scene directory under the train root contains BOTH a
   `*.basis.glb` and a `*.semantic.glb` (and `*.semantic.txt`). If the semantic
   files are missing, stop and tell me — training is impossible without them.

3. Render the YOLO dataset (point --hm3d-root at the actual train scenes dir you
   just downloaded; find it under data/):
     python scripts/make_yolo_dataset.py \
       --hm3d-root <TRAIN_SCENES_DIR> --out data/yolo_goat \
       --frames-per-scene 60 --resolution 512
   Then sanity-check: some files in data/yolo_goat/labels/train/ are non-empty
   and all class ids are in 0..35. If almost all label files are empty, stop and
   report (the semantic->category mapping may need fixing).

4. Finetune. Pick --batch and --imgsz to fit my GPU's VRAM (ask me the GPU model
   if unsure; use --batch 16 --imgsz 512 as a default, lower batch if you hit
   CUDA out-of-memory):
     python scripts/finetune_yolo.py --data data/yolo_goat/data.yaml \
       --epochs 60 --imgsz 512 --batch 16 --device 0
   This exports checkpoints/yolo_goat.pt.

5. Verify the weights load and detect GOAT classes:
     python -c "from ultralytics import YOLO; m=YOLO('checkpoints/yolo_goat.pt'); print(len(m.names),'classes:',list(m.names.values()))"
   Expect 36 classes with names like freezer, dresser, boiler, mirror, etc.

6. Report the final mAP from training, the size of checkpoints/yolo_goat.pt, and
   tell me to send ONLY that file back to my friend.

Work autonomously; only pause to ask me for (a) the HM3D credentials and (b) my
GPU model if you need it for batch sizing. If any step fails, diagnose and fix
before continuing.
```

