# GOAT-Lite Technical Guide

> **What this document is.** A detailed, beginner-friendly reference for every concept, technique, and design decision in the GOAT-Lite codebase. If you or your teammate encounter a term you don't understand, look it up here. This document is updated as we build new modules.
>
> **Last updated:** Week 9 (eval runner, metrics, dataset loader complete).

---

## Table of Contents

1. [Big Picture: What the System Does](#1-big-picture-what-the-system-does)
2. [The Sense-Plan-Act Loop](#2-the-sense-plan-act-loop)
3. [Module-by-Module Breakdown](#3-module-by-module-breakdown)
   - 3.1 [Simulation Environment (`src/sim/env.py`)](#31-simulation-environment)
   - 3.2 [Object Detector (`src/perception/detector.py`)](#32-object-detector)
   - 3.3 [CLIP Encoder (`src/perception/encoder.py`)](#33-clip-encoder)
   - 3.4 [Coordinate Transforms (`src/mapping/transforms.py`)](#34-coordinate-transforms)
   - 3.5 [Perception Pipeline (`src/perception/pipeline.py`)](#35-perception-pipeline)
   - 3.6 [Occupancy Map (`src/mapping/occupancy.py`)](#36-occupancy-map)
   - 3.7 [Frontier Detection (`src/mapping/frontier.py`)](#37-frontier-detection)
   - 3.8 [Instance Memory Database (`src/memory/instance_db.py`)](#38-instance-memory-database)
   - 3.9 [Goal Matcher (`src/matching/goal_matcher.py`)](#39-goal-matcher)
   - 3.10 [A* Planner (`src/planning/astar.py`)](#310-a-planner)
   - 3.11 [Fast Marching Planner (`src/planning/fmm.py`)](#311-fast-marching-planner)
   - 3.12 [Frontier Exploration Policy (`src/planning/exploration.py`)](#312-frontier-exploration-policy)
   - 3.13 [Action Selection (`src/agent/action.py`)](#313-action-selection)
   - 3.14 [Seed Utility (`src/utils/seeds.py`)](#314-seed-utility)
   - 3.15 [Smoke Test (`scripts/smoke_test.py`)](#315-smoke-test)
   - 3.16 [GoatAgent State Machine (`src/agent/goat_agent.py`)](#316-goatagent-state-machine)
   - 3.17 [GOAT-Bench Dataset Loader (`src/sim/goat_dataset.py`)](#317-goat-bench-dataset-loader)
   - 3.18 [Evaluation Metrics (`src/eval/metrics.py`)](#318-evaluation-metrics)
   - 3.19 [Evaluation Runner (`src/eval/runner.py`)](#319-evaluation-runner)
4. [Testing Strategy](#4-testing-strategy)
5. [Glossary of Key Terms](#5-glossary-of-key-terms)
6. [Engineering Log](#6-engineering-log) — dated record of problems hit and how they were fixed

---

## 1. Big Picture: What the System Does

GOAT-Lite is an autonomous navigation agent that lives inside a photorealistic 3D house simulator (Habitat). You give it a goal like `"chair"` or `"the wooden chair near the window"`, and it has to navigate through the house to find and reach that specific object.

The agent doesn't see the whole house at once — it only has a first-person camera view (like looking through a robot's eyes). So it must:
1. **See** — detect objects in its camera view and figure out where they are in 3D space.
2. **Remember** — store what it has seen in a memory database so it doesn't forget past observations.
3. **Plan** — decide where to go next (either toward a known object or to explore new areas).
4. **Act** — execute movement actions (walk forward, turn left, turn right, or stop).

The "lifelong" part means the agent keeps its memory across multiple goals in the same house. If it already saw a chair while looking for a table, it can go straight to that chair when later asked to find one.

---

## 2. The Sense-Plan-Act Loop

Every single timestep (think of it as one "frame" of the agent's life), this happens:

```
Camera image (RGB + Depth)
        │
        ▼
  ┌─────────────┐
  │  PERCEPTION  │  "What objects do I see, and where are they in the world?"
  │  (pipeline)  │
  └──────┬──────┘
         │ list of PerceivedInstance
         ▼
  ┌─────────────┐
  │   MEMORY    │  "Let me update my database of known objects."
  │ (instance   │
  │  database)  │
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐
  │  MATCHING   │  "Does any object in memory match my current goal?"
  └──────┬──────┘
         │ yes/no + which object
         ▼
  ┌─────────────┐
  │  PLANNER    │  "If yes: navigate to it. If no: explore to find new areas."
  └──────┬──────┘
         │ path to follow
         ▼
  ┌─────────────┐
  │   ACTION    │  "Turn left / turn right / walk forward / stop"
  └─────────────┘
```

---

## 3. Module-by-Module Breakdown

### 3.1 Simulation Environment

**File:** `src/sim/env.py`
**Built in:** Week 2

#### What it does

This is the agent's interface to the simulated 3D world. It wraps Facebook's `habitat-sim` library and provides a clean, simple API. Think of it as the "game engine" — it renders what the agent sees, lets the agent move around, and tells the agent where it is.

#### Key data structures

**`GoalSpec`** — Describes what the agent is looking for.
```python
@dataclass
class GoalSpec:
    modality: str       # "category" (e.g. "chair") or "language" (e.g. "red chair near window")
    value: str          # the actual goal text
    episode_step: int   # which step of the overall episode we're on
    subtask_index: int  # which goal number within this episode (0, 1, 2, ...)
```

**`Observation`** — Everything the agent perceives at one moment in time.
```python
@dataclass
class Observation:
    rgb: np.ndarray      # [H, W, 3] — the color image (height x width x RGB channels), uint8 (0-255)
    depth: np.ndarray    # [H, W] — how far away each pixel is, in meters (float32)
    pose: np.ndarray     # [4, 4] — the agent's position and orientation in the world (SE(3) matrix)
    compass: float       # heading angle in radians
    gps: np.ndarray      # [2] — (x, z) position on the ground plane
    current_goal: GoalSpec | None
```

**`ACTION_MAP`** — The four things the agent can do:
- `0` = STOP (declare "I've found it" or give up)
- `1` = move forward 25 cm
- `2` = turn left 30 degrees
- `3` = turn right 30 degrees

#### The HabitatEnv class

**`__init__`** — Sets up the simulator. Key parameters:
- `scene_path`: path to the 3D house file (`.glb` format from HM3D dataset)
- `resolution`: image size, default `(256, 256)` — small for speed, can increase for accuracy
- `agent_height`: 1.41 m — mimics the Hello Robot Stretch, a real-world mobile robot
- `agent_radius`: 0.17 m — how wide the robot is, for collision detection
- `step_size`: 0.25 m — how far "move forward" goes
- `turn_angle`: 30 degrees — how much "turn left/right" rotates

It creates two virtual cameras mounted at the agent's head height:
- **RGB camera** — color image, like a normal photo
- **Depth camera** — measures distance to every pixel (how a LiDAR or depth sensor works)

**`intrinsics`** property — Returns the 3x3 **camera intrinsic matrix** (see [Glossary: Camera Intrinsics](#camera-intrinsics-matrix-k)). This is computed from the camera's field-of-view and resolution, not hardcoded.

**`reset()`** — Places the agent at a random walkable position in the scene. Returns the first Observation.

**`step(action)`** — Executes one action. Returns `(observation, done, info)`. If the action is STOP (0), `done` becomes `True`.

**`get_pose()`** — Returns the agent's 4x4 pose matrix in world coordinates. Uses the `quaternion` library to convert the agent's rotation (stored as a quaternion) into a rotation matrix.

**`get_compass()`** — Extracts the **yaw** angle from the agent's quaternion rotation. Yaw = rotation around the vertical (Y) axis = "which direction am I facing on the ground plane".

**`get_gps()`** — Returns just the (X, Z) ground-plane position, ignoring height.

**`_make_obs()`** — Internal helper that gathers all sensor data into an Observation. Drops the alpha channel from RGB (Habitat returns RGBA, we only need RGB).

#### Why these design decisions

- **Stretch robot dimensions:** We use real Stretch robot specs so results are somewhat transferable to real hardware.
- **256x256 resolution:** Keeps VRAM usage low (~30 MB for detector at this size). Can go up to 640x480 for final evaluation.
- **Physics disabled:** We only need rendering, not physics simulation. Saves compute.
- **Sensor position at agent height:** Camera is at eye level of the robot, not at the floor or ceiling.

---

### 3.2 Object Detector

**File:** `src/perception/detector.py`
**Built in:** Week 2

#### What it does

Takes an RGB image and finds objects in it. Returns a list of bounding boxes (rectangles around detected objects) with class labels and confidence scores.

#### How it works: YOLOv8

**YOLO** stands for "You Only Look Once." It's a family of neural networks designed for real-time object detection. YOLOv8 is the 8th major version.

**YOLOv8n** — the "n" stands for "nano," the smallest and fastest variant. It trades some accuracy for speed and low memory usage. On our 4 GB GPU:
- Model size: ~6 MB (fp16)
- Inference: ~50 FPS at 256x256
- VRAM: ~30 MB

The detector uses **COCO weights** by default — trained on the 80-category COCO dataset (common objects like chairs, tables, people, etc.). In Week 3, we planned to finetune on GOAT-Bench's 36 specific categories, but that's deferred for now.

#### Key parameters

```python
YOLODetector(
    weights="yolov8n.pt",  # path to model weights file
    device="cuda",          # run on GPU ("cuda") or CPU ("cpu")
    fp16=True,              # use half-precision floating point (see Glossary)
    conf=0.35,              # minimum confidence threshold — only report detections above this
    iou=0.5,                # IoU threshold for NMS (see Glossary: NMS)
    imgsz=256,              # input image size — MUST match training/inference
)
```

#### Detection dataclass

```python
@dataclass
class Detection:
    cls_id: int                       # numeric class ID (e.g. 56 = "chair" in COCO)
    cls_name: str                     # human-readable name (e.g. "chair")
    conf: float                       # confidence score 0-1 (how sure the model is)
    bbox: tuple[int, int, int, int]   # (x1, y1, x2, y2) — top-left and bottom-right corners
    mask: np.ndarray | None           # pixel-level mask (unused for now, always None)
```

**Bounding box format `(x1, y1, x2, y2)`:** These are pixel coordinates. `(x1, y1)` is the top-left corner, `(x2, y2)` is the bottom-right corner. So a box `(50, 100, 200, 300)` means "a rectangle from pixel (50,100) to pixel (200,300)".

#### Implementation details

- **`model.fuse()`** — Merges batch normalization layers into convolutional layers. This is an optimization that makes inference faster without changing the output. Batch norm + conv can be mathematically combined into a single operation.
- **`model.predict(..., verbose=False)`** — Suppresses YOLO's default logging output.
- **`boxes.xyxy`** — YOLO stores boxes in multiple formats. `.xyxy` gives us the `(x1, y1, x2, y2)` format we want. The `.cpu().numpy()` chain moves data from GPU to CPU and converts from PyTorch tensor to NumPy array.

---

### 3.3 CLIP Encoder

**File:** `src/perception/encoder.py`
**Built in:** Week 2

#### What it does

Turns images and text into comparable number vectors (**embeddings**). This is the magic that lets us match text goals like "red chair" to actual image crops of objects — without any task-specific training.

#### How CLIP works (the key insight)

**CLIP** (Contrastive Language-Image Pre-training) is a model from OpenAI trained on 400 million image-text pairs from the internet. It learned to map images and text into the **same vector space**: if an image and a text description are semantically related, their vectors will point in similar directions.

```
"a red chair"  →  CLIP text encoder  →  [0.12, -0.34, 0.56, ...]  (512 numbers)
[image of red chair]  →  CLIP image encoder  →  [0.11, -0.33, 0.55, ...]  (512 numbers)
                                                  ↑ these are very similar!

[image of blue table]  →  CLIP image encoder  →  [-0.45, 0.22, -0.18, ...]  (512 numbers)
                                                  ↑ these are very different from "red chair"
```

**Cosine similarity** measures how similar two vectors are. It's the cosine of the angle between them:
- `1.0` = identical direction (perfect match)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite direction

We use this to match goals to stored object embeddings in memory.

#### Our specific model: open_clip ViT-B/32

We use the **open-source reimplementation** of CLIP via the `open_clip` library, not OpenAI's original. Specifically:

- **ViT-B/32** = "Vision Transformer, Base size, 32x32 patch size"
  - **ViT** = Vision Transformer — applies the Transformer architecture (originally for text/NLP) to images by splitting them into patches
  - **B** = Base — medium model size (vs. Small or Large)
  - **32** = each image is split into 32x32 pixel patches before processing
- **laion2b_s34b_b79k** = pretrained on the LAION-2B dataset (2 billion image-text pairs)
- **Output dimension: 512** — each image or text gets a 512-dimensional vector

VRAM usage: ~175 MB at fp16. MobileCLIP-S0 (~55 MB) was considered as a lighter alternative but we went with ViT-B/32 for better accuracy.

#### Implementation details

**`encode_image(crops)`:**
1. Convert each NumPy array crop to a PIL Image
2. Apply CLIP's **preprocessing** (resize to 224x224, normalize pixel values to specific mean/std that the model expects)
3. Stack into a batch tensor, move to GPU
4. Cast to fp16 if enabled
5. Run through the vision transformer with `torch.no_grad()` (we're not training, so don't track gradients — saves memory)
6. Cast output back to fp32 (some operations are unstable in fp16)
7. **L2-normalize** the output vectors
8. Move to CPU and convert to NumPy

**`encode_text(prompts)`:**
1. **Tokenize** the text — convert words to numeric IDs that the model understands
2. Move token IDs to GPU
3. Run through the text transformer
4. L2-normalize
5. Return as NumPy

**Why L2-normalize?** L2 normalization makes every vector have length 1 (unit vector). After this, the **dot product** of two vectors equals their **cosine similarity**. This means comparing embeddings is just a simple matrix multiply — very fast.

```python
# Without normalization, you'd need:
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# With L2-normalized vectors:
cos_sim = np.dot(a, b)  # that's it!
```

---

### 3.4 Coordinate Transforms

**File:** `src/mapping/transforms.py`
**Built in:** Week 3

#### What it does

Converts between three different coordinate systems (frames of reference). This is the geometric backbone of the entire system — if these are wrong, the map and memory positions will be garbage.

#### The three coordinate frames

**1. Camera frame** — centered at the camera lens
- X = right
- Y = down (in our convention after back-projection) or Y-up in OpenGL
- Z = forward (depth direction)
- Units: meters
- Example: "the chair is 0.5m to the right and 3m in front of me"

**2. World frame** — Habitat's global coordinate system
- X = one horizontal axis
- **Y = up** (this is important — Y is vertical, not Z like in some other systems)
- Z = the other horizontal axis
- Units: meters
- Example: "the chair is at position (4.2, 0.8, -1.5) in the house"

**3. Map grid frame** — our 2D top-down occupancy grid
- Row = corresponds to world Z
- Col = corresponds to world X
- Y (height) is ignored — we flatten to 2D
- Units: grid cells (each cell = `resolution` meters, default 5 cm)
- Example: "the chair is at grid cell (row=84, col=160)"

#### Function-by-function

**`make_intrinsics(hfov_deg, width, height)`**

Builds the **camera intrinsic matrix K** (see [Glossary](#camera-intrinsics-matrix-k)). This 3x3 matrix encodes the camera's internal geometry:

```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```

- `fx, fy` = **focal length** in pixels. Computed from the horizontal field of view: `fx = width / (2 * tan(hfov/2))`. For 90° hfov at 256px: `fx = 256 / (2 * tan(45°)) = 128`.
- `cx, cy` = **principal point** — the pixel where the optical axis hits the image. Usually the center: `(width/2, height/2)`.
- We assume square pixels, so `fx = fy`.

**`depth_to_pointcloud_camera(depth, K)`**

The core geometric operation: **back-projection**. Converts a 2D depth image into a 3D point cloud in camera coordinates.

For each pixel `(u, v)` with depth `d`:
```
x_cam = (u - cx) * d / fx
y_cam = (v - cy) * d / fy
z_cam = d
```

This reverses what the camera does: the camera projects 3D→2D (losing depth information), and we undo it using the known depth.

- Returns `[H*W, 3]` — one 3D point per pixel
- Invalid depths (< 0.01 m or > 10.0 m) are zeroed out
- Uses `np.meshgrid` to create a grid of all (u, v) pixel coordinates at once (vectorized, no loops = fast)

**`pointcloud_camera_to_world(pts_camera, pose)`**

Applies a rigid-body transformation to move points from camera frame to world frame.

The **pose** is a 4x4 matrix encoding both rotation and translation:
```
pose = [ R  t ]    R = 3x3 rotation matrix
       [ 0  1 ]    t = 3x1 translation vector (camera position in world)
```

The transform: `point_world = R @ point_camera + t`

This is the standard SE(3) (Special Euclidean group in 3D) transformation — the math of rigid bodies in 3D space.

**`bbox_center_depth_to_world_xyz(bbox, depth_map, K, pose)`**

The function the perception pipeline calls most. Given a bounding box around a detected object:

1. Find the center pixel of the bbox: `cx = (x1+x2)/2, cy = (y1+y2)/2`
2. Sample a small **patch** of depth values around that center (5x5 by default)
3. Take the **median** of valid depths in the patch — median is robust to outliers (if one pixel has a weird depth value, median ignores it, unlike mean)
4. Back-project that single center pixel with the median depth to get `(x_cam, y_cam, z_cam)`
5. Transform to world frame using the pose matrix

Returns `None` if all depths in the patch are invalid (< 0.01 m or > 10.0 m). This happens when the object is too close, too far, or the depth sensor failed at that pixel.

**Why median over mean?** Depth sensors often have noisy or missing values at object edges. The median of a 5x5 patch gives a stable depth estimate even if some pixels are on the background behind the object.

**`world_to_grid(xy_world, origin, resolution)`**

Converts a world (X, Z) position to a grid (row, col):
```
col = round((world_x - origin_x) / resolution)
row = round((world_z - origin_z) / resolution)
```

`origin` is the bottom-left corner of the map in world coordinates. `resolution` is meters per cell (0.05 m = 5 cm).

Note: we drop world Y (height) because our map is 2D top-down.

**`grid_to_world(ij, origin, resolution)`**

The inverse of the above:
```
world_x = origin_x + col * resolution
world_z = origin_z + row * resolution
```

---

### 3.5 Perception Pipeline

**File:** `src/perception/pipeline.py`
**Built in:** Week 3

#### What it does

This is the "glue" module that orchestrates one frame of perception. It takes a raw Observation and produces a list of `PerceivedInstance` objects — each one representing "I see a chair at position (x, y, z) in the world, and here's its visual embedding."

#### PerceivedInstance dataclass

```python
@dataclass
class PerceivedInstance:
    cls_id: int              # numeric class ID from YOLO (e.g. 56)
    cls_name: str            # human-readable name (e.g. "chair")
    conf: float              # YOLO confidence (0-1)
    bbox: tuple[int,int,int,int]  # (x1, y1, x2, y2) in pixel coords
    crop_thumbnail: np.ndarray    # 64x64x3 uint8 — small image of the object for visualization
    clip_embed: np.ndarray        # [512] float32 — L2-normalized CLIP embedding of the crop
    world_xyz: np.ndarray         # [3] float64 — estimated (x, y, z) position in world frame
    seen_step: int                # which simulation step this was observed at
```

This is the "currency" of our system. Every downstream module (memory, matching, visualization) consumes PerceivedInstances.

#### The process() method step by step

```python
def process(self, obs: Observation, step: int) -> list[PerceivedInstance]:
```

**Step 1: Run YOLO detector**
```python
detections = self._detector.detect(obs.rgb)
```
Gets a list of Detection objects (bboxes + classes + confidences).

**Step 2: Filter and validate each detection**

For each detection:

a. **Clip bbox to image bounds** — YOLO sometimes predicts boxes that extend beyond the image edge. We clamp `x1, y1` to `>= 0` and `x2, y2` to `<= image size`. Without this, we'd get invalid array indices when cropping.

b. **Filter by minimum crop size** — If the clipped crop is smaller than `min_crop_size` (default 32 pixels) in either dimension, skip it. Tiny crops produce meaningless CLIP embeddings because CLIP was trained on 224x224 images — a 10x10 crop upscaled to 224x224 is just a blurry blob.

c. **Back-project to world coordinates** — Call `bbox_center_depth_to_world_xyz()` to get the 3D world position. If depth is invalid at the bbox center, skip this detection entirely — we can't store an object without knowing where it is.

d. **Extract image crop** — `obs.rgb[y1:y2, x1:x2]` slices the bounding box region from the image. Note the order: NumPy arrays are indexed as `[row, col]`, which is `[y, x]`.

**Step 3: Batch CLIP encoding**
```python
embeds = self._encoder.encode_image(crops)
```
All valid crops are encoded together in one batch. This is much faster than encoding one at a time because the GPU can process them in parallel. The output is `[N, 512]` — N embeddings of 512 dimensions each, all L2-normalized.

**Step 4: Create thumbnails and assemble results**

For each valid detection:
- Resize the crop to 64x64 using `cv2.resize` with `INTER_AREA` interpolation (best for downscaling — averages pixel groups rather than just sampling one pixel, avoiding aliasing artifacts)
- Pack everything into a `PerceivedInstance`

#### Why batch encoding matters

If there are 5 detections per frame:
- One-at-a-time: 5 GPU kernel launches, 5 memory transfers = ~25ms
- Batched: 1 GPU kernel launch, 1 memory transfer = ~8ms

At 200ms per timestep budget, this ~17ms saving matters.

---

### 3.6 Occupancy Map

**File:** `src/mapping/occupancy.py`
**Built in:** Week 4

#### What it does

Builds a 2D top-down map of the environment from depth images. This is the agent's "mental picture" of the house floor plan — it knows which areas are open space, which have walls/furniture, and which it hasn't seen yet. The map also tracks which object categories have been detected where.

#### The three grids

The `SemanticMap` maintains three parallel grids, all the same size:

**1. Occupancy grid** — `[H, W]` of int8 values:
- `-1` = **unknown** (haven't observed this area yet)
- `0` = **free** (confirmed open space the agent could walk through)
- `1` = **occupied** (wall, furniture, or other obstacle)

This is the grid the planner uses for pathfinding — it plans paths through free cells and avoids occupied ones.

**2. Explored mask** — `[H, W]` of bool:
- `True` = this cell has been observed at least once
- `False` = never seen

Used by frontier detection to find the boundary of what's been explored.

**3. Class counts** — `[H, W, num_classes]` of int16:
- For each cell, counts how many times each object class has been detected there
- Example: `class_counts[100, 200, 5] = 12` means "class 5 (chair) has been detected 12 times at grid cell (100, 200)"
- Used later for semantic queries ("where are the chairs?")

#### Grid sizing and origin

```python
SemanticMap(size_m=24.0, resolution_m=0.05)
```

- **`size_m=24.0`** — the map covers 24m x 24m of physical space. Most HM3D indoor scenes fit within this.
- **`resolution_m=0.05`** — each grid cell represents 5cm x 5cm. This gives us `24.0 / 0.05 = 480` cells per side, so a 480x480 grid.
- **Origin** is at `(-size/2, -size/2)` in world XZ coordinates. This centers the map at the world origin `(0, 0)`, so the agent starts roughly in the middle of the grid.

#### update_from_depth — how it works

This is the core function that builds the map. Called once per relevant timestep with the current depth image and the agent's pose.

**Step 1: Back-project depth to world points**

Uses the transform functions from Week 3:
1. `depth_to_pointcloud_camera()` — converts every pixel of the depth image into a 3D point in camera frame
2. `pointcloud_camera_to_world()` — transforms those points to world coordinates using the agent's current pose

**Step 2: Downsample**

The depth image has 256x256 = 65,536 pixels. Processing all of them is wasteful since many map to the same grid cell. We take every 4th point (`pc_world[::4]`), reducing to ~16,384 points. This barely affects map quality but is 4x faster.

**Step 3: Height filter**

Not every 3D point should become an obstacle on our 2D map:
- Points on the **floor** (Y < 0.1m) — the agent walks on the floor, it's not an obstacle
- Points on the **ceiling** (Y > 1.5m) — the agent can walk under the ceiling

We only keep points between 0.1m and 1.5m height above the floor. These are the walls, furniture, and obstacles that actually block the agent.

**Step 4: Vectorized grid projection**

Instead of converting points one-by-one (slow Python loop), we convert all points to grid coordinates with NumPy vector operations:
```python
cols = np.round(offsets[:, 0] / self._resolution).astype(np.int32)
rows = np.round(offsets[:, 1] / self._resolution).astype(np.int32)
```

This processes thousands of points in microseconds instead of milliseconds.

**Step 5: Mark occupied cells**

All valid points are marked as occupied in bulk:
```python
self._occupancy[rows_valid, cols_valid] = 1
```

NumPy advanced indexing lets us set thousands of cells in one operation.

**Step 6: Ray clearing (Bresenham)**

This is the clever part. For each occupied cell, we trace a ray from the camera position to that cell and mark all cells along the ray as **free**. The logic: if we can see a wall at position X, then everything between us and X must be empty space (otherwise we couldn't see through it).

We use **Bresenham's line algorithm** (via `skimage.draw.line`) to find which grid cells the ray passes through. Bresenham is a classic algorithm from computer graphics that determines which pixels/cells a line between two points passes through — it's integer-only and very fast.

**Optimization:** Many points project to the same grid cell. Instead of tracing a ray for every point, we use `np.unique` to find only the unique occupied cells, then trace one ray per unique cell. This reduced the function from ~440ms to ~8ms.

#### update_from_detections — how it works

When the perception pipeline detects objects, we record where each object class was seen on the map.

For each `PerceivedInstance`:
1. Convert its `world_xyz` to a grid cell `(row, col)`
2. Apply a 3x3 **Gaussian-like kernel** centered at that cell:
```
[1  2  1]
[2  4  2]
[1  2  1]
```
3. Add these weights to `class_counts[:, :, cls_id]`

**Why a kernel instead of a single cell?** The world position is noisy (depth estimation isn't perfect). Spreading the count over a 3x3 neighborhood (15cm x 15cm at 5cm resolution) accounts for this uncertainty. The center gets the highest weight (4) because it's the most likely position.

#### Performance

- `update_from_depth`: ~8ms median (target was < 100ms)
- Key optimization: vectorized grid projection + ray clearing only on unique cells
- Memory: 480x480 grid at int8 = ~230 KB for occupancy, ~230 KB for explored, ~17 MB for class counts (480x480x37 at int16)

---

### 3.7 Frontier Detection

**File:** `src/mapping/frontier.py`
**Built in:** Week 4

#### What it does

Finds **frontiers** — the boundaries between explored free space and unexplored unknown space. These are the most promising places to go to discover new areas. When the agent doesn't know where its goal is, it navigates to the nearest/largest frontier to explore.

#### What is a frontier?

Imagine you've explored a room and can see a doorway. Through the doorway is a hallway you haven't explored yet. The cells at the doorway threshold — free cells touching unknown cells — are the frontier. Going there would reveal new space.

Formally: a frontier cell is a **free** cell that has at least one **unknown** neighbor (in the 8-connected sense — up, down, left, right, and diagonals).

#### The algorithm

**Step 1: Find frontier cells**

```python
free_mask = occupancy == 0           # all free cells
unknown_mask = occupancy == -1       # all unknown cells
unknown_neighbor = binary_dilation(unknown_mask, kernel=3x3)  # expand unknown by 1 pixel
frontier_mask = free_mask & unknown_neighbor  # free cells that touch unknown
```

**Binary dilation** is a morphological operation: it takes a boolean mask and "grows" it by one pixel in all directions. If we dilate the unknown mask, any free cell that was adjacent to an unknown cell will now overlap with the dilated unknown mask. The intersection (`&`) gives us exactly the frontier cells.

**Step 2: Group into connected components**

`scipy.ndimage.label` finds connected components — groups of frontier cells that touch each other. Each group becomes one `Frontier` object. This is the same algorithm used in image processing to find distinct blobs.

**Step 3: Filter and sort**

- Frontiers with fewer cells than `min_size` (default 20) are dropped — they're too small to be worth navigating to (probably just a tiny crack between furniture).
- Remaining frontiers are sorted by size descending — the biggest frontier is the most promising place to explore.

#### Frontier dataclass

```python
@dataclass
class Frontier:
    centroid_ij: tuple[int, int]  # center of the frontier (for navigation targeting)
    size: int                     # number of frontier cells (bigger = more to explore)
    cells: np.ndarray             # [N, 2] all the (row, col) coordinates
```

The `centroid_ij` is the mean position of all cells in the frontier — this is where the planner will aim when navigating to this frontier.

---

### 3.8 Instance Memory Database

**File:** `src/memory/instance_db.py`
**Built in:** Week 5

#### What it does

This is the agent's long-term memory. Every time the perception pipeline detects an object, that detection gets stored here — but intelligently. If the agent sees the same chair from two slightly different angles, those observations get **merged** into a single memory node rather than stored as two separate objects. This is what makes the agent "lifelong" — it builds up a coherent model of what's in the scene across hundreds of timesteps.

#### InstanceNode — what the agent remembers about one object

```python
@dataclass
class InstanceNode:
    node_id: int              # unique ID (auto-incremented)
    cls_id: int               # YOLO class ID (e.g. 56 for chair)
    cls_name: str             # human-readable name
    world_xyz: np.ndarray     # [3] — running mean position from all merged observations
    clip_embed: np.ndarray    # [512] — running mean CLIP embedding, re-normalized
    confidence: float         # running max confidence across all observations
    first_seen_step: int      # when was this object first detected
    last_seen_step: int       # when was it most recently seen
    n_observations: int       # how many times has it been observed
    best_thumbnail: np.ndarray  # 64x64 image from the highest-confidence view
```

**Why running mean for position?** Each detection gives a noisy estimate of where the object is (depth sensors are imperfect, the agent might be at an angle). Averaging over many observations converges on the true position. After 10 observations, the position is much more accurate than any single one.

**Why running mean for embedding?** Similar logic — each view of the object produces a slightly different CLIP embedding (different angle, lighting, occlusion). Averaging produces a more robust representation. The re-normalization step after averaging ensures the embedding stays on the unit sphere (length 1), which is required for cosine similarity to work correctly.

**Why running max for confidence?** The best view of an object (head-on, well-lit, unoccluded) produces the highest confidence. We want the thumbnail from that best view, and we use max confidence as a quality signal.

#### The merging algorithm — the core logic

When a new `PerceivedInstance` arrives, the `_integrate` method decides: merge it with an existing node, or create a new one?

```
New detection arrives (class=chair, position=(1.1, 0.5, 2.1))
    │
    ├── Find all existing nodes with same class (chair)
    │   └── Filter to those within merge_dist_m (0.75m) in XZ distance
    │       ├── No candidates → CREATE new node
    │       └── Has candidates → pick the closest one
    │           ├── Check embedding similarity with closest
    │           │   ├── Similar (cosine > 0.85) → MERGE into existing node
    │           │   └── Dissimilar (cosine ≤ 0.85) → CREATE new node
```

**Why check both distance AND embedding similarity?**

Distance alone isn't enough. Imagine two different chairs 50cm apart (e.g., at a dining table). They're spatially close but visually different objects. The embedding similarity check prevents merging distinct objects that happen to be near each other.

Conversely, embedding alone isn't enough. The same chair seen from the front and back might have somewhat different CLIP embeddings. The distance check ensures we only consider merging when detections are in roughly the same spot.

**The thresholds:**
- `merge_dist_m = 0.75` — objects within 75cm are candidates for merging. This is about the size of a chair. Sweepable in Week 10.
- `merge_embed_sim = 0.85` — cosine similarity must exceed 0.85 for merge. This is fairly strict — the embeddings need to be quite similar. Sweepable in Week 10.

#### Queries — how other modules use the database

**`query_by_class(cls_id)`** — Returns all nodes of a given class. Used for category goals ("find a chair" → get all chair nodes).

**`query_by_embedding(embed, top_k)`** — Returns the top-K nodes most similar to a given embedding, sorted by cosine similarity. Used for language goals ("find the red chair near the window" → encode text with CLIP → find most similar stored object).

**`all_nodes()`** — Returns all nodes. Used for visualization and debugging.

#### Memory budget

Each node stores:
- 512-dim float32 embedding = 2 KB
- 64x64x3 uint8 thumbnail = 12 KB
- Scalar fields = ~100 bytes

Total per node: ~14 KB. For 100 nodes (a realistic scene): ~1.4 MB. Well under the 20 MB budget.

---

### 3.9 Goal Matcher

**File:** `src/matching/goal_matcher.py`
**Built in:** Week 6

#### What it does

Given a goal ("find a chair" or "find the red chair near the window") and the instance memory database, the GoalMatcher finds the best matching object node — or returns None if nothing matches well enough.

This is the bridge between "what the agent is looking for" and "what the agent has seen." It's what closes the loop between perception/memory and planning.

#### Two matching modes

**Category matching** (`modality="category"`, e.g. `value="chair"`):
1. Query the database for all nodes with `cls_name == goal.value`
2. If none found, return `(None, 0.0)`
3. If found, return the node with the highest confidence and its confidence as the score

This is straightforward string matching. No threshold needed — if we've seen a chair and the goal is "chair", it matches. The threshold is effectively 0.0.

**Language matching** (`modality="language"`, e.g. `value="the wooden chair near the window"`):
1. Encode the goal text with CLIP's text encoder to get a 512-dim embedding
2. Compute cosine similarity between this text embedding and every stored node's image embedding
3. If the best score exceeds `language_threshold` (default 0.24), return that node
4. If below threshold, return `(None, score)` — we found something but aren't confident enough

This is where CLIP's cross-modal magic happens: comparing a text embedding against image embeddings in the same vector space.

#### Template wrapping

Short goal values like "chair" don't work well as raw CLIP text inputs. CLIP was trained on full sentences like "a photo of a chair", so single words produce weaker embeddings.

When `use_template=True` (default), short values (≤ 3 words) get wrapped:
- `"chair"` → `"a photo of chair"`
- `"red couch"` → `"a photo of red couch"`

Long descriptions are already sentence-like, so they pass through unchanged:
- `"the chair with red cushion near the window"` → used as-is

The 3-word cutoff is a simple heuristic. If the input has 4+ words, it's probably already a natural description.

#### Threshold calibration

The `language_threshold=0.24` is a starting guess. CLIP cosine similarities for text-image pairs tend to range from ~0.15 (weak match) to ~0.35 (strong match). The threshold will be re-tuned in Week 10 by plotting precision-recall curves on dev scenes.

#### Midterm demo script

**File:** `scripts/midterm_demo.py`

The first end-to-end pipeline test (requires Habitat + a scene). It:
1. Loads a scene and initializes all modules (detector, encoder, pipeline, map, memory, matcher)
2. Random-walks the agent for 300 steps, running perception + memory each step
3. Updates the occupancy map every 2 steps
4. Prints a summary of all stored instance nodes
5. Tests 5 hand-crafted goals (3 category, 2 language) against the matcher
6. Saves a top-down plot showing the occupancy map, agent trajectory, all instance nodes, and matched goals highlighted in red

---

### 3.10 A* Planner

**File:** `src/planning/astar.py`
**Built in:** Week 7

#### What it does

Given a start position and goal position on the occupancy grid, finds the shortest obstacle-free path between them. This is how the agent navigates to a known target — the planner computes the route, then the action selector executes it step by step.

#### How A* works

A* is the gold-standard graph search algorithm for pathfinding. It's like Dijkstra's algorithm (finds shortest paths) but with a **heuristic** that guides the search toward the goal, making it much faster.

For each cell it considers, A* computes:
```
f(cell) = g(cell) + h(cell)
```
- **g(cell)** = actual cost to reach this cell from the start (accumulated step by step)
- **h(cell)** = heuristic estimate of remaining cost to reach the goal (we use Euclidean distance — straight-line, which never overestimates, making A* optimal)
- **f(cell)** = total estimated cost through this cell

A* always expands the cell with the lowest f-score next. This means it explores toward the goal first, avoiding wasted exploration in the wrong direction.

#### 8-connected grid

Each cell has 8 neighbors (up, down, left, right, and 4 diagonals). Moving to a cardinal neighbor costs 1.0, moving diagonally costs √2 ≈ 1.414 (the actual Euclidean distance).

#### Obstacle inflation

Real robots aren't point-sized. If the planner finds a path that passes 1 cell (5cm) from a wall, the physical robot (17cm radius) would clip the wall.

**Inflation** expands obstacles by `inflate_cells` (default 4 = 20cm) in all directions using binary dilation. Cells near obstacles get a penalty cost (`obstacle_penalty=3.0`) rather than being fully blocked — this lets the planner find paths through tight spaces if necessary, but prefers routes with clearance.

#### Implementation details

- Uses Python's `heapq` for the priority queue — efficient O(log n) push/pop
- `g_score` stored as a 2D numpy array for O(1) lookup
- `came_from` dictionary for path reconstruction — walk backwards from goal to start
- `closed` set as a boolean array to avoid re-expanding cells
- Counter in heap entries breaks ties consistently (avoids comparing tuples of equal f-score)

#### Performance

On a 480x480 grid (our real map size): < 2 seconds worst case. Typical paths take ~50-200ms depending on length and obstacle density.

---

### 3.11 Fast Marching Planner

**File:** `src/planning/fmm.py`
**Built in:** Week 7

#### What it does

An alternative to A* that produces **smoother paths**. Uses the Fast Marching Method (FMM) — a continuous wavefront propagation algorithm. Think of dropping a pebble in water and watching the ripples expand; FMM computes how long it takes for the "wave" to reach each cell from the goal.

#### How it works

1. Set travel time at goal = 0, everywhere else = ∞
2. Propagate the wavefront outward through free cells (via `skfmm.travel_time`)
3. From the start, follow the gradient of decreasing travel time downhill to the goal — this naturally produces a smooth path

#### Why have both A* and FMM?

- **A*** produces paths with sharp corners (grid-aligned turns). Fast, guaranteed optimal on the grid.
- **FMM** produces smoother, more natural-looking paths. Better for actual robot execution. But requires the `skfmm` library (optional dependency).

The agent can config-select between them. Default is A* since skfmm may not be installed.

---

### 3.12 Frontier Exploration Policy

**File:** `src/planning/exploration.py`
**Built in:** Week 7

#### What it does

When the agent doesn't know where its goal is, it needs to explore. This module decides **which frontier to explore next** — the "where should I go to discover new areas?" decision.

#### The scoring function

```
score = frontier.size / (distance_to_agent + 1)
```

This balances two objectives:
- **Larger frontiers** are more promising — they likely lead to big unexplored areas
- **Closer frontiers** are cheaper to reach — no point crossing the entire map for a slightly bigger frontier

The `+1` prevents division by zero when the frontier is right next to the agent.

#### Visited frontier tracking

Once the agent has navigated to a frontier and explored it, that frontier's centroid is added to the `visited` set. Future calls to `choose_frontier` will skip any frontier whose centroid is within `visit_radius` (default 10 cells = 50cm) of a visited centroid. This prevents the agent from repeatedly going back to the same explored area.

---

### 3.13 Action Selection

**File:** `src/agent/action.py`
**Built in:** Week 7

#### What it does

Converts a high-level path (list of grid cells) into low-level actions (forward, turn left, turn right). This is the "motor control" layer — given where the agent is and where it needs to go, what's the next physical action?

#### The algorithm

```python
path_to_action(agent_ij, heading, next_ij) -> action
```

1. Compute the **desired heading** — the angle from the agent to the next waypoint using `atan2(dc, dr)`
2. Compute the **angular difference** between current heading and desired heading
3. Wrap the difference to [-π, π] to handle the 359°→1° boundary correctly
4. Decision:
   - If angular error < `angle_tolerance` (20°): go **forward** (close enough to the right direction)
   - If error is positive: **turn right** (target is to the right)
   - If error is negative: **turn left** (target is to the left)
   - If already at the waypoint: **stop**

#### Heading convention

```
heading = 0    → facing +row direction (toward higher row indices)
heading = π/2  → facing +col direction (toward higher col indices)
```

This matches the grid's row/col layout. The agent turns 30° per turn action, so it may need multiple turns to face the right direction before moving forward.

#### Angle wrapping

The most common bug in heading-based navigation. Without wrapping, an agent facing at 350° wanting to turn to 10° would compute a difference of -340° and spin almost all the way around. Wrapping to [-π, π] correctly gives +20° — a small right turn.

```python
diff = (diff + math.pi) % (2 * math.pi) - math.pi
```

---

### 3.14 Seed Utility

**File:** `src/utils/seeds.py`
**Built in:** Week 1

#### What it does

Sets random seeds for reproducibility. When you set a seed, every "random" number generated afterward follows a deterministic sequence. Same seed = same sequence = same results every time you run the code.

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)       # Python's built-in random
    np.random.seed(seed)    # NumPy's random (used in many scientific operations)
    torch.manual_seed(seed) # PyTorch CPU random
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU random (all GPUs)
```

The `try/except` around the torch import means this function works even in contexts where PyTorch isn't installed.

**Why 42?** It's the default in many ML codebases. The actual number doesn't matter — it just needs to be consistent.

---

### 3.15 Smoke Test

**File:** `scripts/smoke_test.py`
**Built in:** Week 1

#### What it does

A quick sanity check that everything is installed correctly and fits within VRAM. It has two modes:

**Models-only mode** (no `--scene` argument):
- Loads YOLOv8n and CLIP ViT-B/32 to GPU
- Runs 100 steps on random noise images
- Reports peak VRAM every 10 steps

**Full mode** (with `--scene path/to/scene.glb`):
- Also loads a Habitat scene and uses real rendered images

**VRAM budgets:**
- Expected peak: 2.6–3.0 GB
- Warning threshold: 3.5 GB (likely fp32 leak)
- If under 2 GB: room for stretch features

**`get_vram_mb()`** uses `torch.cuda.max_memory_allocated()` — this tracks the peak VRAM usage since the last reset, which is more useful than current usage (peak tells you if you'll run out of memory).

---

### 3.16 GoatAgent State Machine

**File:** `src/agent/goat_agent.py`
**Week:** 8
**Purpose:** The top-level controller that wires every subsystem together into a single sense-plan-act loop. This is the "brain" of the agent — it decides what to do each step based on its current state and what it perceives.

#### The State Machine

GoatAgent uses a **finite state machine (FSM)** with four states, represented by the `AgentState` enum:

| State | Meaning | Entry condition |
|-------|---------|----------------|
| `SEARCHING` | No matching object found yet. Explore the environment via frontiers. | Initial state after `reset()`. |
| `APPROACHING` | A candidate object was found in memory. Navigate toward it. | Goal matcher returns a match with score > 0. |
| `VERIFYING` | Agent is close enough to the candidate. Confirm it's really the goal. | Agent within `success_distance` (default 1.0 m) of the matched node. |
| `DONE` | Episode over — either success or timeout. | Visual confirmation in VERIFYING, or timeout, or no plan possible. |

**Why a state machine?** The agent needs different behaviors at different stages of a task. When searching, it should explore new areas. When approaching, it should follow a path. When verifying, it should look carefully. A state machine makes these modes explicit and testable — each state has clear entry/exit conditions and a clear action policy.

#### The `act()` Loop

Each call to `act(obs)` runs one full sense-plan-act cycle:

1. **Timeout check** — if we've exceeded `max_steps` (default 500), transition to DONE and return stop (action 0). This prevents the agent from running forever on a single subtask.

2. **Perception** — run the perception pipeline on the current observation to detect objects. Update the occupancy map with the new depth frame and any detected objects. Update the instance memory with new detections.

3. **Goal matching** — ask the GoalMatcher to search memory for an object that matches the current goal. This returns either a matching InstanceNode with a confidence score, or None.

4. **State transitions** — apply the FSM transition rules:
   - SEARCHING → APPROACHING: matcher found a candidate.
   - APPROACHING → VERIFYING: agent is within `success_distance` of the matched node's world position.
   - VERIFYING → DONE: the goal class is detected in the current camera frame (success!).
   - VERIFYING → APPROACHING (revert): 5 verification steps passed without visual confirmation — the match may have been wrong, so go back to approaching.

5. **Action selection** — depends on the current state:
   - **SEARCHING:** find frontiers on the occupancy map, pick the best one using `choose_frontier()`, plan an A* path to it, and follow that path using `path_to_action()`.
   - **APPROACHING:** plan an A* path to the matched node's world position, and follow it.
   - **VERIFYING:** move forward slowly (action 1) to get a better view of the candidate object.
   - **DONE:** always return stop (action 0).

#### Path Following

The agent maintains a `_current_path` (list of grid cells from A*) and a `_path_idx` (how far along the path it is). When following a path:

- A **lookahead** of 3 cells is used — instead of navigating to the immediate next cell, the agent aims at a waypoint a few cells ahead. This produces smoother movement and avoids excessive turning at every single grid cell.
- `path_to_action()` converts the angular difference between the agent's heading and the direction to the waypoint into an action (forward if aligned, turn left/right if not).
- The path index advances when the agent moves forward.
- If the path is exhausted or becomes invalid, the agent replans.

#### Failure Handling

- **No plan possible:** if A* returns None (path blocked) for 2 consecutive steps, the agent gives up and transitions to DONE. This prevents infinite spinning when the agent is trapped.
- **Stale verification:** if the agent reaches a candidate but can't see it after 5 steps of looking, it reverts to APPROACHING to try again. This handles cases where the memory position was slightly wrong.

#### The `reset()` Method

Called at the start of each subtask. With `keep_memory=False` (default), everything resets. With `keep_memory=True`, the instance memory and map persist across subtasks — this is the **lifelong** aspect of the system. The agent remembers objects it saw while pursuing previous goals, so if it's later asked to find one of those objects, it can navigate directly to it without re-exploring.

#### Constructor Parameters

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `perception` | PerceptionPipeline | — | Runs YOLO + CLIP on each frame |
| `memory` | InstanceDatabase | — | Stores and merges detected objects |
| `matcher` | GoalMatcher | — | Matches goals to memory entries |
| `semantic_map` | SemanticMap | — | 2D occupancy grid for planning |
| `intrinsics` | np.ndarray (3x3) | — | Camera intrinsic matrix for depth projection |
| `max_steps` | int | 500 | Hard timeout per subtask |
| `success_distance` | float | 1.0 | Distance threshold (meters) for APPROACHING → VERIFYING |

#### Testing Strategy

Tests use **fake subsystems** (FakePerception, FakeMemory, FakeMatcher, FakeMap) that return configurable results. This lets us test the state machine logic in isolation without needing GPU, Habitat, or real models. Key tests verify:
- Initial state is SEARCHING after reset
- SEARCHING → APPROACHING when matcher finds a match
- APPROACHING → VERIFYING when agent is close
- Timeout triggers DONE and returns stop action
- Perception pipeline runs every step and feeds memory
- `keep_memory=True` preserves memory across resets
- Actions are always valid (0–3)
- Agent runs many steps without crashing

---

### 3.17 GOAT-Bench Dataset Loader

**File:** `src/sim/goat_dataset.py`
**Week:** 9
**Purpose:** Parse GOAT-Bench episode files and provide structured data for evaluation.

#### GOAT-Bench Data Format

GOAT-Bench organizes data by scene. Each scene has a `.json.gz` content file containing:

- **`episodes`**: list of episodes, each with:
  - `episode_id`, `scene_id`, `start_position` (3D), `start_rotation` (quaternion)
  - `tasks`: list of subtasks as tuples `[category, modality, goal_key, ...]`
- **`goals`**: dict mapping `{scene_base}_{category}` to a list of goal instances, each with:
  - `object_category`, `object_id`, `position` (3D ground truth)
  - `lang_desc` (natural language description for language goals)
  - `image_goals` (reference images for image goals)
  - `view_points` (valid viewpoints where the goal is visible)

#### Modality Mapping

GOAT-Bench uses different names than our internal convention:
- `"object"` → `"category"` (find any instance of this class)
- `"description"` → `"language"` (find the specific instance described in natural language)
- `"image"` → `"image"` (find the instance shown in a reference image)

#### Key Functions

- **`load_scene_episodes()`**: parse one scene's content file into `EpisodeSpec` objects
- **`load_val_unseen()`**: load all 360 val_unseen episodes across 36 scenes
- **`sample_dev_subset()`**: deterministically sample 30 episodes (218 subtasks) for fast iteration

---

### 3.18 Evaluation Metrics

**File:** `src/eval/metrics.py`
**Week:** 9
**Purpose:** Compute standard navigation metrics: Success Rate (SR) and SPL.

#### Success Rate (SR)

A subtask is **successful** if BOTH conditions are met when the agent calls STOP:
1. The agent is within `success_distance` (1.0 m) of the ground-truth goal instance position, measured on the XZ plane (ignoring height)
2. The correct goal instance is visible in the agent's camera view

This is stricter than just "being near any instance of that class" — GOAT-Bench measures against a **specific** ground-truth instance identified by `object_id`.

#### SPL (Success weighted by Path Length)

```
SPL = success * shortest_path / max(shortest_path, actual_path)
```

SPL rewards efficiency: a successful agent that took the shortest possible path gets SPL = 1.0, while one that wandered gets a lower score proportional to how much extra distance it traveled. Failed subtasks always get SPL = 0.0.

The `shortest_path` is the **geodesic distance** — the shortest walkable path through the navmesh, computed by Habitat's pathfinder. This accounts for walls and obstacles.

#### Aggregation

Results are broken down by:
- **Modality**: SR/SPL for category, language, and image goals separately
- **Subtask index**: SR/SPL for the 1st, 2nd, 3rd, etc. subtask in each episode — this is the "lifelong curve" that shows whether memory helps on later subtasks

---

### 3.19 Evaluation Runner

**File:** `src/eval/runner.py`
**Week:** 9
**Purpose:** Run the agent on GOAT-Bench episodes, collecting per-subtask results.

#### Episode Execution

For each episode:
1. Set the agent to the episode's start position and rotation
2. For each subtask in order:
   - Create a `GoalSpec` from the subtask's modality and category
   - Reset the agent with `keep_memory=True` (except for the first subtask)
   - Compute the geodesic shortest path from current position to the goal
   - Run the agent's `act()` loop until it returns STOP or hits the step limit
   - Track the actual path length by accumulating GPS displacements
   - Evaluate success using `subtask_success()`
   - Compute SPL

#### Failure Classification

Each failed subtask is categorized:
- **`timeout`**: agent hit the 500-step limit without calling STOP
- **`wrong_stop`**: agent stopped close to the goal but didn't see it (wrong orientation)
- **`no_plan`**: agent stopped far from the goal (couldn't find or reach it)

This taxonomy helps diagnose the top failure buckets during debugging.

#### Report Generation

`src/eval/report.py` saves three artifacts:
- **`results.csv`**: one row per subtask with all metrics
- **`summary.json`**: aggregated SR, SPL, per-modality and per-subtask-index breakdowns
- **`failures.jsonl`**: one JSON line per failure for easy analysis

---

## 4. Testing Strategy

### Philosophy

We follow a **test-first** approach per PLAN.md: write tests based on the spec, then implement until tests pass.

### Test structure

Tests use **fake/mock objects** instead of real models (which require GPU and large downloads). For example:

**`FakeDetector`** — Returns a pre-configured list of detections, no YOLO needed.

**`FakeEncoder`** — Returns deterministic 512-dim embeddings based on the crop's mean pixel intensity. These are properly L2-normalized, so downstream code behaves correctly.

**`_make_obs()`** — Creates a synthetic Observation with uniform RGB (all pixels = 128) and uniform depth. The identity pose matrix (`np.eye(4)`) places the camera at the origin, looking along the Z axis.

This is called **duck typing** — FakeDetector isn't a subclass of YOLODetector, but it has the same `.detect()` method, so Python treats it the same way. Pyright (the type checker) warns about this but it works fine at runtime.

### What the tests cover

**Transform tests (12 tests in `tests/test_map.py`):**
- Intrinsic matrix values for known FOV
- Back-projected center pixel lands on the optical axis
- Point cloud output shape
- Zero depth produces no valid points
- Identity pose preserves coordinates
- Translation-only pose shifts points correctly
- 90° rotation swaps axes correctly
- Bbox center back-projection returns correct depth
- Zero depth returns None
- World↔grid roundtrip preserves coordinates within resolution
- Origin maps to cell (0,0)
- Positive offset produces expected cell indices

**Pipeline tests (12 tests in `tests/test_perception_pipeline.py`):**
- PerceivedInstance fields have correct shapes
- Empty detections → empty output
- Single detection → one PerceivedInstance with correct fields
- Small crops (< 32px) are filtered out
- Large enough crops pass through
- Multiple detections → correct count and order
- Thumbnail is always 64x64x3
- Invalid depth (0.0) → detection skipped
- Known depth + identity pose → predictable world_xyz
- Bboxes extending past image edge are clipped, not errored
- Step number propagates correctly
- All embeddings are L2-normalized (norm ≈ 1.0)

**Occupancy map tests (14 tests in `tests/test_occupancy.py`):**
- Grid dimensions match size_m / resolution_m
- Initial occupancy is all unknown (-1)
- Initial explored is all False
- World (0,0) maps to center of grid
- World↔grid roundtrip within resolution tolerance
- Depth update marks cells as occupied
- Depth update marks cells along rays as free (Bresenham clearing)
- Depth update marks cells as explored
- Height filter prevents floor/ceiling from becoming obstacles
- Multiple depth updates from different poses accumulate explored area
- Out-of-bounds points don't crash
- Detection update increments class counts at correct class index
- Gaussian kernel spreads counts to neighboring cells
- Multiple detections of different classes both register

**Frontier tests (7 tests in `tests/test_frontier.py`):**
- Fully unknown map → no frontiers
- Fully explored map → no frontiers
- Partial hallway → frontier at the open end
- Room with doorway → frontier at the door opening
- min_size parameter filters small frontiers
- Frontiers sorted by size descending
- Frontier dataclass has correct fields (centroid_ij, size, cells)

**Planner tests (19 tests in `tests/test_planner.py`, 2 skipped if skfmm missing):**
- A* straight line path, path around wall via gap, no path when fully blocked
- Start equals goal returns single-cell path, adjacent cells return 2-cell path
- Diagonal path is shorter than Manhattan distance
- Obstacle inflation keeps path away from walls
- Out-of-bounds start returns None, goal on occupied returns None
- Large 480x480 map completes in < 2 seconds
- FMM straight line path (skipped if skfmm missing), FMM no path when blocked
- Frontier exploration: chooses nearest-large, skips visited, returns None when empty
- Action selection: forward when facing target, turn left/right when target is to the side, stop when at goal

**Goal matcher tests (10 tests in `tests/test_matcher.py`):**
- Category matching finds existing class and returns highest-confidence node
- Category matching returns None for non-existent class
- Category matching returns None on empty database
- Category score equals node confidence
- Language matching returns best embedding match
- Language matching returns None when below threshold
- Language matching returns None on empty database
- Template wrapping: short values get "a photo of" prefix
- Template wrapping: long sentences used directly without wrapping

**Instance memory tests (16 tests in `tests/test_memory.py`):**
- InstanceNode dataclass fields have correct shapes
- Empty database returns no nodes
- Single detection creates one node with correct fields
- Node IDs auto-increment and are unique
- query_by_class returns correct nodes, empty for unknown class
- query_by_embedding returns sorted results, best match first
- 3 close detections of same class merge into 1 node with n_observations=3
- 2 chairs 2m apart → 2 separate nodes
- Close detections with dissimilar embeddings → 2 nodes (embedding gate prevents merge)
- Different classes at same location never merge
- Merged node position is running mean of observations
- Merged node embedding is re-normalized running mean
- Thumbnail kept from highest-confidence observation
- Confidence is running max across merged observations
- Multiple detections in single update call handled correctly
- 100-node database memory stays under 20 MB budget

---

## 5. Glossary of Key Terms

### A* (A-star)
A graph search algorithm that finds the shortest path between two points. Combines the actual cost to reach a cell (g) with a heuristic estimate of remaining cost (h). By always expanding the most promising cell (lowest g+h), it finds optimal paths while exploring far fewer cells than brute-force search. The heuristic must never overestimate (be "admissible") — we use Euclidean distance, which is always ≤ the true path length.

### Angle wrapping
Correcting angular differences to the range [-π, π] to handle the discontinuity at ±180°. Without wrapping, the difference between 350° and 10° would be -340° instead of the correct +20°. Formula: `diff = (diff + π) % (2π) - π`.

### Back-projection
The reverse of what a camera does. A camera **projects** 3D points onto a 2D image (losing depth). **Back-projection** uses a known depth value to recover the original 3D position from a 2D pixel coordinate.

Formula: given pixel `(u, v)` with depth `d` and intrinsics `K`:
```
x = (u - cx) * d / fx
y = (v - cy) * d / fy
z = d
```

### Binary dilation
A morphological image operation that "grows" a boolean mask by one pixel in all directions. If any neighbor of a False pixel is True, that pixel becomes True in the output. We use it in frontier detection: dilating the unknown mask lets us find free cells that are adjacent to unknown cells.

### Bounding box (bbox)
A rectangle around a detected object in an image. Format `(x1, y1, x2, y2)`: top-left corner to bottom-right corner, in pixel coordinates.

### Bresenham's line algorithm
A classic computer graphics algorithm that determines which grid cells a straight line between two points passes through. Uses only integer arithmetic, making it very fast. We use it for ray clearing in the occupancy map — tracing a line from the camera to each detected obstacle and marking intermediate cells as free.

### Camera intrinsics matrix (K)
A 3x3 matrix that encodes a camera's internal properties:
```
K = [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
```
- `fx, fy` = focal lengths in pixels (how "zoomed in" the camera is)
- `cx, cy` = principal point (where the optical axis hits the image sensor, usually the center)

It does NOT include the camera's position or orientation in the world — that's the **extrinsics** (the pose matrix).

### CLIP (Contrastive Language-Image Pre-training)
A neural network from OpenAI that maps both images and text into the same 512-dimensional vector space. Semantically related images and text end up with similar vectors. We use it to match text goals ("find the chair") to stored image crops of objects.

### Connected components
Groups of adjacent pixels/cells that share a property. `scipy.ndimage.label` finds connected components in a boolean mask — it assigns a unique integer label to each group of touching True cells. We use it to group frontier cells into distinct frontiers.

### Confidence score
A number from 0 to 1 indicating how sure the detector is about a detection. Higher = more confident. We filter at `conf=0.35` by default, meaning detections below 35% confidence are discarded.

### Cosine similarity
A measure of how similar two vectors are, based on the angle between them. Ranges from -1 (opposite) to +1 (identical direction). After L2 normalization, this simplifies to a dot product.

### Dataclass
A Python `@dataclass` decorator that auto-generates `__init__`, `__repr__`, and other methods for classes that are mainly containers for data. Less boilerplate than writing `def __init__(self, x, y, z): self.x = x; self.y = y; self.z = z`.

### Depth image / depth map
A 2D array where each pixel value is the distance (in meters) from the camera to the surface at that pixel. Not a color — it's a measurement. A value of `3.5` at pixel (100, 100) means "the surface at pixel (100, 100) is 3.5 meters away."

### Duck typing
A Python concept: "if it walks like a duck and quacks like a duck, it's a duck." In our tests, `FakeDetector` has a `.detect()` method that matches `YOLODetector.detect()`, so Python accepts it wherever a YOLODetector is expected — no inheritance needed.

### Downsample
Reducing the number of data points by keeping every Nth one. In `update_from_depth`, we take every 4th point from the point cloud (`pc[::4]`), reducing computation 4x while barely affecting map quality since neighboring pixels map to the same or adjacent grid cells anyway.

### Embedding
A fixed-length vector (array of numbers) that represents something (an image, a word, a sentence) in a way that captures its meaning. Semantically similar things have similar embeddings (close together in vector space).

### Epoch
One complete pass through the entire training dataset during model training. "30 epochs" means the model sees every training example 30 times.

### Finite State Machine (FSM)
A computational model with a fixed number of states. The system is always in exactly one state, and transitions between states are triggered by specific conditions. In GoatAgent, the four states are SEARCHING, APPROACHING, VERIFYING, and DONE. FSMs are popular in robotics and game AI because they make behavior explicit, testable, and easy to debug — you can always ask "what state is the agent in?" and "why did it transition?"

### fp16 / fp32 (half-precision / single-precision)
How many bits are used to represent a floating-point number:
- **fp32** (float32): 32 bits — standard precision, ~7 decimal digits
- **fp16** (float16): 16 bits — half precision, ~3 decimal digits

fp16 uses half the memory and is faster on modern GPUs, but can cause numerical instability in some operations (like softmax on large tensors). We use fp16 for inference to fit within 4 GB VRAM, with occasional casts back to fp32 for sensitive operations.

### Frontier
In exploration robotics, a frontier is the boundary between known free space and unknown space. It's the most informative place to go — moving to a frontier will reveal new areas. Our agent navigates to the largest nearby frontier when it doesn't know where its goal is.

### Fast Marching Method (FMM)
A numerical algorithm that simulates wavefront propagation — like dropping a stone in water and measuring when the ripples reach each point. Used for pathfinding by propagating from the goal, then following the gradient downhill from the start. Produces smoother paths than A* because it operates in continuous space rather than on a discrete grid.

### FOV (Field of View)
How wide the camera's "cone of vision" is, in degrees. Our camera uses 90° horizontal FOV — it can see 45° to the left and 45° to the right of center. A wider FOV sees more but with more distortion.

### Geodesic Distance
The shortest walkable path between two points, respecting walls and obstacles. Unlike Euclidean (straight-line) distance, geodesic distance follows the navigable mesh (navmesh). If two points are 3m apart in a straight line but separated by a wall, the geodesic distance might be 15m because you have to walk around. Used in SPL computation.

### Gradient
In neural networks, gradients tell you how to adjust model weights to reduce error. During inference (using a trained model), we don't need gradients, so we use `torch.no_grad()` to skip computing them — this saves ~50% of memory.

### Heuristic (in search algorithms)
An estimate of the remaining cost from a given state to the goal. In A*, the Euclidean distance heuristic estimates the straight-line distance to the goal. A "good" heuristic is close to the true cost but never overestimates (admissible), which guarantees A* finds the optimal path.

### Habitat / habitat-sim
Facebook AI Research's 3D simulation platform. It renders photorealistic indoor scenes from the HM3D (Habitat-Matterport 3D) dataset and lets virtual agents navigate them. It's the "game engine" our agent lives in.

### HM3D (Habitat-Matterport 3D)
A dataset of ~1000 real indoor spaces, captured with Matterport 3D scanners and reconstructed as 3D meshes. We load these into Habitat to create realistic test environments.

### IoU (Intersection over Union)
A measure of how much two bounding boxes overlap. Used in NMS to decide if two boxes are detecting the same object:
```
IoU = (area of overlap) / (area of union)
```
IoU = 1.0 means identical boxes. IoU = 0.0 means no overlap.

### L2 normalization
Dividing a vector by its length (L2 norm) so the result has length 1. After normalization, the dot product of two vectors equals their cosine similarity.
```python
normalized = vector / np.linalg.norm(vector)
```

### Lifelong (navigation)
The ability to retain knowledge across multiple tasks within the same environment. In GOAT-Lite, when the agent finishes searching for one object and is given a new goal, it keeps its instance memory and occupancy map via `reset(obs, keep_memory=True)`. If it saw a chair while looking for a table, it can navigate directly to that chair later without re-exploring. This is a key advantage over episodic agents that start from scratch each time.

### Lookahead (path following)
Instead of steering toward the immediately next cell on a path, the agent looks a few cells ahead (default 3) and steers toward that waypoint. This produces smoother trajectories — without lookahead, the agent would make tiny corrections at every grid cell, resulting in jerky zig-zag movement. The tradeoff is that the agent may cut corners slightly, but in practice this is negligible.

### mAP (mean Average Precision)
The standard metric for object detection accuracy. mAP@0.5 means "average precision at IoU threshold 0.5." Higher is better. Our target for the finetuned YOLO is mAP@0.5 >= 0.55.

### Morphological operations
Image processing techniques that operate on shapes/structures in binary images. Common ones: **dilation** (grow regions), **erosion** (shrink regions), **opening** (erosion then dilation — removes small noise), **closing** (dilation then erosion — fills small gaps). We use dilation in frontier detection.

### Navmesh (Navigation Mesh)
A precomputed mesh of walkable surfaces in a 3D environment. Habitat uses the navmesh to determine which positions are navigable and to compute geodesic distances. The navmesh is stored alongside each HM3D scene as a `.navmesh` file and accounts for walls, furniture, stairs, etc.

### NMS (Non-Maximum Suppression)
When a detector produces multiple overlapping boxes for the same object, NMS keeps only the highest-confidence one and removes the rest. The `iou=0.5` parameter means: if two boxes have IoU > 0.5, the weaker one is suppressed.

### Merging (instance memory)
The process of combining multiple observations of the same physical object into a single memory node. Our merging uses two gates: spatial proximity (within 0.75m) AND embedding similarity (cosine > 0.85). When merging, position and embedding are updated via running mean, confidence via running max, and the best thumbnail is kept. This deduplication is critical — without it, 100 detections of one chair would create 100 nodes instead of 1.

### Obstacle inflation
Expanding obstacles on the occupancy grid by the robot's radius so the planner treats the robot as a point. If a wall is inflated by 4 cells (20cm), any path the point-robot finds through the inflated grid will have at least 20cm clearance from the real wall — enough for the physical robot (17cm radius) to pass safely.

### Occupancy grid
A 2D grid where each cell records whether that physical location is free (passable), occupied (blocked by an obstacle), or unknown (not yet observed). The fundamental data structure for 2D robot navigation — the planner uses it to find paths, and the explorer uses it to find frontiers.

### OpenGL convention
A coordinate system where the camera looks along the **negative Z axis**, with X pointing right and Y pointing up. Habitat uses this convention. It's different from some robotics conventions where Z is forward.

### Point cloud
A set of 3D points representing surfaces in the scene. Created by back-projecting every pixel of a depth image into 3D space. Shape: `[N, 3]` where each row is an `(x, y, z)` coordinate.

### Pose / SE(3)
A 4x4 matrix encoding an object's position and orientation in 3D space:
```
[ R | t ]     R = 3x3 rotation matrix
[ 0 | 1 ]     t = 3x1 translation (position)
```
SE(3) = "Special Euclidean group in 3 dimensions" — the mathematical group of all rotations and translations in 3D.

### Running mean
An incremental average that updates with each new observation without needing to store all past values. Formula: `new_mean = (old_mean * n + new_value) / (n + 1)`. We use this in the instance database to maintain average position and embedding as new detections arrive. Equivalent to the full average but computed incrementally.

### Running max
Keeping track of the maximum value seen so far. Updated as: `new_max = max(old_max, new_value)`. Used for confidence in the instance database — we want the highest confidence ever observed for an object, not the average.

### Ray clearing
A technique for building occupancy maps. When a depth sensor sees an obstacle at distance D, we know that everything between the sensor and the obstacle (distance 0 to D) must be free space (otherwise the sensor couldn't see through it). We "clear" those intermediate cells by marking them as free. Implemented using Bresenham's line algorithm to efficiently find which grid cells the ray passes through.

### Quaternion
A 4-component number `(w, x, y, z)` used to represent 3D rotations. More compact than a 3x3 rotation matrix (4 numbers vs 9) and avoids **gimbal lock** (a problem where Euler angles lose a degree of freedom at certain orientations). Habitat stores rotations as quaternions internally.

### Resolution (image)
Image dimensions in pixels, e.g., 256x256. More pixels = more detail but more compute and VRAM.

### Resolution (map)
Meters per grid cell. Our default is 0.05 m (5 cm). A 24m x 24m room at 5cm resolution = a 480x480 grid.

### SPL (Success weighted by Path Length)
The standard efficiency metric for navigation: `SPL = success * shortest_path / max(shortest_path, actual_path)`. A perfect agent that always takes the shortest path scores SPL = 1.0. An agent that succeeds but wanders gets a lower SPL proportional to how much extra it traveled. Failed subtasks contribute SPL = 0.0. SPL is the primary metric in the GOAT-Bench leaderboard — it rewards both finding the goal AND finding it efficiently.

### Success Rate (SR)
The fraction of subtasks where the agent successfully navigated to the correct goal instance. In GOAT-Bench, success requires being within 1.0m of the goal AND having it visible in the camera when STOP is called. SR measures capability (can the agent find things?) while SPL measures efficiency (does it find them quickly?).

### Tokenization
Converting text into numbers that a neural network can process. The word "chair" might become token ID `7245`. CLIP uses its own tokenizer trained alongside the model.

### VRAM (Video RAM)
Memory on the GPU. Our budget is 4 GB. Everything that runs on the GPU (model weights, input tensors, intermediate activations) must fit in VRAM simultaneously.

### Yaw / Pitch / Roll
Three angles describing a 3D rotation:
- **Yaw** = rotation around the vertical axis (turning left/right)
- **Pitch** = tilting up/down
- **Roll** = tilting sideways

Our agent only yaws (turns left/right on the ground plane). We extract yaw from the quaternion in `get_compass()`.

### YOLO (You Only Look Once)
A family of real-time object detection neural networks. "Only look once" means it processes the entire image in a single forward pass (unlike older methods that scanned the image with sliding windows). YOLOv8n is the nano (smallest) variant of version 8.

---

## 6. Engineering Log

A dated, append-only record of problems hit while getting the system running and
how each was diagnosed and fixed. Newest session at the bottom. The point of
this section is that each *symptom* is written next to its *cause*, because most
of these presented as something misleading.

---

### 2026-08-10 — Porting the detector-finetune pipeline to Windows/WSL2

**Context.** Week 9 was blocked on `checkpoints/yolo_goat.pt`: with stock COCO
YOLOv8n the agent cannot see 30 of the 36 GOAT categories, so dev-eval SR was
0%. The original plan was to render the training set on a second machine
(`TRAIN_YOLO.md`). That machine was unavailable, so the pipeline was moved to
the Windows box, which has the disk and an RTX 3050 (4 GB).

#### 6.1 habitat-sim does not run on Windows

**Symptom.** No `habitat-sim` candidate for `win-64` on conda-forge or
`aihabitat`.

**Cause.** There is no Windows build. Not a resolver problem.

**Fix.** Run the render pipeline inside WSL2 (Ubuntu 24.04), with data on
`/mnt/d` so it lands on the D: drive rather than inside the distro's
`ext4.vhdx` — that file lives on C:, which had only ~43 GB free. Miniforge plus
`habitat-sim 0.3.3 py3.9 headless` from the `aihabitat` channel.

#### 6.2 No GPU rendering in WSL: "unable to find CUDA device 0"

**Symptom.**

```
Platform::WindowlessEglApplication::tryCreateContext():
    unable to find CUDA device 0 among 1 EGL devices in total
WindowlessContext: Unable to create windowless context
```

and, once that was worked around, a segfault.

**Cause.** Two independent problems.

1. WSL2 ships **no NVIDIA EGL/GLX driver** — `/usr/lib/wsl/lib` has
   `libcuda.so` and `libd3d12core.so` but no `libEGL_nvidia.so`. Hardware GL
   comes from Mesa's **d3d12** gallium driver via `/dev/dxg`, so no EGL device
   advertises a CUDA id, yet habitat's default `gpu_device_id=0` asks Magnum to
   find one.
2. Mesa defaults to the **llvmpipe** software rasteriser, and the EGL platforms
   habitat tries (`device`, GBM) both fail because WSL has no `/dev/dri`.

**Diagnosis.** `eglinfo -B` showed only the Wayland platform working, renderer
`llvmpipe`. Forcing `GALLIUM_DRIVER=d3d12` with
`LD_LIBRARY_PATH=/usr/lib/wsl/lib` gave `D3D12 (Intel(R) UHD Graphics)` —
hardware, but the wrong GPU. Probing each EGL platform in turn showed
`surfaceless` works where `device` does not.

**Fix.** Four environment variables plus one code change:

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export GALLIUM_DRIVER=d3d12                      # else llvmpipe (CPU)
export EGL_PLATFORM=surfaceless                  # device/GBM fail: no /dev/dri
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA    # else the Intel iGPU
export HABITAT_GPU_DEVICE_ID=-1                  # skip EGL<->CUDA matching
```

`HabitatEnv` and `make_yolo_dataset.py` read `HABITAT_GPU_DEVICE_ID` (default
`0`), so native-Linux machines are unaffected.

**Result.** `D3D12 (NVIDIA GeForce RTX 3050 Laptop GPU)`, RGB + semantic at
512x512, **~128 fps**.

#### 6.3 Every .glb was a 133-byte text file

**Symptom.** `Utility::Json: unexpected v at apartment_1.glb:1:1` followed by
`Trade::GltfImporter::openData(): invalid JSON`.

**Cause.** `git-lfs` was not installed. habitat's downloader clones from
HuggingFace, so every `.glb` arrived as an LFS **pointer file** starting
`version https://git-lfs.github.com/spec/v1` — hence "unexpected v".

**Fix.** `apt-get install git-lfs && git lfs install`, then `git lfs pull`.
Cheap early check: `ls -l` any `.glb`; 133 bytes means pointer, not mesh.

#### 6.4 numpy 2.x silently breaks habitat-sim

**Symptom.** `AttributeError: _ARRAY_API not found`, then
`ImportError: numpy.core.multiarray failed to import`, raised from
`quaternion/__init__.py` during `import habitat_sim`.

**Cause.** Installing torch/ultralytics upgrades numpy to 2.x, and
habitat-sim's `quaternion` C extension is compiled against numpy 1.x.

**Fix.** Pin `numpy==1.26.4` **after** installing torch, and use the last
numpy-1.x-compatible builds of the other binary packages:
`opencv-python-headless==4.10.0.84` and `pillow==10.4.0`. Verify with real array
interop rather than imports alone — `cv2` imported cleanly while still being
ABI-incompatible.

Working set: numpy 1.26.4, torch 2.6.0+cu124, ultralytics 8.4.117,
open_clip_torch 3.3.0, habitat-sim 0.3.3.

#### 6.5 requirements.txt is not installable as written

`numpy==1.24.4` conflicts with habitat-sim, and `ultralytics==8.1.34` calls
`torch.load()` without `weights_only=False`, which torch >= 2.6 refuses. The
pins were left in place (they may hold on the original Linux box) and the
working combination documented in `TRAIN_YOLO.md` Section 0.

**Also missing entirely: `scikit-fmm`.** `src/planning/fmm.py` catches the
`ImportError` and returns `None`, so the agent silently degrades to the A*
planner and two planner tests skip. Now added to `requirements.txt`. This
mattered — the eval would otherwise have run a different planner than the one
the method section describes.

#### 6.6 A WSL DNS failure that looked like an auth failure

**Symptom.** `curl rc=6` against `api.matterport.com`, immediately after the
same URL had worked.

**Cause.** WSL's NAT DNS proxy (`nameserver 10.255.255.254`) stopped resolving
mid-session. `rc=6` is "couldn't resolve host", but in context it read as a
rejected HM3D token.

**Fix.** `1.1.1.1` in `/etc/resolv.conf`, plus `generateResolvConf = false` in
`/etc/wsl.conf` so WSL stops regenerating the file.

#### 6.7 An apparent 10 KB/s download of a 40 GB dataset

**Symptom.** HM3D tars downloading at ~10 KB/s — an ETA measured in weeks.

**Cause.** Self-inflicted contention: the CLIP weights fetch (triggered by the
test suite), a pip install, and an unnecessary `hm3d_example_full` download were
all running at once. An A/B test with everything else stopped measured
**4.5 MB/s** on a single connection. aria2 with 16 connections returned 0 bytes
— its basic-auth handling does not survive the CDN redirect — so plain `curl` is
the right tool here.

Secondary lesson: an aggressive `--speed-limit 2048 --speed-time 120` made curl
abort and re-handshake constantly under contention. Relaxed to 1 KB/s over 300 s.

The download uses `curl -C -` for resume rather than habitat's downloader,
because a 27 GB fetch over wifi will drop and the downloader restarts from zero.

#### 6.8 The big one: semantics never loaded, so every label file was empty

**Symptom.** Exactly the failure `TRAIN_YOLO.md` warns about. Across 75 rendered
frames: **0 boxes, 0 GOAT instances, 0 distinct categories.** Buried in the log:

```
SSD Load Failure! File with SemanticAttributes-provided name
  `.../00800-TEEsavR23oF/TEEsavR23oF.basis.scn` exists but failed to load.
```

**Cause.** `make_sim()` set `scene_dataset_config_file = "default"`. With
`"default"` habitat treats the `.basis.glb` as a bare stage and guesses a
sibling semantic descriptor `<scene>.basis.scn`, which does not exist in HM3D.
The `.semantic.glb` / `.semantic.txt` pair is therefore never attached,
`sim.semantic_scene.objects` is empty, and every frame yields no boxes.

HM3D ships `hm3d_annotated_*_basis.scene_dataset_config.json` (in the
`*-semantic-configs-*` tar) whose stage defaults declare:

```json
"semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.txt",
"semantic_asset":               "%%CONFIG_NAME_AS_ASSET_FILENAME%%.semantic.glb",
"has_semantic_textures": true
```

**Proof.** Same scene, only `scene_dataset_config_file` changed:

| `scene_dataset_config_file` | semantic objects | distinct categories |
|---|---|---|
| `"default"` (old) | 0 | 0 |
| annotated config (new) | **661** | **129** |

**Fix.** `make_yolo_dataset.py` gained `--scene-dataset-config`, auto-detected
from `--hm3d-root` by `find_scene_dataset_config()`, and **hard-fails** when no
config is found. A second guard aborts if the first 5 scenes produce no boxes,
rather than spending hours writing an empty dataset.

**Result.** 75 frames produced 129 boxes, 52% of frames non-empty, 25/36 GOAT
classes present in a 3-scene sample.

#### 6.9 Whole categories lost to naming variants

**Symptom.** After the fix, `stairs` appeared 65 times in the *unmapped* list
while GOAT's `stair` matched only 5 instances. `island` had **zero** direct
matches and existed only as `kitchen island`.

**Cause.** HM3D-Semantics category strings are free-form annotator text, and
`name_to_id()` matches exactly.

**Fix.** Added `HM3D_CATEGORY_ALIASES` and `hm3d_name_to_id()` to
`goat_classes.py`, used *only* when ingesting HM3D annotations. `name_to_id()`
is unchanged, so the 36 emitted class names and the matcher are untouched.

Curated by hand from a 12-scene census; several automatic suggestions were
rejected as different objects: `book rack`, `piano stool`, `piano bench`,
`mirror frame`, `refrigerator cabinet`, `stair wall`, `stair handle`, and
`glasses` (eyewear, not a glass surface).

**Result.** Same 3 scenes: 129 -> **175 boxes** (+36%); non-empty frames
52% -> 59%.

#### 6.10 Training validated before the data existed

To avoid discovering a training-time problem after hours of rendering, the
trainer was exercised on a synthetic 36-class dataset:

- `scripts/finetune_yolo.py` runs end to end and exports a checkpoint whose
  `model.names` equals `GOAT_CATEGORIES` exactly (TRAIN_YOLO.md step 5).
- **Peak VRAM at `--batch 24 --imgsz 512`: 2247 MiB of 4096.** Batch 24-32 is
  safe on the 4 GB card; the documented `--batch 16` is conservative.

#### 6.11 imgsz mismatch between training and inference

`TRAIN_YOLO.md` specifies `--imgsz 512` while `detector.py` and
`agent_default.yaml` both inferred at 256 — precisely the pitfall in PLAN.md
Section 9.4. Resolved to 512 on both sides.

Still open: `HabitatEnv` renders the agent camera at 256x256, so at eval time a
256 render is upscaled to 512 for the detector, while training images are
rendered at 512 natively. This is a mild sharpness domain gap. The alternative
is raising the agent camera to 512 (slower sim, more VRAM). Flagged, not yet
decided.

#### 6.12 GOAT-Bench episodes: sourcing and verification

The eval needs per-scene shards at
`data/datasets/goat_bench/hm3d/v1/val_unseen/content/*.json.gz`. These are not
part of HM3D; they come from the GOAT-Bench repo (`Ram81/goat-bench`), hosted on
Google Drive as a 68 MB zip and publicly downloadable via `gdown`.

Loader verification against the published paper numbers:

| quantity | loaded | paper |
|---|---|---|
| episodes | 360 | 360 |
| subtasks | 2669 | 2669 |
| scenes | 36 | 36 |

Modality split: 991 category / 856 language / 822 image. The 30-episode dev
subset is deterministic across runs at seed 42.

#### 6.13 Category subtasks were hard-coded to failure (37% of the benchmark)

**Symptom.** Only 1678 of 2669 subtasks had `goal_info`. The 991 without it were
*exactly* the category-modality subtasks.

**Cause.** GOAT-Bench writes a category task as `['freezer', 'object', None]` —
there is deliberately no `object_id`, because **any** instance of the category
counts as success. The loader only built a `GoalInfo` when `goal_key` was not
None, so every category subtask got `goal_info=None`. `runner.py` then hit:

```python
else:
    # No goal info — can't evaluate properly
    success = False
```

So every category subtask was scored as a failure regardless of what the agent
did. Overall SR was capped at 63%, and **SR for the category modality was
structurally pinned at 0%** — one of the two primary modalities in PLAN.md.

This is worth emphasising because it would have been extremely easy to
misattribute during Week 9 debugging: the detector was already suspected, so a
category SR of 0 would have looked like more evidence for that, and no amount of
detector improvement would ever have moved it.

**Fix.** `SubtaskSpec` gained `goal_candidates: list[GoalInfo]` — the set of
instances that count as reaching the goal:

- language / image -> the one referenced instance (unchanged behaviour)
- category -> every annotated instance of that category in the scene

`runner.py` now scores against that list: SPL uses the geodesic distance to the
nearest instance, and success uses the closest one. `goal_info` is retained for
the language description and image goals.

**Result.** Scoreable subtasks 1678 -> **2669 (all of them)**. Regression tests
in `tests/test_goat_dataset.py` cover both directions: category subtasks must
expose candidates, and language/image subtasks must *not* widen to every
instance of their category.

#### 6.14 Split separation, to make val leakage structurally impossible

Both HM3D tars extract **flat**: val as `00800-<hash>/`, train as `000NN-<hash>/`.
Dropped into one directory they are indistinguishable to `find_scenes()`, which
simply walks the tree — so the detector would have been trained on val_unseen
scenes and every reported number would have been invalid. Nothing would have
errored; the SR would just have been quietly optimistic.

Extraction therefore routes by split into separate roots:

```
/mnt/d/goat-data/hm3d_train   <- detector training only  (145 annotated scenes)
/mnt/d/goat-data/hm3d_val     <- evaluation only         (36 annotated scenes)
```

`extract_split.sh` asserts the two roots share no scene id. Physical separation
beats a filter, because a filter can be forgotten at the next call site.

Related disk note: `hm3d-train-habitat-v0.2.tar` holds all 800 train scenes
(~27 GB), but HM3D-Semantics annotates only **145**, and a scene without
`.semantic.glb` can never yield a label. `extract_train_basis.sh` therefore
extracts only the 145 annotated scene dirs, saving roughly 22 GB.

#### 6.15 Eval preflight (checks worth repeating before any long run)

With the val split and episodes in place, the following were verified before
committing to a 30-episode run:

- **Scene resolution.** All 36 distinct `scene_id`s resolve to files that exist
  via `_resolve_scene_path`. 10 episodes per scene, 360 total.
- **Navmesh loads.** `pathfinder.is_loaded` is True, 75.4 m² navigable on the
  first scene, and a sample geodesic returns a finite 4.77 m. Without this,
  `_geodesic_distance` returns `inf` and every SPL is silently 0.
- **Episode start poses are navigable.**
- **Depth is in metres** (PLAN.md Section 9 pitfall 2). Random navigable poses
  give medians of 0.5-2.2 m and maxima to 6.2 m.

Two observations that look alarming but are not bugs:

1. `env.reset()` leaves the agent at an unset default pose, which on the first
   val scene sits ~0.24 m from geometry — depth then spans a 7 mm range and
   looks like a units error. The eval never uses that pose; `run_episode` sets
   the episode start pose explicitly. Only the *pose* is degenerate.
2. At a genuine episode start pose, **~70% of depth pixels are zero**. Habitat
   returns 0 where there is no geometry, and HM3D scans have open doorways and
   holes. Normal, but it means occupancy updates are sparser than the raw pixel
   count suggests — worth remembering if the map looks thin while debugging.

**Full test suite is green for the first time: 139 passed, 0 skipped.** The 7
`test_agent_smoke` tests had been skipping for want of HM3D scenes, so
`HabitatEnv`, `YOLODetector` and `ClipEncoder` had never actually been exercised
against a real scene in this environment. The repo's expected paths are provided
by gitignored symlinks:

```
data/hm3d                        -> hm3d_val    (eval root, 36 scenes)
data/hm3d_train                  -> hm3d_train  (145 scenes)
data/datasets/goat_bench/hm3d/v1 -> GOAT-Bench episodes
```

#### 6.16 The agent spawned upside down in every episode

This is the big one, and it had nothing to do with the detector.

**Symptom.** A 1-episode dry run finished all 10 subtasks but with
`path_length = 0.00 m` on every one, including two that ran the full 500 steps.
`distance_to_goal` repeated exactly per category (mirror 4.06 three times,
dresser 4.31 four times), i.e. the agent's final pose was identical every time.
Eight subtasks ended `no_plan` after exactly **12** steps -- and 360/30 = 12,
one full turn-in-place scan.

**Ruling things out.** Direct actuation was fine: 10 `move_forward` steps moved
the agent 2.495 m, `get_gps()` tracked it exactly, all 8 headings produced
motion, navmesh loaded, start pose navigable. So the environment worked and the
*agent* was never emitting forward.

Instrumenting the FSM showed why: after 12 steps the occupancy map was still
completely empty -- `explored=0, free=0, occupied=0, unknown=230400`. No map, so
no frontiers, so no plan, so spin 12 times and stop.

`update_from_depth()` *was* being called every step. Tracing it stage by stage:

```
world Y: min=-6.939  median=-0.536  p95=-0.536  max=-0.536
agent y = -0.536
```

Every projected point landed at or below camera height -- maximum exactly equal
to it -- so the height band `[floor+0.1, floor+1.5]` kept **0 of 16384 points**
at any floor value. And 100% of valid depth pixels were in the bottom half of
the image: the camera was staring at the floor.

**Cause.** `runner.py` did:

```python
state.rotation = qt.from_float_array(episode.start_rotation)
```

habitat's episode JSON stores rotations as **(x, y, z, w)**. numpy-quaternion's
`from_float_array` reads **(w, x, y, z)**. For a typical yaw-only start pose the
misread turns the rotation into a ~180 degree roll:

| | current | fixed |
|---|---|---|
| rotation angle | **180.0 deg** | 3.7 deg |
| agent "up" vector | **[0, -1, 0]** | [0, 1, 0] |
| episodes upside down | **360 / 360** | **0 / 360** |

**Why it hid so well.** The *forward* vector is identical under both readings
(`[0.065, 0, -0.998]`). The agent walks in the correct direction, so actuation
tests, navmesh checks and `test_step_forward_changes_position` all pass. Only
the roll is wrong -- the camera is upside down. This is PLAN.md Section 9
pitfall 1 ("every bug in mapping traces back to coordinate frames") in its most
literal form, and it is why the earlier `path_to_action` heading fix looked
correct in isolation yet the eval still produced nothing.

**Fix.** `quaternion_from_coeffs()` in `goat_dataset.py`, documenting the (x, y,
z, w) order, used by `runner.py`. `EpisodeSpec.start_rotation`'s docstring now
states the order and warns against `from_float_array`.

**Result** on the same episode and step budget:

| | before | after |
|---|---|---|
| explored cells after 12 steps | 0 | 15218 |
| frontiers found | 0 | 12 |
| agent moves | no | yes |

Regression tests in `tests/test_goat_dataset.py` assert the fixed reading keeps
"up" up **and** that `from_float_array` on the same coefficients flips it -- the
failure mode itself is pinned, not just the happy path.

#### 6.17 Camera pose vs agent pose: the map was built from the ceiling

With the upside-down spawn fixed the agent moved, but only briefly, then stalled
with `explored` frozen. The map was badly wrong in a second way.

**Symptom.** Occupancy composition was inverted: **25829 occupied cells vs 8705
free** in a scene with only 75.4 m^2 (~30000 cells) of navigable area. A* then
had nowhere to path, so 38 detected frontiers were all unreachable.

**Cause.** `HabitatEnv.get_pose()` returned the **agent base** pose, but that
pose is used to back-project depth, and depth is measured from the **camera**,
mounted 1.41 m higher:

```
agent_y (floor)      = -0.536
pose translation y   = -0.536   <-- base, not camera
camera y should be   =  0.874
```

Every mapped point therefore sat 1.41 m too low. Combined with the second
problem -- `floor_height` defaulting to **0.0** while HM3D floors sit at
arbitrary world Y (-0.536 here) -- the height band `[floor+0.1, floor+1.5]`
ended up sampling roughly 1.5-2.9 m above the real floor: **the ceiling**. A
ceiling projects onto the entire floor plan, so nearly every cell read as an
obstacle.

**Fix.**
1. `get_pose()` now returns the sensor pose, preferring habitat's own
   `state.sensor_states["rgb"]` and falling back to the mount offset. It also
   feeds `bbox_center_depth_to_world_xyz`, so instance positions were wrong by
   the same 1.41 m.
2. `SemanticMap` gained an optional `camera_height`; when set, the floor is
   derived per update as `pose[1,3] - camera_height` instead of a fixed
   absolute `floor_height`. `build_agent()` passes 1.41. Existing callers that
   rely on the absolute band are unaffected.
3. Added `HabitatEnv.get_floor_height()`.

**Result** (single frame, then 16 agent steps):

| | before | after |
|---|---|---|
| occupied vs free | 25829 / 8705 | 7112 / 4706 |
| agent moves | no | yes |
| frontiers | 38 unreachable | 26, 24 reachable |
| detections mapped | 0 | 9 |

#### 6.18 What was NOT broken

Worth recording, because each was suspected and cleared with a direct
measurement rather than by reading code:

- **`path_to_action` heading convention** (fixed earlier in 2f7693d) is
  correct. Predicted forward `(-sin h, -cos h)` matched the simulator's actual
  forward on **12/12** headings; `turn_left` = +30 deg, `turn_right` = -30 deg
  as assumed; and a closed-loop drive to a point 2.00 m away finished 0.22 m
  off.
- **A\*** works: 111 waypoints to the chosen frontier, 24/26 frontiers
  reachable once the map was sane.
- **Actuation and navmesh**: 10 forward steps = 2.495 m, `get_gps()` tracks it,
  motion in all 8 headings, navmesh loaded, start poses navigable.
- **Depth units** are metres.

The lesson for the report: three separate defects (upside-down spawn, camera-vs-
base pose, absolute floor height) all presented as the single symptom "agent
does not move and SR is 0", and the two components most suspected -- the
detector and the heading convention -- were innocent.

#### 6.19 curl `--retry` silently discarded 13 GB

**Symptom.** `hm3d-train-habitat-v0.2.tar` reached 15 GB, then later showed
**2.0 GB** and a progress bar at 8%.

**Cause.** `curl -C - --retry 1000` computes the resume offset **once**, when
the transfer starts. A retry firing mid-transfer reopened the output file and
restarted from byte 0, truncating the partial download. The `--retry` and
`-C -` flags do not compose the way the flag names suggest.

**Fix.** Drop curl's internal `--retry` and drive retries from an outer shell
loop, so every attempt is a fresh `curl -C -` that re-reads the on-disk size.
The loop also warns if the file ever shrinks (a server ignoring `Range`).

Second gotcha in the same fix: the Matterport CDN does not return
`Content-Length` on a `HEAD` through its redirect, so completion cannot be
detected by comparing sizes. Completion is now taken from curl's exit status
(0 = finished; 22 with no new bytes = 416 Range Not Satisfiable = already
complete), with a `Range: bytes=0-0` probe reading the total from
`Content-Range` where available.

#### 6.20 Detector results (`checkpoints/yolo_goat.pt`)

Rendered dataset: **5971 images / 23177 boxes** from all **145/145** annotated
HM3D train scenes (5352 train + 619 val over 14 held-out scenes). No val_unseen
scene appears anywhere in it.

Trained YOLOv8n, imgsz 512, batch 24. Peak VRAM 2.2 GB of 4 GB. Two practical
notes: ultralytics reported `Slow image access (11.3 MB/s)` while the dataset
sat on `/mnt/d`, so it was copied to WSL's ext4 (2809 MB/s) -- training was
otherwise I/O-bound, not GPU-bound. And `Remapped 4/36 cls head rows from
pretrained weights by class name`, so four categories inherit COCO
initialisations.

Training reached **epoch 39 of 60** before the process was killed (the laptop
slept; the `time` column jumps from 2940 s at epoch 33 to 30053 s at epoch 34).
Weights had already been written and stripped, so `best.pt` is intact.
**Caveat for the report:** the LR schedule was set for 60 epochs and never
annealed, so these numbers understate what a completed run would give.

**Overall: mAP50 = 0.149, mAP50-95 = 0.099.** Best epoch 38.

Aggregate mAP over 36 heavily imbalanced classes is the wrong lens for
navigation -- what matters is recall on categories that actually appear as
goals. Per class, performance tracks training-box count almost monotonically:

| class | train boxes | R | mAP50 |
|---|---|---|---|
| picture | 4171 | 0.53 | 0.51 |
| pillow | 4653 | 0.47 | 0.34 |
| microwave | 268 | 0.53 | 0.57 |
| refrigerator | 404 | 0.50 | 0.32 |
| mirror | 1188 | 0.35 | 0.35 |
| plant | 981 | 0.33 | 0.15 |
| nightstand | 202 | 0.31 | 0.42 |
| book | 1624 | 0.17 | 0.19 |
| ... | | | |
| piano / statue / boiler / calendar / footrest | 40-62 | **0.00** | <0.05 |
| christmas tree | 0 | 0.00 | 0.00 |

**Only 7 of 36 classes reach recall >= 0.30.** Twelve have zero recall; those
rows typically show `P=1.000, R=0.000`, i.e. the model emits a single confident
box and misses everything else -- classic long-tail collapse. Four classes
(`hanging clothes`, `exercise bike`, `freezer`, `shower glass`) have no val
instances at all, so they cannot even be measured.

This is a **data-availability** ceiling in HM3D-Semantics, not a modelling
failure, and the report should say so explicitly. Anything built on top of this
detector will do well on `picture`/`pillow`/`mirror`/`refrigerator` subtasks and
poorly on the tail, independent of the navigation policy.

#### 6.21 Remaining blocker: the occupancy map disagrees with the navmesh

With the finetuned detector installed, a 2-episode smoke test still gave
**SR = 0%**, but the failure is now clearly in mapping, not perception.

Action histograms per subtask:

```
subtask 0  {fwd: 77, left: 23, right: 20}   moved 1.07 m
subtask 1  {fwd: 87, left: 17, right: 16}   moved 0.01 m
subtask 2  {left: 120}                      moved 0.00 m   APPROACHING<->SEARCHING
```

The agent commands forward constantly and does not move. Measured directly:

| measurement | value |
|---|---|
| forward actions blocked (no motion) | **31 / 49** |
| habitat-reported collisions | 0 |
| map says FREE and navmesh agrees navigable | **37%** |
| map says OCCUPIED yet navmesh says navigable | **45%** |

So the occupancy grid is close to uncorrelated with real navigability: it
invents free space to plan through and invents obstacles that block valid
routes. A* then returns paths into walls, the agent grinds against geometry,
and later subtasks spin because everything nearby reads as blocked.

**Likely dominant cause.** `update_from_depth` is a one-shot, irreversible
write: `self._occupancy[rows, cols] = 1` marks a cell occupied forever from a
*single* depth sample, and ray-clearing only writes free where a cell is not
already occupied. There is no evidence accumulation and no way to undo a
mistake, so projection error, depth noise and distant geometry (`max_depth` is
10 m) monotonically fill the grid with obstacles over hundreds of steps. The map
can only get worse the longer an episode runs, which matches subtasks 2+ being
worse than subtask 0.

**Proposed fix** (not yet applied -- this is a design change to a core module,
so per PLAN.md Section 10.7 it should be a conscious decision rather than
drift): replace the binary write with count-based or log-odds evidence --
require k hits before a cell counts as occupied, let ray-clearing supply
negative evidence, and threshold for the planner. Optionally reduce `max_depth`
for mapping, since far depth samples carry the largest projection error.

### 2026-08-12 — Five fixes between "agent cannot move" and "agent cannot succeed"

The detector existed by now (`checkpoints/yolo_goat.pt`, mAP50 0.149) but the
2-episode smoke test still gave SR 0%. Five defects were found and fixed, each
at a different stage of the pipeline. **SR is still 0% at the end of the day** --
what changed is that the failure mode moved forward through the stack each time,
and the geometry is now provably correct.

#### 6.22 Occupancy was one-shot and irreversible

`update_from_depth` set `occupancy[r, c] = 1` from a SINGLE depth pixel and
never revised it, and ray-clearing was gated on "only if not already occupied".
Every projection error was permanent, so the grid filled with phantom obstacles
and A* ran out of routes. Against the navmesh: 37% of "free" cells were
navigable, 45% of "occupied" cells were. 31 of 49 forward commands were
physically blocked.

Replaced with (a) `min_points_per_cell=3` -- a real surface fills a 5 cm cell
with many pixels, a stray sample contributes one -- and (b) clamped log-odds so
free evidence can outvote a stale obstacle. The clamp is what makes recovery
possible at all.

Result: blocked forwards 63% -> 0%, occupied cells 12057 -> 640, `no_plan`
subtasks 10/17 -> 0, median path travelled 0.00 m -> 12.39 m.

#### 6.23 The detector confidence gate blinded the agent

`conf=0.35` on a recall-limited detector. Over 40 val frames: conf 0.35 gave
detections in 20% of frames, 0.25 -> 30%, 0.15 -> 50%, 0.10 -> 65%. The agent
saw nothing in four frames out of five, so instance memory stayed empty and the
matcher had nothing to match. Lowered to 0.15; misses cost more than false
alarms because VERIFYING re-checks candidates.

Note found while doing this: **`build_agent()` never reads
`configs/agent_default.yaml`.** The code defaults are what run at eval time.

#### 6.24 The matcher chased the most confident instance, not the nearest

For category and image goals any instance of the category counts, but
`_match_category` returned `max(candidates, key=confidence)`. The agent walked
past the object it was standing beside -- 0.98 m away, inside the success
radius -- to chase a better-scored one across the scene. `match()` now takes the
agent's XZ and selects on distance.

Final distance to goal, same episode: language/dresser 4.33 -> 0.60 m, another
language/dresser 6.51 -> 0.87 m.

#### 6.25 Language goals could never be verified

`_goal_in_view` compared a detection's `cls_name` against `goal.value`. For a
language goal that value is the whole description ("bathroom mirror. start by
locating the sink..."), which never equals a class name. VERIFYING therefore
always failed, the FSM bounced back to APPROACHING and looped to timeout **while
standing on the target**. That is 856 of 2669 val_unseen subtasks (32%)
structurally unable to succeed. Now compares against the matched instance's
class.

#### 6.26 Depth back-projection used the wrong camera convention

The big one. `depth_to_pointcloud_camera` and `bbox_center_depth_to_world_xyz`
built OpenCV points (Y down, Z = +depth), but the rotation applied to them comes
from the habitat sensor quaternion, which is OpenGL (Y up, camera looks down
-Z). Mixing them mirrors the cloud through the camera.

Measured with the agent upright, fraction of reconstructed points landing in
front of the agent along its heading:

| convention | in front |
|---|---|
| OpenCV (Y down, Z = +d) -- what the code did | **0.0%** |
| OpenGL (Y up, Z = -d) -- habitat's actual | **100.0%** |

Every occupancy cell and every memory node was displaced. Instance nodes for a
goal category sat ~3 m from the true object, which is why the agent could stand
0.60 m from a dresser, believe its target was metres away, and never enter
VERIFYING.

Map quality across the day's three mapping fixes -- cells the map calls occupied
that are actually navigable per the navmesh:

| | |
|---|---|
| original | 45% |
| after the occupancy rewrite (6.22) | 26% |
| after the convention fix | **0%** |

Three tests asserted `z = +depth` and had been encoding the bug; they now assert
`z = -depth`. Worth remembering that a green suite only pins whatever convention
the tests were written against.

#### 6.27 Open: navigation regressed after the geometry was corrected

Honest status. With the mirrored map, obstacles were placed *behind* the agent,
so the space ahead read as free and the agent wandered unobstructed by accident.
With geometry correct, obstacles sit where they really are and blocked forwards
rose 0% -> 41%, with 4 subtasks back to `no_plan`.

Ruled out so far:
- height band is healthy (floor near 0.0, walls to +3.09 above floor, 37-68%
  of each frame in-band)
- A* already inflates 4 cells (0.20 m) against a 0.17 m agent radius
- obstacles are correct (0% of occupied cells navigable)

Still suspect, in order: the occupancy log-odds parameters were tuned against
the *mirrored* map and likely need retuning now that geometry is right; and
`_follow_path` uses a 3-cell (0.15 m) lookahead against a 0.25 m step, so the
agent may overshoot waypoints and drift into walls. Neither is measured yet.

Also unresolved: `map says FREE` agreement is only 42.8%, i.e. the map is
over-generous about free space near walls, which is consistent with the
overshoot theory.

### 2026-08-17 — The visualiser, and the failure moving into perception

#### 6.28 Build the visualiser first

`src/utils/viz.py` and `scripts/make_video.py` were empty stubs for the whole
project. Filling the first one (plus `scripts/debug_episode.py`) found two
bugs in two frames that fourteen rounds of printing numbers had not.

The view shows, side by side: the occupancy grid, the current plan, the agent's
real track, every remembered instance with the matched one highlighted, the
**ground-truth** goal positions and their success radii (which the agent never
sees), the first-person view with detector boxes, and the live FSM state.

#### 6.29 Wall-mounted goals were unreachable by construction

**Symptom.** One frame: the agent in open floor, state APPROACHING, **path
length 0**, turning in place 2.72 m from a picture whose instance it had
matched correctly. The matched node sat exactly on a ground-truth cross.

**Cause.** Pictures, mirrors, window glass and radiators hang on or against
walls, so the instance's estimated position lands on the wall. That cell reads
OCCUPIED and `plan_astar` refuses:

```python
if occupied[gr, gc]:
    return None
```

Every wall-mounted category — most of the vocabulary — could never be
approached. It also explains why every success all session was a dresser:
dressers stand proud of the wall.

**Fix.** `_approach_cell()` plans to the nearest known-free cell within
`success_distance` of the object instead of to the object. That is what the
success criterion asks for: be within 1 m of it, not standing on it.

#### 6.30 Open: the agent now stops confidently at the wrong object

With navigation working, the failure moved into perception. A traced subtask:

```
dist to TRUE goal  12.17 m
dist to node        0.70 m
state  DONE     action STOP
```

The agent matched a false instance, navigated to it correctly, verified it and
stopped — 12 m from the real microwave. Navigation, planning and control all
behaved; the target was wrong.

This is the direct cost of lowering detector `conf` from 0.35 to 0.15 (6.23).
That fix was necessary — at 0.35 the agent saw nothing in 4 frames out of 5 —
but it admits false positives into instance memory. Worse, a *consistent* false
positive also defeats VERIFYING: the node carries the detector's label, so when
the same wrong detection reappears on arrival, verification passes.

Options, none yet chosen:

1. **Require evidence before a node is targetable** — mirror the occupancy
   log-odds idea in `InstanceDatabase`: a node seen once at conf 0.15 should not
   be a navigation goal. Requires N observations or an aggregate confidence.
   Principled, and does not sacrifice recall.
2. **Two thresholds** — a low `conf` for mapping//memory and a higher one for
   accepting a node as a goal.
3. **Finish the 60-epoch detector run** (stopped at 39/60 when the laptop slept;
   the LR schedule never annealed) and re-measure precision.

Current state on the 2-episode smoke test: median distance to goal 3.31 m, best
0.68 m, SR still 0%. Navigation metrics are good — blocked forwards 0%,
map-vs-navmesh obstacle error 0%. **The binding constraint is now detector
precision, not navigation.**

#### Categories at risk (for the report's error analysis)

From a 12-scene val census, these GOAT categories had **zero** instances even
after aliasing: `exercise bike`, `photo mount`, `shower glass`. Several more are
very rare — `boiler`, `calendar`, `christmas tree`, `footrest`, `freezer` and
`photo` had one instance each. Expect poor detector recall on these and say so
explicitly in the write-up, rather than letting it read as a general failure.
Per-category counts are saved to `logs/category_census.json`.
