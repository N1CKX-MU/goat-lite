# GOAT-Lite Technical Guide

> **What this document is.** A detailed, beginner-friendly reference for every concept, technique, and design decision in the GOAT-Lite codebase. If you or your teammate encounter a term you don't understand, look it up here. This document is updated as we build new modules.
>
> **Last updated:** Week 4 (occupancy map + frontier detection complete).

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
   - 3.8 [Seed Utility (`src/utils/seeds.py`)](#38-seed-utility)
   - 3.9 [Smoke Test (`scripts/smoke_test.py`)](#39-smoke-test)
4. [Testing Strategy](#4-testing-strategy)
5. [Glossary of Key Terms](#5-glossary-of-key-terms)

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

### 3.8 Seed Utility

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

### 3.9 Smoke Test

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

---

## 5. Glossary of Key Terms

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

### fp16 / fp32 (half-precision / single-precision)
How many bits are used to represent a floating-point number:
- **fp32** (float32): 32 bits — standard precision, ~7 decimal digits
- **fp16** (float16): 16 bits — half precision, ~3 decimal digits

fp16 uses half the memory and is faster on modern GPUs, but can cause numerical instability in some operations (like softmax on large tensors). We use fp16 for inference to fit within 4 GB VRAM, with occasional casts back to fp32 for sensitive operations.

### Frontier
In exploration robotics, a frontier is the boundary between known free space and unknown space. It's the most informative place to go — moving to a frontier will reveal new areas. Our agent navigates to the largest nearby frontier when it doesn't know where its goal is.

### FOV (Field of View)
How wide the camera's "cone of vision" is, in degrees. Our camera uses 90° horizontal FOV — it can see 45° to the left and 45° to the right of center. A wider FOV sees more but with more distortion.

### Gradient
In neural networks, gradients tell you how to adjust model weights to reduce error. During inference (using a trained model), we don't need gradients, so we use `torch.no_grad()` to skip computing them — this saves ~50% of memory.

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

### mAP (mean Average Precision)
The standard metric for object detection accuracy. mAP@0.5 means "average precision at IoU threshold 0.5." Higher is better. Our target for the finetuned YOLO is mAP@0.5 >= 0.55.

### Morphological operations
Image processing techniques that operate on shapes/structures in binary images. Common ones: **dilation** (grow regions), **erosion** (shrink regions), **opening** (erosion then dilation — removes small noise), **closing** (dilation then erosion — fills small gaps). We use dilation in frontier detection.

### NMS (Non-Maximum Suppression)
When a detector produces multiple overlapping boxes for the same object, NMS keeps only the highest-confidence one and removes the rest. The `iou=0.5` parameter means: if two boxes have IoU > 0.5, the weaker one is suppressed.

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

### Ray clearing
A technique for building occupancy maps. When a depth sensor sees an obstacle at distance D, we know that everything between the sensor and the obstacle (distance 0 to D) must be free space (otherwise the sensor couldn't see through it). We "clear" those intermediate cells by marking them as free. Implemented using Bresenham's line algorithm to efficiently find which grid cells the ray passes through.

### Quaternion
A 4-component number `(w, x, y, z)` used to represent 3D rotations. More compact than a 3x3 rotation matrix (4 numbers vs 9) and avoids **gimbal lock** (a problem where Euler angles lose a degree of freedom at certain orientations). Habitat stores rotations as quaternions internally.

### Resolution (image)
Image dimensions in pixels, e.g., 256x256. More pixels = more detail but more compute and VRAM.

### Resolution (map)
Meters per grid cell. Our default is 0.05 m (5 cm). A 24m x 24m room at 5cm resolution = a 480x480 grid.

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
