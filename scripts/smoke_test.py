"""
Smoke test: verify env + models load and run together within VRAM budget.

Steps:
  1. Load a Habitat env with an HM3D scene at 256x256 RGB+depth.
  2. Load YOLOv8n (default COCO weights) to CUDA in fp16.
  3. Load open_clip ViT-B/32 to CUDA in fp16.
  4. Step the env forward 100 times, calling detector + CLIP each step.
  5. Print peak VRAM usage every 10 steps.

Expected peak VRAM: ~2.6-3.0 GB on RTX 3050.
"""

import argparse
import sys

import numpy as np
import torch


def get_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def load_detector(device: str = "cuda"):
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    model.to(device)
    model.fuse()
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=256, half=True, verbose=False)
    print(f"  Detector loaded. VRAM: {get_vram_mb():.0f} MB")
    return model


def load_encoder(device: str = "cuda"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device,
    )
    model = model.half().eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print(f"  CLIP encoder loaded. VRAM: {get_vram_mb():.0f} MB")
    return model, preprocess, tokenizer


def make_habitat_env(scene_path: str):
    import habitat_sim

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = 1.41
    agent_cfg.radius = 0.17

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [256, 256]
    rgb_spec.position = [0.0, 1.41, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [256, 256]
    depth_spec.position = [0.0, 1.41, 0.0]

    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    print(f"  Habitat env loaded. Scene: {scene_path}")
    return sim


def run_with_habitat(scene_path: str, steps: int = 100):
    """Full smoke test with Habitat env."""
    print("\n[1/3] Loading Habitat environment...")
    sim = make_habitat_env(scene_path)

    print("[2/3] Loading perception models...")
    torch.cuda.reset_peak_memory_stats()
    detector = load_detector()
    clip_model, clip_preprocess, clip_tokenizer = load_encoder()

    print(f"\n[3/3] Running {steps} steps...")
    actions = ["move_forward", "turn_left", "turn_right"]

    for step in range(steps):
        obs = sim.get_sensor_observations()
        rgb = obs["rgb"][:, :, :3]

        results = detector.predict(rgb, imgsz=256, half=True, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            crops = []
            for box in boxes.xyxy[:5].cpu().numpy().astype(int):
                x1, y1, x2, y2 = box
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(clip_preprocess(
                        __import__("PIL").Image.fromarray(crop)
                    ))
            if crops:
                batch = torch.stack(crops).half().to("cuda")
                with torch.no_grad():
                    clip_model.encode_image(batch)

        action = actions[step % 3]
        sim.step(action)

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1:3d}/{steps} | "
                  f"Detections: {len(boxes):2d} | "
                  f"Peak VRAM: {get_vram_mb():.0f} MB")

    sim.close()
    peak = get_vram_mb()
    print(f"\n{'='*50}")
    print(f"DONE. Peak VRAM: {peak:.0f} MB")
    if peak > 3500:
        print("WARNING: Peak VRAM > 3.5 GB — check for fp32 leaks!")
    elif peak < 2000:
        print("Peak VRAM under 2 GB — headroom for stretch features.")
    else:
        print("VRAM within expected range (2.0-3.5 GB). Good to go.")


def run_models_only(steps: int = 100):
    """Model-only smoke test (no Habitat scene required)."""
    print("\n[1/2] Loading perception models...")
    torch.cuda.reset_peak_memory_stats()
    detector = load_detector()
    clip_model, clip_preprocess, clip_tokenizer = load_encoder()

    print(f"\n[2/2] Running {steps} steps with synthetic frames...")
    for step in range(steps):
        rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        results = detector.predict(rgb, imgsz=256, half=True, verbose=False)
        boxes = results[0].boxes

        if len(boxes) > 0:
            crops = []
            for box in boxes.xyxy[:5].cpu().numpy().astype(int):
                x1, y1, x2, y2 = box
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(clip_preprocess(
                        __import__("PIL").Image.fromarray(crop)
                    ))
            if crops:
                batch = torch.stack(crops).half().to("cuda")
                with torch.no_grad():
                    clip_model.encode_image(batch)

        if (step + 1) % 20 == 0:
            text = clip_tokenizer(["a photo of a chair", "a photo of a couch"])
            with torch.no_grad():
                clip_model.encode_text(text.to("cuda"))

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1:3d}/{steps} | "
                  f"Detections: {len(boxes):2d} | "
                  f"Peak VRAM: {get_vram_mb():.0f} MB")

    peak = get_vram_mb()
    print(f"\n{'='*50}")
    print(f"DONE. Peak VRAM: {peak:.0f} MB")
    if peak > 3500:
        print("WARNING: Peak VRAM > 3.5 GB — check for fp32 leaks!")
    elif peak < 2000:
        print("Peak VRAM under 2 GB — headroom for stretch features.")
    else:
        print("VRAM within expected range (2.0-3.5 GB). Good to go.")


def main():
    parser = argparse.ArgumentParser(description="GOAT-Lite smoke test")
    parser.add_argument("--scene", type=str, default=None,
                        help="Path to HM3D .glb scene file. "
                             "If omitted, runs models-only test.")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Cannot run smoke test.")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB")

    if args.scene:
        run_with_habitat(args.scene, args.steps)
    else:
        print("\nNo --scene provided. Running models-only smoke test.")
        print("(Re-run with --scene <path/to/scene.glb> for full test.)")
        run_models_only(args.steps)


if __name__ == "__main__":
    main()
