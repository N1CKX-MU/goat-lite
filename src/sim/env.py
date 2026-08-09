from __future__ import annotations

import os
from dataclasses import dataclass, field

import habitat_sim
import numpy as np
import quaternion as qt

from src.utils.seeds import set_seed


@dataclass
class GoalSpec:
    modality: str  # "category", "language", or "image"
    value: str
    episode_step: int = 0
    subtask_index: int = 0


@dataclass
class Observation:
    rgb: np.ndarray          # [H, W, 3] uint8
    depth: np.ndarray        # [H, W] float32, meters
    pose: np.ndarray         # [4, 4] SE(3) world frame
    compass: float           # heading in radians
    gps: np.ndarray          # [2] xz position in world frame
    current_goal: GoalSpec | None = None


# Action mapping: 0=stop, 1=forward, 2=turn_left, 3=turn_right
ACTION_MAP = {
    0: None,  # stop
    1: "move_forward",
    2: "turn_left",
    3: "turn_right",
}


def default_gpu_device_id() -> int:
    """habitat GPU device id, overridable via ``HABITAT_GPU_DEVICE_ID``.

    Must be -1 under WSL2: there is no NVIDIA EGL driver there, GL is served by
    Mesa's d3d12 backend, and no EGL device advertises a CUDA id -- so the
    default of 0 aborts with "unable to find CUDA device 0". On a normal Linux
    box with the NVIDIA driver, leave it at 0.
    """
    return int(os.environ.get("HABITAT_GPU_DEVICE_ID", "0"))


class HabitatEnv:
    def __init__(
        self,
        scene_path: str,
        resolution: tuple[int, int] = (256, 256),
        agent_height: float = 1.41,
        agent_radius: float = 0.17,
        step_size: float = 0.25,
        turn_angle: float = 30.0,
        seed: int = 42,
        gpu_device_id: int | None = None,
    ):
        set_seed(seed)
        self._resolution = resolution
        self._agent_height = agent_height

        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_path
        backend_cfg.enable_physics = False
        backend_cfg.random_seed = seed
        backend_cfg.gpu_device_id = (
            default_gpu_device_id() if gpu_device_id is None else gpu_device_id
        )

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.height = agent_height
        agent_cfg.radius = agent_radius

        # Action specs
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=step_size)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_angle)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_angle)
            ),
        }

        # RGB sensor
        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "rgb"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = list(resolution)
        rgb_spec.position = [0.0, agent_height, 0.0]

        # Depth sensor
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = list(resolution)
        depth_spec.position = [0.0, agent_height, 0.0]

        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

        # Store intrinsics from sensor spec
        self._hfov = float(rgb_spec.hfov) if hasattr(rgb_spec, 'hfov') else 90.0

        cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self._sim = habitat_sim.Simulator(cfg)
        self._agent = self._sim.get_agent(0)
        self._step_count = 0
        self._done = False
        self._current_goal: GoalSpec | None = None

    @property
    def intrinsics(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix."""
        h, w = self._resolution
        hfov_rad = np.deg2rad(self._hfov)
        fx = w / (2.0 * np.tan(hfov_rad / 2.0))
        fy = fx  # square pixels
        cx, cy = w / 2.0, h / 2.0
        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1],
        ], dtype=np.float32)

    def reset(self, episode_id: str | None = None) -> Observation:
        self._step_count = 0
        self._done = False
        # Reset agent to a random navigable point
        state = habitat_sim.AgentState()
        state.position = self._sim.pathfinder.get_random_navigable_point()
        self._agent.set_state(state)
        return self._make_obs()

    def step(self, action: int) -> tuple[Observation, bool, dict]:
        if action == 0 or self._done:
            self._done = True
            return self._make_obs(), True, {"action": "stop"}

        action_name = ACTION_MAP.get(action)
        if action_name is None:
            raise ValueError(f"Invalid action: {action}")

        self._sim.step(action_name)
        self._step_count += 1

        obs = self._make_obs()
        info = {"action": action_name, "step": self._step_count}
        return obs, False, info

    def get_pose(self) -> np.ndarray:
        """Return 4x4 SE(3) pose matrix in world frame."""
        state = self._agent.get_state()
        pos = state.position
        rot = state.rotation

        R = qt.as_rotation_matrix(rot)

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
        return T

    def get_compass(self) -> float:
        """Return agent heading in radians (0 = +Z, increases CW viewed from above)."""
        state = self._agent.get_state()
        rot = state.rotation
        # Extract yaw from quaternion (numpy-quaternion: w, x, y, z)
        w, x, y, z = rot.w, rot.x, rot.y, rot.z
        # Yaw around Y axis
        siny_cosp = 2.0 * (w * y + x * z)
        cosy_cosp = 1.0 - 2.0 * (y * y + x * x)
        heading = np.arctan2(siny_cosp, cosy_cosp)
        return float(heading)

    def get_gps(self) -> np.ndarray:
        """Return [x, z] position in world frame."""
        state = self._agent.get_state()
        pos = state.position
        return np.array([float(pos[0]), float(pos[2])], dtype=np.float64)

    def get_scene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = self._sim.pathfinder.get_bounds()
        return np.array(bounds[0]), np.array(bounds[1])

    def set_goal(self, goal: GoalSpec) -> None:
        self._current_goal = goal

    def close(self) -> None:
        self._sim.close()

    def _make_obs(self) -> Observation:
        sensor_obs = self._sim.get_sensor_observations()
        rgb = sensor_obs["rgb"][:, :, :3].copy()  # drop alpha
        depth = sensor_obs["depth"].copy()

        return Observation(
            rgb=rgb,
            depth=depth,
            pose=self.get_pose(),
            compass=self.get_compass(),
            gps=self.get_gps(),
            current_goal=self._current_goal,
        )
