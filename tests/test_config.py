"""configs/agent_default.yaml must mirror the defaults that actually run.

`build_agent()` does not read the YAML -- constructor defaults in `src/` are the
effective configuration -- so the YAML is documentation. Documentation that
nothing checks goes stale silently, and this one had: it was missing
`min_observations`, `trust_confidence`, `image_threshold`, every verification
constant and the whole occupancy evidence scheme, while still reading as though
it described the running system. CLAUDE.md lists that split as a known trap.

These tests close it. Each value in the YAML is compared against the code it
claims to document, read out of the live signature rather than copied here, so
changing one side without the other fails. When that happens the fix is to
update the YAML, not to loosen the test: the YAML is also the report's
hyperparameter appendix, and the sweep in PLAN.md Section 7.10 ends by writing
the winning configuration into it.
"""

from __future__ import annotations

import inspect
import math
import os

import pytest
import yaml

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "agent_default.yaml",
)


@pytest.fixture(scope="module")
def cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _defaults(fn):
    """Declared default of every keyword parameter of a callable."""
    return {
        name: p.default
        for name, p in inspect.signature(fn).parameters.items()
        if p.default is not inspect.Parameter.empty
    }


class TestMatcher:
    def test_matches_goal_matcher_defaults(self, cfg):
        from src.matching.goal_matcher import GoalMatcher
        d = _defaults(GoalMatcher.__init__)
        m = cfg["matching"]
        assert m["category_threshold"] == d["category_threshold"]
        assert m["language_threshold"] == d["language_threshold"]
        assert m["use_template"] == d["use_template"]
        assert m["min_observations"] == d["min_observations"]
        assert m["trust_confidence"] == d["trust_confidence"]
        assert m["image_threshold"] == d["image_threshold"]

    def test_image_threshold_is_the_neutral_value(self, cfg):
        """-1.0 is the minimum a cosine can take. 0.0 is NOT neutral: two
        unrelated CLIP embeddings score around zero and land either side."""
        assert cfg["matching"]["image_threshold"] == -1.0


class TestMemory:
    def test_matches_instance_database_defaults(self, cfg):
        from src.memory.instance_db import InstanceDatabase
        d = _defaults(InstanceDatabase.__init__)
        assert cfg["memory"]["merge_dist_m"] == d["merge_dist_m"]
        assert cfg["memory"]["merge_embed_sim"] == d["merge_embed_sim"]


class TestMapping:
    def test_matches_semantic_map_defaults(self, cfg):
        from src.mapping.occupancy import SemanticMap
        d = _defaults(SemanticMap.__init__)
        m = cfg["mapping"]
        for key in ("size_m", "resolution_m", "num_classes", "min_height",
                    "max_height", "min_points_per_cell", "l_occ", "l_free",
                    "clamp", "occ_threshold", "free_threshold"):
            assert m[key] == d[key], key


class TestPerception:
    def test_matches_detector_defaults(self, cfg):
        from src.perception.detector import YOLODetector
        d = _defaults(YOLODetector.__init__)
        p = cfg["perception"]["detector"]
        assert p["conf"] == d["conf"]
        assert p["iou"] == d["iou"]
        assert p["imgsz"] == d["imgsz"]
        assert p["fp16"] == d["fp16"]

    def test_matches_encoder_defaults(self, cfg):
        from src.perception.encoder import ClipEncoder
        d = _defaults(ClipEncoder.__init__)
        e = cfg["perception"]["encoder"]
        assert e["model_name"] == d["model_name"]
        assert e["pretrained"] == d["pretrained"]
        assert e["fp16"] == d["fp16"]

    def test_matches_pipeline_defaults(self, cfg):
        from src.perception.pipeline import PerceptionPipeline
        d = _defaults(PerceptionPipeline.__init__)
        assert cfg["perception"]["min_crop_size"] == d["min_crop_size"]
        assert cfg["perception"]["thumbnail_size"] == d["thumbnail_size"]

    def test_detector_weights_path_is_the_finetuned_checkpoint(self, cfg):
        """Without this file the detector silently falls back to COCO
        yolov8n, whose vocabulary covers almost none of the GOAT categories."""
        assert cfg["perception"]["detector"]["weights"].endswith(
            "yolo_goat.pt")


class TestPlanner:
    def test_matches_astar_defaults(self, cfg):
        from src.planning.astar import plan_astar
        d = _defaults(plan_astar)
        p = cfg["planner"]
        assert p["inflate_cells"] == d["inflate_cells"]
        assert p["obstacle_penalty"] == d["obstacle_penalty"]
        assert p["unknown_penalty"] == d["unknown_penalty"]

    def test_inflation_covers_the_agent_radius(self, cfg):
        """Inflation is a hard constraint, so it has to exceed the 0.17 m
        agent radius at the map's resolution."""
        cells = cfg["planner"]["inflate_cells"]
        res = cfg["mapping"]["resolution_m"]
        assert cells * res > 0.17


class TestAgent:
    def test_matches_goat_agent_defaults(self, cfg):
        from src.agent.goat_agent import GoatAgent
        d = _defaults(GoatAgent.__init__)
        assert cfg["agent"]["max_steps"] == d["max_steps"]
        assert cfg["agent"]["success_distance"] == d["success_distance"]

    def test_matches_the_state_machine_constants(self, cfg):
        from src.agent import goat_agent as ga
        a = cfg["agent"]
        assert a["search_giveup_turns"] == ga.SEARCH_GIVEUP_TURNS
        assert a["approach_giveup_turns"] == ga.APPROACH_GIVEUP_TURNS
        assert a["lookahead_cells"] == ga.LOOKAHEAD_CELLS
        assert a["arrive_cells"] == ga.ARRIVE_CELLS
        assert a["final_arrive_cells"] == ga.FINAL_ARRIVE_CELLS
        assert a["replan_every_steps"] == ga.REPLAN_EVERY_STEPS
        assert a["verify_giveup_steps"] == ga.VERIFY_GIVEUP_STEPS
        assert a["commit_distance_frac"] == ga.COMMIT_DISTANCE_FRAC
        assert math.isclose(math.radians(a["verify_facing_tol_deg"]),
                            ga.VERIFY_FACING_TOL)

    def test_camera_height_matches_build_agent(self, cfg):
        from scripts.run_dev_eval import build_agent
        d = _defaults(build_agent)
        assert cfg["agent"]["camera_height"] == d["camera_height"]
        # The evidence gate is threaded through build_agent as a sweep axis, so
        # the two places that name it must agree.
        assert cfg["matching"]["min_observations"] == d["min_observations"]

    def test_lookahead_exceeds_one_forward_step(self, cfg):
        """A forward step is 0.25 m; a shorter lookahead overshoots every
        waypoint, and the arrival radius has to clear it too."""
        res = cfg["mapping"]["resolution_m"]
        step_cells = 0.25 / res
        assert cfg["agent"]["lookahead_cells"] > step_cells
        assert cfg["agent"]["arrive_cells"] > step_cells

    def test_verification_budget_allows_a_full_survey(self, cfg):
        """The old budget of 5 could not bring a goal more than 150 deg
        off-axis into frame, so a goal behind the agent was unconfirmable."""
        turns_for_360 = 360.0 / 30.0
        assert cfg["agent"]["verify_giveup_steps"] >= turns_for_360

    def test_commit_band_sits_inside_the_success_radius(self, cfg):
        frac = cfg["agent"]["commit_distance_frac"]
        assert 0.0 < frac < 1.0
        # One forward step of 0.25 m from the band's edge must not overshoot
        # into the range where the detector is blind and framing fails.
        commit_m = frac * cfg["agent"]["success_distance"]
        assert commit_m - 0.25 > 0.3
