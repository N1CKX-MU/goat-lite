"""Debug visualisation for the navigation stack.

Renders what the agent believes alongside what is actually true, so a failure
can be seen rather than inferred. Every bug found in this project so far was
found by printing numbers; a single frame showing the map, the plan, the track,
the remembered instances and the real goal answers most questions immediately.

Colours match the occupancy semantics used throughout: free / occupied /
unknown, with the goal and the agent picked out against them.
"""

from __future__ import annotations

import numpy as np

# BGR, because OpenCV. Kept muted so the overlays read clearly on top.
C_UNKNOWN = (58, 55, 52)
C_FREE = (105, 140, 120)
C_OCCUPIED = (60, 90, 190)
C_AGENT = (255, 255, 255)
C_HEADING = (255, 235, 200)
C_TRACK = (200, 190, 120)
C_PATH = (90, 220, 255)
C_NODE = (190, 150, 255)
C_NODE_MATCH = (120, 255, 160)
C_GOAL = (70, 110, 255)
C_TEXT = (240, 240, 240)
# A node whose class equals the goal category: a candidate the matcher could
# have picked. Distinguishing these from the rest is what makes a wrong match
# legible -- otherwise every remembered instance is an anonymous dot.
C_NODE_CANDIDATE = (110, 200, 255)


def occupancy_to_bgr(occupancy: np.ndarray) -> np.ndarray:
    """Colour the tri-state grid. Returns [H, W, 3] uint8 BGR."""
    img = np.zeros((*occupancy.shape, 3), dtype=np.uint8)
    img[occupancy == -1] = C_UNKNOWN
    img[occupancy == 0] = C_FREE
    img[occupancy == 1] = C_OCCUPIED
    return img


def render_topdown(
    semantic_map,
    agent_xy,
    heading: float,
    *,
    path=None,
    track=None,
    nodes=None,
    matched_node=None,
    goal_positions=None,
    goal_category: str | None = None,
    label_candidates: bool = True,
    success_radius: float = 1.0,
    out_size: int = 520,
    margin_cells: int = 24,
):
    """Top-down debug view of the agent's world model.

    Args:
        semantic_map: the SemanticMap.
        agent_xy: world (x, z) of the agent.
        heading: compass heading in radians.
        path: list of (row, col) grid cells currently planned.
        track: list of world (x, z) the agent has actually visited.
        nodes: iterable of memory nodes (need ``.world_xyz``).
        matched_node: the node the agent is currently targeting, if any.
        goal_positions: ground-truth [x, y, z] of acceptable goal instances.
        success_radius: metres; drawn around each ground-truth goal.

    Returns [out_size, out_size, 3] uint8 BGR, cropped to the explored region.
    """
    import cv2

    occ = semantic_map.get_occupancy()
    img = occupancy_to_bgr(occ)
    res = semantic_map._resolution

    def to_cell(xz):
        return semantic_map.world_to_grid(np.asarray(xz, dtype=np.float64))

    # ── crop to what has actually been seen, else the view is mostly empty ──
    seen = np.argwhere(occ != -1)
    if len(seen) > 0:
        r0, c0 = seen.min(0) - margin_cells
        r1, c1 = seen.max(0) + margin_cells
    else:
        r0, c0, r1, c1 = 0, 0, occ.shape[0], occ.shape[1]
    ar, ac = to_cell(agent_xy)
    r0, c0 = min(r0, ar - margin_cells), min(c0, ac - margin_cells)
    r1, c1 = max(r1, ar + margin_cells), max(c1, ac + margin_cells)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1 = min(occ.shape[0], int(r1)); c1 = min(occ.shape[1], int(c1))

    # square the crop so nothing is distorted by the resize
    h, w = r1 - r0, c1 - c0
    if h > w:
        pad = (h - w) // 2
        c0, c1 = max(0, c0 - pad), min(occ.shape[1], c1 + pad)
    elif w > h:
        pad = (w - h) // 2
        r0, r1 = max(0, r0 - pad), min(occ.shape[0], r1 + pad)

    view = img[r0:r1, c0:c1]
    if view.size == 0:
        view = img
        r0 = c0 = 0
    scale = out_size / max(view.shape[0], view.shape[1])

    def to_px(xz):
        r, c = to_cell(xz)
        return int((c - c0) * scale), int((r - r0) * scale)

    def cell_to_px(rc):
        r, c = rc
        return int((c - c0) * scale), int((r - r0) * scale)

    canvas = cv2.resize(
        view, (int(view.shape[1] * scale), int(view.shape[0] * scale)),
        interpolation=cv2.INTER_NEAREST,
    )

    # ── where the agent has actually been ───────────────────────────────
    if track:
        pts = [to_px(p) for p in track]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, C_TRACK, 1, cv2.LINE_AA)

    # ── the plan it is currently following ──────────────────────────────
    if path:
        pts = [cell_to_px(p) for p in path]
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, a, b, C_PATH, 2, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, pts[-1], 5, C_PATH, -1, cv2.LINE_AA)

    # ── what it remembers seeing ────────────────────────────────────────
    # Nodes of the goal's own class are drawn distinctly and labelled: a wrong
    # match is only diagnosable if you can see which candidates were available
    # and which one was taken.
    if nodes:
        for n in nodes:
            p = to_px(np.asarray(n.world_xyz)[[0, 2]])
            is_match = matched_node is not None and n is matched_node
            is_candidate = (
                goal_category is not None
                and getattr(n, "cls_name", None) == goal_category
            )
            if is_match:
                colour, radius = C_NODE_MATCH, 5
            elif is_candidate:
                colour, radius = C_NODE_CANDIDATE, 4
            else:
                colour, radius = C_NODE, 2
            cv2.circle(canvas, p, radius, colour, -1, cv2.LINE_AA)
            if is_match:
                cv2.circle(canvas, p, 10, C_NODE_MATCH, 1, cv2.LINE_AA)
            if label_candidates and (is_candidate or is_match):
                conf = getattr(n, "confidence", None)
                txt = getattr(n, "cls_name", "?")
                if conf is not None:
                    txt = f"{txt} {conf:.2f}"
                cv2.putText(canvas, txt, (p[0] + 7, p[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, colour, 1, cv2.LINE_AA)

    # ── the truth, which the agent never sees ───────────────────────────
    if goal_positions is not None:
        for g in goal_positions:
            g = np.asarray(g, dtype=np.float64)
            p = to_px((g[0], g[2]))
            cv2.circle(canvas, p, int(success_radius / res * scale),
                       C_GOAL, 1, cv2.LINE_AA)
            cv2.drawMarker(canvas, p, C_GOAL, cv2.MARKER_CROSS, 13, 2)

    # ── the agent ───────────────────────────────────────────────────────
    ap = to_px(agent_xy)
    fwd = np.array([-np.sin(heading), -np.cos(heading)])
    tip = to_px(np.asarray(agent_xy) + fwd * 0.6)
    cv2.line(canvas, ap, tip, C_HEADING, 2, cv2.LINE_AA)
    cv2.circle(canvas, ap, 5, C_AGENT, -1, cv2.LINE_AA)

    return canvas


def draw_detections(rgb: np.ndarray, detections) -> np.ndarray:
    """First-person view with detector boxes. Takes RGB, returns BGR."""
    import cv2

    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in detections or []:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), C_NODE_MATCH, 1)
        label = f"{d.cls_name} {d.conf:.2f}"
        cv2.putText(img, label, (x1, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_NODE_MATCH, 1, cv2.LINE_AA)
    return img


def compose_frame(topdown, fpv, lines, width_pad: int = 300):
    """Top-down + first-person + a text column of live state."""
    import cv2

    h = topdown.shape[0]
    fpv_r = cv2.resize(fpv, (h // 2, h // 2), interpolation=cv2.INTER_NEAREST)

    panel = np.full((h, width_pad, 3), 28, dtype=np.uint8)
    panel[: fpv_r.shape[0], : fpv_r.shape[1]] = fpv_r
    y = fpv_r.shape[0] + 24
    for line in lines:
        cv2.putText(panel, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, C_TEXT, 1, cv2.LINE_AA)
        y += 20

    return np.hstack([topdown, panel])


def legend_lines() -> list[str]:
    return [
        "white dot  agent",
        "cyan line  planned path",
        "olive      actual track",
        "green      MATCHED node",
        "blue       same-class candidate",
        "violet     other memory nodes",
        "red cross  TRUE goal",
        "red circle 1.0 m success",
    ]
