#!/usr/bin/env bash
# Prepare a Kaggle (or any fresh) machine for the full val_unseen run.
# PLAN.md Section 7.11 step 3 names this script; this is it.
#
# WHAT THIS DOES NOT DO: download HM3D. HM3D is licensed and needs a Matterport
# account token (PLAN.md Section 5.6), so there is no URL this script could
# fetch it from, and a script that pretends otherwise fails six hours into a
# notebook. On Kaggle the working route is to attach the scenes and episodes as
# a private Kaggle dataset you uploaded once, and point this script at it.
#
# So this script does the two things that CAN be done correctly:
#   1. installs the dependency combination that actually works together, and
#   2. verifies the on-disk layout against what the loaders really require,
#      failing loudly and specifically when something is missing.
#
# The second half is the point. Every path checked here is a path that, when
# wrong, produces a *silently degraded* run rather than a crash -- a missing
# detector checkpoint falls back to COCO weights whose vocabulary covers almost
# none of the GOAT categories, and SR goes to zero for a reason no log states.
#
# Usage, inside a Kaggle notebook cell:
#     !bash scripts/setup_kaggle.sh --data-root /kaggle/input/goat-lite-data
#     !bash scripts/setup_kaggle.sh --verify-only          # check, install nothing
#
# Exit codes: 0 ready, 1 layout problem, 2 install problem.

set -uo pipefail

DATA_ROOT=""
VERIFY_ONLY=0
SKIP_INSTALL=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)    DATA_ROOT="$2"; shift 2 ;;
        --verify-only)  VERIFY_ONLY=1; SKIP_INSTALL=1; shift ;;
        --skip-install) SKIP_INSTALL=1; shift ;;
        --repo-root)    REPO_ROOT="$2"; shift 2 ;;   # for testing the FAIL paths
        -h|--help)      sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

say()  { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }

# ── 1. dependencies ──
# These versions are not arbitrary. The pins in requirements.txt do NOT install
# together on a modern stack: numpy 1.24.4 breaks habitat-sim's quaternion
# extension, and ultralytics 8.1.34 calls torch.load() in a way torch >= 2.6
# rejects. This is the combination verified to work (TRAIN_YOLO.md Section 0).
if [[ "$SKIP_INSTALL" -eq 0 ]]; then
    say "installing the verified dependency combination"
    pip install -q \
        "numpy==1.26.4" \
        "ultralytics==8.4.117" \
        "open_clip_torch==3.3.0" \
        "opencv-python-headless==4.10.0.84" \
        "pillow==10.4.0" \
        "scikit-fmm==2025.1.29" \
        "scikit-image>=0.22" \
        "scipy>=1.11" \
        || { bad "pip install failed"; exit 2; }
    ok "python packages"

    # habitat-sim is a conda package and is NOT pip-installable. On Kaggle the
    # practical options are a conda install into the notebook or a prebuilt
    # wheel attached as a dataset; neither can be assumed here, so this only
    # reports what it finds instead of guessing.
    if python -c "import habitat_sim" 2>/dev/null; then
        ok "habitat_sim importable ($(python -c 'import habitat_sim;print(habitat_sim.__version__)' 2>/dev/null))"
    else
        bad "habitat_sim NOT importable -- install it before running the eval"
        echo "        conda install -c aihabitat -c conda-forge habitat-sim=0.3.1 headless withbullet"
        echo "        (every module imports through src/sim/env.py, so even the"
        echo "         pure-logic tests will not collect without it)"
    fi
fi

# ── 2. stage data from an attached dataset ──
if [[ -n "$DATA_ROOT" ]]; then
    say "staging data from $DATA_ROOT"
    if [[ ! -d "$DATA_ROOT" ]]; then
        bad "--data-root does not exist: $DATA_ROOT"; exit 1
    fi
    mkdir -p "$REPO_ROOT/data" "$REPO_ROOT/checkpoints"
    # Kaggle input is read-only, so symlink rather than copy: HM3D val is tens
    # of GB and the notebook's writable space is limited.
    for sub in hm3d datasets; do
        if [[ -e "$DATA_ROOT/$sub" && ! -e "$REPO_ROOT/data/$sub" ]]; then
            ln -s "$DATA_ROOT/$sub" "$REPO_ROOT/data/$sub" && ok "linked data/$sub"
        fi
    done
    if [[ -f "$DATA_ROOT/yolo_goat.pt" && ! -e "$REPO_ROOT/checkpoints/yolo_goat.pt" ]]; then
        ln -s "$DATA_ROOT/yolo_goat.pt" "$REPO_ROOT/checkpoints/yolo_goat.pt" \
            && ok "linked checkpoints/yolo_goat.pt"
    fi
fi

# ── 3. verify the layout the loaders actually require ──
# Paths here mirror src/eval/runner.py::_resolve_scene_path and
# src/sim/goat_dataset.py::load_val_unseen. If either moves, this must too.
say "verifying the layout"
FAILED=0

EPISODES="$REPO_ROOT/data/datasets/goat_bench/hm3d/v1/val_unseen/content"
if [[ -d "$EPISODES" ]]; then
    N_SCENE_FILES=$(find "$EPISODES" -name '*.json.gz' | wc -l)
    if [[ "$N_SCENE_FILES" -eq 36 ]]; then
        ok "episodes: 36 scene files"
    elif [[ "$N_SCENE_FILES" -gt 0 ]]; then
        warn "episodes: $N_SCENE_FILES scene files, expected 36 for val_unseen"
    else
        bad "episodes: $EPISODES holds no .json.gz"; FAILED=1
    fi
else
    bad "episodes missing: $EPISODES"
    echo "        load_val_unseen() reads <root>/content/*.json.gz"
    FAILED=1
fi

HM3D="$REPO_ROOT/data/hm3d"
if [[ -d "$HM3D" ]]; then
    # _resolve_scene_path maps "hm3d/val//00894-HY1NcmCgn3n/HY1NcmCgn3n.basis.glb"
    # onto "data/hm3d/00894-HY1NcmCgn3n/HY1NcmCgn3n.basis.glb" -- note the scene
    # directory is NOT nested under a "val/" level. A val/ level here is the
    # single most common layout mistake and produces "scene not found" per
    # episode rather than once.
    N_GLB=$(find -L "$HM3D" -maxdepth 2 -name '*.basis.glb' | wc -l)
    if [[ "$N_GLB" -gt 0 ]]; then
        ok "scenes: $N_GLB .basis.glb files under data/hm3d/<NNNNN-HASH>/"
    else
        bad "scenes: no .basis.glb at data/hm3d/<NNNNN-HASH>/<HASH>.basis.glb"
        if find -L "$HM3D" -maxdepth 3 -name '*.basis.glb' -print -quit | grep -q .; then
            echo "        found some deeper than 2 levels -- you probably have"
            echo "        data/hm3d/val/<scene>/. _resolve_scene_path expects no"
            echo "        'val' level; move the scene directories up one."
        fi
        FAILED=1
    fi
else
    bad "scenes missing: $HM3D"; FAILED=1
fi

CKPT="$REPO_ROOT/checkpoints/yolo_goat.pt"
if [[ -f "$CKPT" ]]; then
    ok "detector: checkpoints/yolo_goat.pt present"
else
    bad "detector MISSING: checkpoints/yolo_goat.pt"
    echo "        This does NOT crash. YOLODetector falls back to COCO"
    echo "        yolov8n.pt, whose vocabulary covers almost none of the 36"
    echo "        GOAT categories; SR goes to 0 and the only hint is a"
    echo "        dropped-class-id warning per scene. Do not start the full"
    echo "        run without this file."
    FAILED=1
fi

# ── 4. prove it end to end, cheaply ──
if [[ "$FAILED" -eq 0 && "$VERIFY_ONLY" -eq 0 ]]; then
    say "parsing one episode"
    cd "$REPO_ROOT" || exit 1
    PYTHONPATH=. python - <<'PY' || FAILED=1
import sys
try:
    from src.sim.goat_dataset import load_val_unseen
    eps = load_val_unseen()
    n_sub = sum(len(e.subtasks) for e in eps)
    scenes = {e.scene_id for e in eps}
    print(f"  ok    {len(eps)} episodes, {n_sub} subtasks, {len(scenes)} scenes")
    if len(eps) != 360 or n_sub != 2669:
        print(f"  warn  expected 360 episodes / 2669 subtasks for val_unseen")
except Exception as e:
    print(f"  FAIL  could not parse episodes: {type(e).__name__}: {e}")
    sys.exit(1)
PY
fi

say "result"
if [[ "$FAILED" -eq 0 ]]; then
    echo "  Ready. The full split is ONE run (PLAN.md Section 8) -- it is not a"
    echo "  configuration to tune against."
    echo
    echo "    python -u scripts/run_full_eval.py --resume --output-dir outputs/full_eval"
    echo
    echo "  Expect 6-10 h on a P100. --resume checkpoints after every scene, so"
    echo "  a notebook timeout costs one scene, not the run."
    exit 0
else
    echo "  NOT ready -- see the FAIL lines above."
    exit 1
fi
