#!/usr/bin/env bash
# Download the HM3D train scenes + semantics needed to rebuild the detector
# dataset. PLAN.md Section 4 reserved this filename; this is it.
#
# CREDENTIALS NEVER GO ON THE COMMAND LINE OR INTO A LOG. Export them first:
#
#     export HM3D_USERNAME='<your matterport/hm3d token user>'
#     export HM3D_PASSWORD='<your matterport/hm3d token pass>'
#     bash scripts/setup_hm3d.sh --train
#
# Get the token from https://matterport.com/partners/facebook (PLAN.md 5.6).
# Note: habitat's downloader takes them as argv, so they are briefly visible in
# `ps` to other users on this machine. On a single-user box that is fine; on a
# shared one, run it when you are the only session.
#
# Three uids are needed and the third is the one that is easy to miss:
#   hm3d_train_habitat_v0.2         the .basis.glb scenes (~20 GB)
#   hm3d_train_semantic_annots_v0.2 the .semantic.glb / .semantic.txt pairs
#   hm3d_train_semantic_configs_v0.2  the ANNOTATED scene_dataset_config
#
# Without the configs tar, habitat treats each .basis.glb as a bare stage, never
# attaches the semantics, and `sim.semantic_scene.objects` is empty -- measured
# as 0 semantic objects against 661 with it (engineering log 6.18). The render
# then writes a complete-looking dataset with no boxes in it.
#
# Usage:
#   bash scripts/setup_hm3d.sh --train        # download, then verify
#   bash scripts/setup_hm3d.sh --verify-only  # check what is already on disk
#   bash scripts/setup_hm3d.sh --list         # show hm3d uids this build offers

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Resolve an interpreter that can actually import habitat_sim. `python` is not
# on PATH outside an activated conda env, and the error it gives
# ("command not found") says nothing about the real requirement.
PY="${PY:-}"
if [[ -z "$PY" ]]; then
    for cand in python python3 "$HOME/miniconda3/envs/goat/bin/python" \
                "$HOME/anaconda3/envs/goat/bin/python"; do
        if command -v "$cand" >/dev/null 2>&1 && \
           "$cand" -c "import habitat_sim" >/dev/null 2>&1; then
            PY="$cand"; break
        fi
    done
fi
if [[ -z "$PY" ]]; then
    echo "  FAIL  no interpreter found that can import habitat_sim." >&2
    echo "        conda activate goat, or set PY=/path/to/python" >&2
    exit 2
fi
DATA_PATH="$REPO_ROOT/data"
MODE=""
RETRIES="${RETRIES:-6}"

UIDS=(hm3d_train_habitat_v0.2
      hm3d_train_semantic_annots_v0.2
      hm3d_train_semantic_configs_v0.2)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)       MODE="train"; shift ;;
        --verify-only) MODE="verify"; shift ;;
        --list)        MODE="list"; shift ;;
        --data-path)   DATA_PATH="$2"; shift 2 ;;
        -h|--help)     sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done
[[ -z "$MODE" ]] && { sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 1; }

say()  { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  ok    %s\n' "$*"; }
bad()  { printf '  FAIL  %s\n' "$*"; }
warn() { printf '  warn  %s\n' "$*"; }

cd "$REPO_ROOT" || exit 1

if [[ "$MODE" == "list" ]]; then
    $PY -m habitat_sim.utils.datasets_download --list 2>&1 | grep -i hm3d
    exit 0
fi

TRAIN_ROOT="$DATA_PATH/scene_datasets/hm3d/train"

verify() {
    say "verifying $TRAIN_ROOT"
    local failed=0
    if [[ ! -d "$TRAIN_ROOT" ]]; then
        bad "train root missing: $TRAIN_ROOT"
        return 1
    fi

    local n_glb n_sem n_txt
    n_glb=$(find -L "$TRAIN_ROOT" -maxdepth 2 -name '*.basis.glb' | wc -l)
    n_sem=$(find -L "$TRAIN_ROOT" -maxdepth 2 -name '*.semantic.glb' | wc -l)
    n_txt=$(find -L "$TRAIN_ROOT" -maxdepth 2 -name '*.semantic.txt' | wc -l)
    printf '  %-28s %s\n' "basis.glb scenes:" "$n_glb"
    printf '  %-28s %s\n' "semantic.glb:" "$n_sem"
    printf '  %-28s %s\n' "semantic.txt:" "$n_txt"

    # 145 of HM3D's train scenes carry HM3D-Semantics annotations, and that is
    # the number that matters: make_yolo_dataset.py's find_scenes() only returns
    # scenes that have BOTH files. The previous detector used all 145/145 and
    # rendered 5971 images / 23177 boxes (6.20).
    if [[ "$n_sem" -ge 145 ]]; then
        ok "annotated scenes: $n_sem (>= the 145 the last build used)"
    elif [[ "$n_sem" -gt 0 ]]; then
        warn "only $n_sem annotated scenes; the last build used 145"
    else
        bad "no .semantic.glb -- the render would produce zero boxes"
        failed=1
    fi
    [[ "$n_txt" -lt "$n_sem" ]] && { bad "fewer .semantic.txt than .semantic.glb"; failed=1; }

    # The annotated scene-dataset config. 6.18: without this, zero boxes.
    local cfg
    cfg=$(find -L "$DATA_PATH/scene_datasets/hm3d" -maxdepth 2 \
          -name 'hm3d_annotated*basis.scene_dataset_config.json' | head -1)
    if [[ -n "$cfg" ]]; then
        ok "annotated config: ${cfg#$REPO_ROOT/}"
    else
        bad "no hm3d_annotated_*_basis.scene_dataset_config.json"
        echo "        This is the 6.18 failure: habitat then treats each glb as"
        echo "        a bare stage, attaches no semantics, and every rendered"
        echo "        frame yields zero boxes. Download"
        echo "        hm3d_train_semantic_configs_v0.2."
        failed=1
    fi

    # Split separation (6.14): no val_unseen scene may appear in train.
    local overlap
    overlap=$(comm -12 \
        <(find -L "$TRAIN_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sed 's/^[0-9]*-//' | sort -u) \
        <(find -L "$DATA_PATH/scene_datasets/hm3d/val" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' 2>/dev/null | sed 's/^[0-9]*-//' | sort -u) \
        | head -5)
    if [[ -n "$overlap" ]]; then
        bad "train/val scene overlap -- this would leak the eval split:"
        echo "$overlap" | sed 's/^/        /'
        failed=1
    else
        ok "no train/val scene overlap"
    fi

    return $failed
}

if [[ "$MODE" == "verify" ]]; then
    verify; exit $?
fi

# ── download ──
if [[ -z "${HM3D_USERNAME:-}" || -z "${HM3D_PASSWORD:-}" ]]; then
    bad "HM3D_USERNAME / HM3D_PASSWORD are not set in the environment"
    echo "        export them first so the token stays out of shell history"
    echo "        and out of this script's arguments:"
    echo
    echo "          read -rs -p 'hm3d user: ' HM3D_USERNAME; export HM3D_USERNAME"
    echo "          read -rs -p 'hm3d pass: ' HM3D_PASSWORD; export HM3D_PASSWORD"
    exit 2
fi

df -h "$DATA_PATH" | tail -1 | awk '{print "  disk available on data/: " $4}'
say "downloading ${#UIDS[@]} uids into $DATA_PATH"
echo "  roughly 20 GB for the scenes; the two semantic tars are much smaller."

for uid in "${UIDS[@]}"; do
    say "$uid"
    attempt=1
    while (( attempt <= RETRIES )); do
        # One uid per invocation, so a failure costs one tar and not the set.
        # Retrying the WHOLE invocation is deliberate: 6.19 records curl's
        # -C - computing its resume offset once, so a retry firing mid-transfer
        # restarted at byte 0 and silently discarded 13 GB. Re-entering the
        # downloader recomputes what is already on disk each time instead.
        if $PY -m habitat_sim.utils.datasets_download \
                --username "$HM3D_USERNAME" --password "$HM3D_PASSWORD" \
                --uids "$uid" --data-path "$DATA_PATH"; then
            ok "$uid"
            break
        fi
        warn "attempt $attempt/$RETRIES failed for $uid"
        (( attempt++ ))
        sleep $(( attempt * 10 ))
    done
    if (( attempt > RETRIES )); then
        bad "gave up on $uid after $RETRIES attempts"
        echo "        Re-run this script; completed uids are skipped cheaply."
        exit 1
    fi
done

if verify; then
    say "next"
    echo "  1. render the dataset (about 145 scenes):"
    echo "       python scripts/make_yolo_dataset.py \\"
    echo "           --hm3d-root data/scene_datasets/hm3d/train \\"
    echo "           --out data/yolo_goat --frames-per-scene 150 --resolution 512"
    echo "     (the last build used 60 and left five goal categories at recall"
    echo "      0.00 on 40-62 boxes each; 150 attacks that directly -- 6.20)"
    echo "  2. finetune for the full schedule:"
    echo "       python scripts/finetune_yolo.py --data data/yolo_goat/data.yaml \\"
    echo "           --epochs 60 --imgsz 512 --batch 24 --device 0"
    echo "     (batch 24 peaked at 2.2 GB of 4 GB when this last ran)"
    exit 0
else
    bad "downloads finished but verification failed -- see above"
    exit 1
fi
