#!/usr/bin/env python3
"""Full val_unseen evaluation — the one final run (PLAN.md Sections 7.11, 8).

This is deliberately a thin wrapper over the development harness rather than a
second implementation: a separate code path for the headline number is how the
headline number ends up being produced by code nobody tested. Everything here
is the same loop, the same metrics and the same artifacts as
`scripts/run_dev_eval.py`, with the subsetting turned off.

`val_unseen` in full is 360 episodes and 2,669 subtasks. On the development
subset this machine tracks roughly 1.7 min per subtask, which puts the full
split near 75 hours locally -- so this is intended for the Kaggle P100 run,
with PLAN.md Section 7.11's 100-episode mid-scale run as the documented local
fallback. `--resume` exists because a run of that length will be interrupted.

    # the real thing (Kaggle, or a very patient machine)
    python -u scripts/run_full_eval.py --output-dir outputs/full_eval

    # mid-scale fallback: 100 episodes, same harness, same artifacts
    python -u scripts/run_full_eval.py --n-episodes 100 --resume \
        --output-dir outputs/mid_eval

Before using any of it, read PLAN.md Section 8: this split is to be run ONCE,
on a configuration already locked by the development evaluation and the
ablations. Running it to compare configurations turns the held-out split into a
development target and the number stops meaning anything.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_dev_eval import main

if __name__ == "__main__":
    if not any(a.startswith("--n-episodes") for a in sys.argv[1:]):
        # Default to the whole split; run_dev_eval reads n_episodes <= 0 as "all".
        sys.argv.extend(["--n-episodes", "0"])
    if not any(a.startswith("--output-dir") for a in sys.argv[1:]):
        sys.argv.extend(["--output-dir", "outputs/full_eval"])
    print(__doc__.strip().splitlines()[0])
    main()
