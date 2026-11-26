"""Run the whole pipeline: simulations, figures, reports, animations.

Usage::

    python scripts/run_all.py            # everything
    python scripts/run_all.py --no-anim  # skip the animations

Cached simulation results in ``outputs/cache`` are reused by the figure
and report scripts. Delete ``outputs/cache`` to recompute from scratch.
The compute scripts use all CPUs unless ``QG_WORKERS`` is set.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

COMPUTE = ["compute_resonance", "compute_cases", "compute_branches",
           "compute_dynamics", "compute_lagrangian", "compute_eddies",
           "compute_verification"]
FIGURES = ["fig01_model", "fig02_verification", "fig03_cases",
           "fig04_resonance", "fig05_linear_drag", "fig06_branches",
           "fig07_dynamics", "fig08_mixing", "fig09_eddies"]
REPORTS = ["make_reports"]
ANIMS = ["anim01_mountain", "anim02_vacillation", "anim03_eddies"]


def run(name):
    """Run one script in a fresh interpreter and time it."""
    t0 = time.time()
    print("== %s" % name, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    env.pop("QG_PREVIEW", None)
    subprocess.run([sys.executable, os.path.join(HERE, name + ".py")],
                   check=True, env=env)
    print("   %.0f s" % (time.time() - t0), flush=True)


def main():
    steps = COMPUTE + FIGURES + REPORTS
    if "--no-anim" not in sys.argv:
        steps += ANIMS
    for s in steps:
        run(s)


if __name__ == "__main__":
    main()
