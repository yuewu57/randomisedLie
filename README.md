# Randomised Lie-group integrators

Reproducible numerical experiments for the working manuscript on randomised structure-preserving Lie-group integrators for ODEs with low temporal regularity.

The code implements and compares:

- randomised Lie–Euler (RLE);
- two-stage randomised Runge–Kutta–Munthe-Kaas (RRKMK2);
- two-stage randomised Lie commutator-free (RLCF2);
- paired/antithetic RRKMK2 and RLCF2;
- deterministic Lie–Euler, commutator-free midpoint, RKMK midpoint, and fourth-order RKMK controls.

The main experiments cover rough and smooth problems on `SO(3)` and `S^2`. The manufactured rough problems have explicit reference trajectories, so the low-regularity experiments do not rely on a time-stepping reference solver.

## Environment

The v4 campaign was tested with Python 3.13.5 and the pinned dependencies in:

`requirements_v4_YW_01102026.txt`

Create an environment with:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_v4_YW_01102026.txt
```

## Full experiment campaign

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python experiments/randomised_lie_verification_v4_YW_01102026.py \
  --mode all --paths 1024 --maxN 1024 --out results

python experiments/independent_checks_v4_YW_01102026.py --out results

python experiments/plot_results_v4_YW_01102026.py \
  --results results --out figures
```

Windows users may omit the two environment-variable assignments or set them in their shell.

A smaller smoke test is:

```bash
python experiments/randomised_lie_verification_v4_YW_01102026.py \
  --mode main --paths 64 --maxN 128 --out smoke_results
```

Do not mix smoke-test output with the paper campaign.

## Reproducibility notes

- Randomness is introduced by the numerical integrator, not by the ODE.
- Fixed seeds are stored by the experiment driver.
- No projection or renormalisation is applied after each Lie-group step.
- The rough reference trajectory is explicit.
- The smooth reference problem uses a refined DOP853 solve as a numerical diagnostic.
- Common-node coupled distances between RLCF2 and RRKMK2 are recorded directly.
- Generated `results/`, `smoke_results/`, and `figures/` are intentionally not version-controlled.

## Version

Current reproducibility code: **v4_YW_01102026**.

This repository is research code accompanying an evolving manuscript. Mathematical claims should be checked against the current manuscript rather than inferred solely from numerical output.
