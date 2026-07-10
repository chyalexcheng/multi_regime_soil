# Multi-regime MCC + Collisional Stress Model

This folder contains a modular re-implementation of the Modified Cam-Clay (MCC)
+ dynamic stress constitutive model used to simulate triaxial shear tests.
Core physics lives in `mcc_core.py`, the servo-controlled solver lives in
`triax_solve.py`, and plotting helpers live in `mcc_plotting.py`. Three driver
scripts are provided for the three main use cases described below.

## 1. Single triaxial shear test (`cmcc_driver.py`)

Runs one drained (pressure-conserved) or undrained (volume-conserved) triaxial shear test at a given over-consolidation
ratio (OCR), and produces the full set of diagnostic plots (p-q, e-p, stress
ratio, etc.) plus an `.npz` file with the input/output data.

```bash
python cmcc_driver.py --mode drained --ocr 1
python cmcc_driver.py --mode undrained --ocr 3
```

Arguments:
- `--mode`: `drained` or `undrained` (also accepts `1`/`2`)
- `--ocr`: over-consolidation ratio, e.g. `1` (normally consolidated) or `3`
  (over-consolidated)

Outputs are saved to the current directory (`.`) by default.

## 2. Multi-rate triaxial shear study (`cmcc_diff_rates_driver.py`)

Runs the same triaxial shear path three times (by default) with different
acceleration/deceleration durations for the imposed shear-strain rate, to
study rate-dependent (dynamic) effects. Produces per-rate plots plus a
comparison plot (e.g. void ratio vs. shear rate with the rate-induced
dilatancy reference line) overlaying all runs.

```bash
python cmcc_diff_rates_driver.py --mode drained --ocr 1 \
    --accel-times 0.02 0.1 0.5 --total-time 2.5 --output-dir .
```

Arguments:
- `--mode`: `drained` or `undrained` (also accepts `1`/`2`)
- `--ocr`: over-consolidation ratio, e.g. `1` or `3`
- `--accel-times`: list of acceleration/deceleration durations [s] to compare
  (default: `0.02 0.1 0.5`)
- `--total-time`: total loading duration [s] (default: `2.5`)
- `--output-dir`: directory to save `.npz` files and figures (default: `.`)

## 3. mu(I) / phi(I) rheology comparison (`cmcc_evpc_mu_I_compare.py`)

Runs (or reuses saved results from) triaxial shear tests at several initial
pressures to build stress-ratio ($\mu$) and solid-volume-fraction ($\phi$)
curves as functions of the inertial number $I$, then compares them against a
classical $\mu(I)$/$\phi(I)$ rheology fit as well as the CMCC model's own
rate-dependent dynamic stress prediction.

```bash
python cmcc_evpc_mu_I_compare.py
```

The script is interactive and will prompt for:
- **Deformation mode**: `1` for drained (pressure-conserved).
- **Use saved data? [y/n]**:
  - `y` — reuse the previously saved `{mode}_steady_state.npy` and
    `{mode}_d_gamma_history.npy` files instead of re-running the simulations
    (fast, just re-generates the plots).
  - `n` — re-run the triaxial shear tests for all initial pressures
    (`p0 = 150, 300, 450` kPa) from scratch and overwrite the saved `.npy`
    files (slower, needed the first time or after changing parameters).

Produces plots of $\mu$ vs. $I$, $\phi$ vs. $I$, void ratio vs. shear rate,
and dynamic-stress-rate comparisons, saved as PNG files in the current
directory.
