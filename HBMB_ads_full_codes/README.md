# HBMB AdS section – full runnable codes

This folder contains all Python scripts used to generate the **AdS** toy-model results for the HBMB paper.
All figures are saved automatically.

## Contents

- `ads_hbmb_toy_fast.py`
  - AdS toy: Gaussian blob on the boundary screen `r = Rb`.
  - Projects to `c_{lm}` and reconstructs a bulk-slice at `r = r_slice` with an AdS-like radial kernel.
  - Saves: error vs cutoff, coverage vs cutoff, maps, and kernel profile plots.

- `Hydrogen_HBMB_ads_FAST_physical.py`
  - Hydrogen angular control target (default: 2p_z → `Y_{1,0}`).
  - Produces horizon maps and bulk-slice maps + error/coverage curves.

- `Hydrogen_HBMB_ads_FAST_superposition.py`
  - Complex superposition of hydrogenic angular states (phase-correctness test).
  - Saves magnitude and phase maps + error/coverage.

- `HBMB_ads_high_l_stress_test.py`
  - High-ℓ stress test with a critical mode at `(ℓ*,m*)=(21,7)`.
  - Saves **both** boundary (horizon) error and bulk-slice error vs cutoff,
    plus maps at `ℓ_max=20` and `ℓ_max=21`.
  - Annotates the plateau level on the plot.

- `HBMB_ads_hkll_compare.py`
  - Direct numerical comparison between:
    - HBMB mode-synthesis (sum over `c_{lm} f_ℓ(r) Y_{lm}`), and
    - an HKLL-style angular smearing integral using a truncated Legendre-series kernel.
  - Uses a random subset of evaluation points for speed.

- `run_all_ads.sh`
  - Convenience script to run everything in the recommended order.

- `ads_section_summary.json`
  - Optional output summary written by `run_all_ads.sh` (you can regenerate it).

## Requirements

- Python 3.10+
- numpy
- scipy
- matplotlib

## Usage

From this folder:

```bash
python ads_hbmb_toy_fast.py
python Hydrogen_HBMB_ads_FAST_physical.py
python Hydrogen_HBMB_ads_FAST_superposition.py
python HBMB_ads_high_l_stress_test.py
python HBMB_ads_hkll_compare.py
```

Or run everything:

```bash
bash run_all_ads.sh
```

Figures are saved under `figs_ads_*` subfolders.
