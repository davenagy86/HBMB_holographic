"""Run all AdS scripts in sequence.

This is a convenience runner. Each script saves its own figures.
"""

from __future__ import annotations

import subprocess
import sys

SCRIPTS = [
    "ads_hbmb_toy_fast.py",
    "Hydrogen_HBMB_ads_FAST_physical.py",
    "Hydrogen_HBMB_ads_FAST_superposition.py",
    "HBMB_ads_high_l_stress_test.py",
    "HBMB_ads_hkll_compare.py",
]


def main() -> int:
    for s in SCRIPTS:
        print(f"\n=== Running: {s} ===")
        r = subprocess.run([sys.executable, s])
        if r.returncode != 0:
            print(f"ERROR: {s} failed with code {r.returncode}")
            return r.returncode
    print("\nAll AdS scripts finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
