#!/usr/bin/env python3
"""Size the P0 probe grid: how much error does P-quantization inject into softmaxSum?

Worst case for an absolute grid G is m/(2*G) added to l, so a fixed G is NOT a
fixed relative perturbation -- it depends on how many small terms a row has.
This simulates realistic score distributions and reports the actual relative
error on l = sum(exp(s - max)) for several grids.

Pure Python, no NPU needed.
"""

import math
import random
import sys

GRID_CANDIDATES = [1024, 65536, 1 << 17, 1 << 21]
ROWS = 400


def quantize(e: float, grid: int) -> float:
    return math.floor(e * grid + 0.5) / grid


def rel_err(score, grid):
    mx = max(score)
    exact = math.fsum(math.exp(s - mx) for s in score)
    approx = math.fsum(quantize(math.exp(s - mx), grid) for s in score)
    if exact <= 0.0:
        return 0.0, 0.0
    return abs(approx - exact) / exact, abs(approx - exact)


def main() -> int:
    random.seed(20260920)
    # three regimes: flat scores (all e ~ 1, big l), peaked (one dominant token,
    # l ~ 1), heavy tail (many mid-sized e)
    cases = {
        "flat  m=2048 s~U(-4,0)": lambda: [random.uniform(-4.0, 0.0) for _ in range(2048)],
        "peak  m=2048 exp decay": lambda: [0.0] + [-random.uniform(0.5, 12.0) for _ in range(2047)],
        "gauss m=2048 s~N(0,1)": lambda: [random.gauss(0.0, 1.0) for _ in range(2048)],
        "small m=8   s~N(0,1)": lambda: [random.gauss(0.0, 1.0) for _ in range(8)],
    }
    print("%-26s %-8s %12s %12s %12s" % ("case", "grid", "rel_err_max", "abs_err_max", "l_typical"))
    for name, gen in cases.items():
        for grid in GRID_CANDIDATES:
            rmax = amax = 0.0
            lsum = 0.0
            for _ in range(ROWS):
                score = gen()
                r, a = rel_err(score, grid)
                mx = max(score)
                lsum += math.fsum(math.exp(s - mx) for s in score)
                rmax = max(rmax, r)
                amax = max(amax, a)
            print("%-26s %-8d %12.3e %12.3e %12.3f"
                  % (name, grid, rmax, amax, lsum / ROWS))
    print("\nread: pick the coarsest grid whose rel_err stays in the band you want to test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
