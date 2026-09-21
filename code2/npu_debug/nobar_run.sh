#!/bin/bash
# step-8' (barrier diet) paired grid: A(step-7) -> B(nobar) -> A(step-7), one detached session.
# Why three passes: same rule as vec8_pair.sh -- a single round can invent ~0.5us of drift,
# so the decision rests on the two step-7 brackets agreeing with each other. Here the two
# kernels differ ONLY by the five PipeBarrier<PIPE_V> call sites, so aiv_vec_time is the
# drift readout (V work unchanged) and aiv_scalar_time is where the effect should show up.
# f32_big_g.sh runs right after pass 2 because it measures whatever is currently installed.
set -u
T=/home/developer/mhc_test
echo "##### nobar_run version=2026-09-21"
export SHAPES="1024 8 20|1024 4 20|1024 6 20|2000 8 20|256 8 20|100 6 20|64 8 20|20 6 20|1 8 20"

echo "##### pass 1: step-7 baseline (with PIPE_V barriers)"
bash $T/prof_bench.sh $T/k_vec8_g31_str.cpp str7_p1 20 1
echo "##### pass 2: step-8' barrier diet"
bash $T/prof_bench.sh $T/k_vec8_nobar.cpp nobar 20 1
echo "##### pass 2b: large-g element compare on the installed build"
bash $T/f32_big_g.sh
echo "##### pass 3: step-7 again"
bash $T/prof_bench.sh $T/k_vec8_g31_str.cpp str7_p3 20 1
echo "##### nobar_run_all_done"
