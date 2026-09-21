#!/bin/bash
# step-7 paired grid: B(step-6b) -> A(step-7) -> B(step-6b) in one detached session.
# Why three passes: §11.20 #5 - a single round can invent ~0.5us of drift, so the
# decision rests on the two step-6b brackets agreeing with each other. The n=8 and the
# 20/64/100 shapes are byte-identical instruction streams across the two kernels (the
# strided branch is fp32 n<N_MAX only) => they are the drift readout, not a result.
set -u
T=/home/developer/mhc_test
export SHAPES="1024 8 20|1024 4 20|1024 6 20|2000 8 20|256 8 20|100 6 20|64 8 20|20 6 20|1 8 20"

echo "##### pass 1: step-6b (baseline)"
bash $T/prof_bench.sh $T/k_vec7_g31.cpp vec7g31_p1 20 1
echo "##### pass 2: step-7 (strided n<8 DMA)"
bash $T/prof_bench.sh $T/k_vec8_g31_str.cpp vec8str 20 1
echo "##### pass 3: step-6b again"
bash $T/prof_bench.sh $T/k_vec7_g31.cpp vec7g31_p3 20 1
echo "##### vec8_pair_all_done"
