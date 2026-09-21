set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
for cs in p1 q2h r5_blocks e2one; do
 for f in "" SFA_F32=1; do
  for nb in 1 2 4 8; do
   for kb in 8 16 32; do
    for ks in 1 2; do
     r=$(env SFA_FORCE_NB=$nb SFA_FORCE_NBLK=$kb SFA_FORCE_KS=$ks $f timeout 200 ./test_sfa_dev cases/$cs.bin 3 none 2>&1 | grep -aoE "超差 [0-9]+/[0-9]+" | head -1)
     bad=$(echo "$r" | grep -oE "[1-9][0-9]*/[0-9]+" || true)
     if [ -n "$bad" ]; then printf "BAD  %-10s %-4s nb=%-2s kb=%-2s ks=%s  %s\n" $cs "$f" $nb $kb $ks "$r"; fi
    done
   done
  done
 done
done
echo SWEEP_DONE
