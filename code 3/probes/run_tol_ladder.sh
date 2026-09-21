set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
TAG="${TAG:-P23}"
for cs in p1 p4 p6 q3h e2one r8_heads; do
  for tol in 1e-2 1e-3 1e-4 1e-5; do
    line=$(env SFA_FORCE_NB= SFA_FORCE_NBLK= SFA_FORCE_KS= timeout 200 ./test_sfa_dev cases/$cs.bin 1 diff $tol $tol 2>&1 | grep -aoE "超差 [0-9]+/[0-9]+ +maxAbs=[0-9.e+-]+" | tr '\n' ' ')
    printf "%-4s %-9s tol=%-5s %s\n" "$TAG" "$cs" "$tol" "$line"
  done
done
