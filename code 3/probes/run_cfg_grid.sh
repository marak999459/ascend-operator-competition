#!/bin/bash
# 同场次 (nb, n_blk, ks) 网格：先 npu.sh sync 当前本地 code 3/code，再在**远端副本**上
# 打 SoC 补丁 + FORCE 旋钮补丁，构建，然后逐格跑 ./test_sfa_dev <case> 3 diff。
# 用法: run_grid.sh <TAG>      输出 /tmp/grid_<TAG>.txt
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
SOC=ascend910_93
TAG="${1:?TAG}"
OUT="/tmp/grid_${TAG}.txt"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

bash "$REPO/code 3/npu_debug/npu.sh" sync > /tmp/grid_${TAG}_sync.log 2>&1
timeout 300 ssh -F $S -o ConnectTimeout=25 $H "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT $SOC)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"$SOC\")|' op_host/sparse_flash_attention.cpp && \
  python3 ~/code\ 3/probes/p23_sweep_patch.py op_host/sparse_flash_attention.cpp && \
  grep -c SFA_FORCE_NB op_host/sparse_flash_attention.cpp" 2>&1 | grep -v Warning

: > $OUT
timeout 900 ssh -F $S -o ConnectTimeout=25 $H "cd ~/sfa_real && bash build.sh" 2>&1 \
  | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:" | head -10 | tee -a $OUT

timeout 3000 ssh -F $S -o ConnectTimeout=25 $H "$ENVR; \
for cs in p1 p4 p6 q2h big1; do \
 for nb in 1 2 4 8; do \
  for kb in 16 32; do \
   for ks in 1 2; do \
    if [ \$nb -gt 4 ] && [ \$cs != big1 ]; then continue; fi; \
    line=\$(env SFA_FORCE_NB=\$nb SFA_FORCE_NBLK=\$kb SFA_FORCE_KS=\$ks timeout 200 ./test_sfa_dev cases/\$cs.bin 3 diff 2>&1 | grep -aoE '平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|FAIL' | tr '\n' ' '); \
    printf '%-6s nb=%-2s kb=%-2s ks=%s | %s\n' \$cs \$nb \$kb \$ks \"\$line\"; \
   done; \
  done; \
 done; \
done" 2>&1 | grep -v Warning | tee -a $OUT
echo "GRID_DONE $TAG"
