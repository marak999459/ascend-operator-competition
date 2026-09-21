#!/bin/bash
# P31 探针：P29(k=40) 之后，(nb, k, ks) 三轴的"还能不能再大/再小"实测
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p31_grid.txt
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
bash "$REPO/code 3/npu_debug/npu.sh" sync > /tmp/p31_sync.log 2>&1
timeout 300 ssh -F $S -o ConnectTimeout=25 $H "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  python3 ~/code\ 3/probes/p23_sweep_patch.py op_host/sparse_flash_attention.cpp && \
  grep -c SFA_FORCE_NB op_host/sparse_flash_attention.cpp" 2>&1 | grep -av Warning
: > $OUT
timeout 900 ssh -F $S $H "cd ~/sfa_real && bash build.sh" 2>&1 | grep -aE "构建 OK|FAIL|error:" | head -5 | tee -a $OUT
for pass in 1 2; do
timeout 2500 ssh -F $S $H "$ENVR; \
for cs in p1 p2 q1h q2h p4 p6 big1; do \
 for nb in 1 2 4; do \
  for kb in 32 36 40; do \
   for ks in 1 2; do \
    if [ \$nb -gt 1 ] && [ \"\$cs\" != p2 ] && [ \"\$cs\" != q2h ] && [ \"\$cs\" != p4 ] && [ \"\$cs\" != p6 ] && [ \"\$cs\" != big1 ]; then continue; fi; \
    if [ \$nb -eq 4 ] && [ \$kb -eq 40 ] && [ \"\$cs\" != big1 ]; then continue; fi; \
    if [ \$kb -eq 40 ] && [ \$nb -eq 4 ]; then continue; fi; \
    line=\$(env SFA_FORCE_NB=\$nb SFA_FORCE_NBLK=\$kb SFA_FORCE_KS=\$ks timeout 200 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE '平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|FAIL|ERROR|err' | head -3 | tr '\n' ' '); \
    printf 'P%s %-6s nb=%-2s kb=%-2s ks=%s | %s\n' \$pass \$cs \$nb \$kb \$ks \"\$line\"; \
   done; \
  done; \
 done; \
done" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "P31_GRID_DONE"
