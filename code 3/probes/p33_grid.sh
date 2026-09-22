#!/bin/bash
# P33 探针：**P32 构建上**重跑 (nb, k, ks) 强制档网格。
# 为什么还要跑一遍：P31 那张 184 格的表是在 **P29 构建**上量的，那时 nb>=2 的 k=48 档
# 根本不可行（越物理 UB）⇒ 它"模型现选 == 全场最快 7/7"的结论对 **nb 轴**只在 k<=40 的世界里成立。
# P32 把 nb=1/2/4 的 48 档和 nb=8 的 40 档同时打开 ⇒ 每个 nb 的"最便宜档"整体换了，
# 而代价模型的 nb 选择用的是与 k 无关的 UnitCalls（§15.50 已立案）⇒ **nb 轴可能已经翻船**。
# 判读：每案取 AUTO（模型自选，读 SFA_PICK）当基准，找有没有比它更快的格。
# 补丁只作用在远端副本 ~/sfa_real/code（SoC sed + p23_sweep_patch.py），本地提交源不动。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p33_grid.txt
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
bash "$REPO/code 3/npu_debug/npu.sh" sync > /tmp/p33_sync.log 2>&1
timeout 300 ssh -F $S -o ConnectTimeout=25 $H "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  python3 ~/code\ 3/probes/p23_sweep_patch.py op_host/sparse_flash_attention.cpp && \
  grep -c SFA_FORCE_NB op_host/sparse_flash_attention.cpp" 2>&1 | grep -av Warning
: > $OUT
timeout 900 ssh -F $S $H "cd ~/sfa_real && bash build.sh" 2>&1 | grep -aE "构建 OK|FAIL|error:" | head -5 | tee -a $OUT
for pass in 1 2; do
timeout 2500 ssh -F $S $H "$ENVR; \
for cs in p1 q1h p2 q2h p4 p6 big1; do \
  a=\$(timeout 200 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK [^ ]+ [^ ]+ [^ ]+|平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|FAIL' | tr '\n' ' '); \
  printf 'P%s %-6s AUTO            | %s\n' \$pass \$cs \"\$a\"; \
  for nb in 1 2 4 8; do \
    if [ \$nb -gt 1 ] && [ \"\$cs\" = p1 ]; then continue; fi; \
    if [ \$nb -gt 2 ] && [ \"\$cs\" = q1h ]; then continue; fi; \
    for kb in 32 40 48; do \
      if [ \$nb -eq 8 ] && [ \$kb -gt 40 ]; then continue; fi; \
      for ks in 1 2; do \
        line=\$(env SFA_FORCE_NB=\$nb SFA_FORCE_NBLK=\$kb SFA_FORCE_KS=\$ks timeout 200 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|FAIL|ERROR|err' | head -4 | tr '\n' ' '); \
        printf 'P%s %-6s nb=%-2s kb=%-2s ks=%s | %s\n' \$pass \$cs \$nb \$kb \$ks \"\$line\"; \
      done; \
    done; \
  done; \
done" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "P33_GRID_DONE"
