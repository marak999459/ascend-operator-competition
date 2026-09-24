#!/bin/bash
# P109 runner：远端副本打 nb_sweep 补丁 → 建 → 探活门 → 跑 p109_nbgrid_body.sh → 还原干净构建。
# ⚠️ 探针只落远端，本地提交源一个字都不动（退出时 trap 还原）。
# 用法：bash "code 3/probes/p109_nbgrid.sh"        （REPS=3 加速；ONLY="w3 w4" 缩范围）
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 3 none" 2>&1 | grep -aE "超差|批量口径|SFA_PICK" | tail -3
}
trap restore EXIT

bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | grep -aE "md5|=== " | tail -10
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  echo HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" | tail -1
( cd "$REPO/code 3/probes" && tar cf - nb_sweep_patch.py p109_nbgrid_body.sh ) | nssh "cd ~/sfa_real && tar xf -"
# 补丁 + **命中数当门**（§7.2）
nssh "cd ~/sfa_real && python3 nb_sweep_patch.py code/op_host/sparse_flash_attention.cpp && \
  echo HITS_FORCE_NB=\$(grep -c 'SFA_FORCE_NB' code/op_host/sparse_flash_attention.cpp) \
       HITS_FORCE_KS=\$(grep -c 'SFA_FORCE_KS' code/op_host/sparse_flash_attention.cpp) \
       HITS_PICK=\$(grep -c 'SFA_PICK' code/op_host/sparse_flash_attention.cpp)" | tail -2
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
# 探活门（§7.8）——不通就中止，别带着坏构建扫完整批
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; ALIVE=1; break
  fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
if [ "${ALIVE:-0}" != "1" ]; then echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; fi
nssh "REPS=${REPS:-5} ONLY='${ONLY:-}' bash ~/sfa_real/p109_nbgrid_body.sh" 2>&1 | grep -av "^Warning"
