#!/bin/bash
# P105 网格：把 (nb, kv_shard) 两档一起扫，读"加深切分到底值不值"。
# 与 P17a 的 run_nbsweep.sh 同一套纪律：只改**远端副本**（nb_sweep_patch.py 的
# SFA_FORCE_NB / SFA_FORCE_KS 两个旋钮），退出时一律还原成干净构建。
# 用法：
#   SMOKE=1 bash code\ 3/probes/p105_grid.sh      # 只跑 3 发：ks=2 回归 + ks=4/10 会不会挂
#   bash    code\ 3/probes/p105_grid.sh           # 全网格（含 AUTO 自选档）
set -u
HOST_ALIAS="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
SSH_CFG="$HOME/.atomgitdevenv/.ssh/config"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
nssh() { timeout "${TMO:-2400}" ssh -F "$SSH_CFG" -o ConnectTimeout=25 "$HOST_ALIAS" "$@"; }
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 3 none" 2>&1 | grep -aE "超差|批量口径" | tail -3
}
trap restore EXIT

# 1) 打旋钮补丁 —— grep -c 命中数当门（P55 那次自欺就是少了这道门）
( cd "$REPO/code 3/probes" && tar cf - nb_sweep_patch.py p105_grid_body.sh ) \
  | nssh "cd ~/sfa_real && tar xf -"
nssh "cd ~/sfa_real && python3 nb_sweep_patch.py code/op_host/sparse_flash_attention.cpp && \
  grep -c 'SFA_FORCE\|SFA_PICK' code/op_host/sparse_flash_attention.cpp" | tail -2
# 2) 增量重建（改的是 host cpp）
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
# 2b) 探活门：设备被残留上下文占住时**每一发**都 rc=2（§7.8 那条"先怀疑自己的字节"的镜像
#     形态：这次字节没问题、占住的是别人）。整批网格白跑一遍就是刚才那次 ⇒ 必须先撞活再扫。
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; break
  fi
  echo ">>> 探活第 $try 次失败（设备无响应），等 60 s"; sleep 60
done
# 3) 扫
SMOKE="${SMOKE:-0}"; REPS="${REPS:-5}"
nssh "SMOKE=$SMOKE REPS=$REPS bash ~/sfa_real/p105_grid_body.sh" 2>&1 | grep -av "^Warning"
