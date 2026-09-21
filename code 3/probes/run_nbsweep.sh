#!/bin/bash
# P17a 前置：SBS=2 形态下的 (nb, kv_shard) 组合扫描。只在远端副本上跑，本地提交源零改动。
# 用法：bash code\ 3/probes/run_nbsweep.sh [case...]     默认 p1s2 p2s2 p4s2 p6s2
# 结束（含失败）后一律 npu.sh sync + SoC 补丁 + 干净构建还原，并用 p1 diff 确认还原。
set -u
HOST_ALIAS="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
SSH_CFG="$HOME/.atomgitdevenv/.ssh/config"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
nssh() { timeout "${TMO:-1800}" ssh -F "$SSH_CFG" -o ConnectTimeout=25 "$HOST_ALIAS" "$@"; }
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
CASES="${*:-p1s2 p2s2 p4s2 p6s2}"

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
  nssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|逐位|批量口径|时间:" | tail -5
}
trap restore EXIT

# 1) 推补丁并应用（远端副本）
( cd "$REPO/code 3/probes" && tar cf - nb_sweep_patch.py ) | nssh "cd ~/sfa_real && tar xf -"
nssh "cd ~/sfa_real && python3 nb_sweep_patch.py code/op_host/sparse_flash_attention.cpp && grep -c 'SFA_FORCE\|SFA_PICK' code/op_host/sparse_flash_attention.cpp"
# 2) 重建算子（增量即可：改的是 host cpp，cmake 按 mtime 重编）
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
# 3) 推扫描主体并执行
( cd "$REPO/code 3/probes" && tar cf - sweep_body.sh ) | nssh "cd ~/sfa_real && tar xf -"
nssh "AUTO_ONLY=${AUTO_ONLY:-0} bash ~/sfa_real/sweep_body.sh $CASES" 2>&1 | grep -av "^Warning"
