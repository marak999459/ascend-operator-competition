#!/bin/bash
# 跑 §"idx 标量读 / MTE2 搬运" 差分探针：**只改远端副本**，本地提交源不动。
# 用法: run_idxscan_probe.sh [noidx|nomte|both ...]     默认 noidx nomte
# 结束（含失败）后一律 npu.sh sync + build 覆盖回干净版。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
  timeout 900 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位" | tail -3
}
trap restore EXIT

for m in "${@:-noidx nomte}"; do
  for mm in $m; do
    echo "########## probe = $mm ##########"
    python3 "$REPO/code 3/probes/mk_probe_idxscan.py" "$mm" | \
      timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
    timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error|Error" | head -5
    timeout 900 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; for n in p1 p6 big1; do printf '%-6s ' \$n; timeout 900 ./test_sfa_dev cases/\$n.bin 5 none 2>&1 | grep -a '时间' ; done"
  done
done
