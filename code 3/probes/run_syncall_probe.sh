#!/bin/bash
# SyncAll 可行性探针（P11 的前置）：**只改远端副本**，本地提交源不动，结束一律还原干净构建。
# 用法: run_syncall_probe.sh [bar]
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位" | tail -3
}
trap restore EXIT

for m in "${@:-bar}"; do
  echo "########## syncall probe = $m ##########"
  python3 "$REPO/code 3/probes/mk_probe_syncall.py" "$m" | \
    timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error|Error" | head -8
  # 每个用例外面套 120 s 硬超时：barrier 挂了就当场现形，不会把整轮拖死
  timeout 900 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; for n in r1_min p1 big1; do printf '%-8s ' \$n; \
    timeout 120 ./test_sfa_dev cases/\$n.bin 3 diff 2>&1 | grep -aoE '批量口径[^|]*|平均 [0-9.]+ ms|不逐位|超差 [1-9][0-9]*/[0-9]+' | head -3 | tr '\n' '|'; echo \" rc=\${PIPESTATUS[0]}\"; done"
done
