#!/bin/bash
# blockDim 固定开销探针：**只改远端副本的 op_host**，本地提交源不动，结束一律还原干净构建。
# 用法: run_blockdim_probe.sh [bd1 bdauto]
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
HSO="code/op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
LINK='-I$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I$HOME/sfa_real/vendor/custom/op_api/include -L$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi'

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位|批量" | tail -4
}
trap restore EXIT

for m in "${@:-bd1 bdauto}"; do
  echo "########## blockdim probe = $m ##########"
  python3 "$REPO/code 3/probes/mk_probe_blockdim.py" "$m" | \
    timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cat > ~/sfa_real/$HSO" || { echo "PUSH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error|Error" | head -6
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev $LINK 2>&1 | head -5"
  timeout 900 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; for n in r1_min r8_heads p1 big1; do printf '%-10s ' \$n; \
    timeout 300 ./test_sfa_dev cases/\$n.bin 3 diff 2>&1 | grep -aoE '平均 [0-9.]+ ms|单发/批量 = [0-9.]+|不逐位|超差 [1-9][0-9]*/[0-9]+' | head -3 | tr '\n' '|'; echo; done"
done
