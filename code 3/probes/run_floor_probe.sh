#!/bin/bash
# 固定开销解剖探针：**只改远端副本**（op_host + op_kernel），本地提交源不动。
# 用法: run_floor_probe.sh [bare tilread nopro ...]     默认 base bare tilread nopro
# 结束（含失败）后一律 npu.sh sync + 干净构建还原。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
HSO="code/op_host/sparse_flash_attention.cpp"
KKR="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
L1='grep -aoE "平均 [0-9.]+ ms|批量口径[^|]*|单发/批量 = [0-9.]+" | head -4'

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位|单发/批量" | tail -4
}
trap restore EXIT

# harness 与探针无关，先编一次
timeout 600 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev \
  -I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include \
  -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib \
  -lascendcl -lnnopbase -lcust_opapi 2>&1 | head -5"

for m in "${@:-base bare tilread nopro}"; do
  echo "########## floor probe = $m ##########"
  B=/tmp/floor_$m.txt
  python3 "$REPO/code 3/probes/mk_probe_floor.py" "$m" > "$B" || { echo "GEN FAIL"; continue; }
  awk '/^===HOST===/{f=1;next} /^===KERNEL===/{f=2;next} f==1' "$B" > "$B.host"
  awk '/^===KERNEL===/{f=1;next} f==1' "$B" > "$B.kernel"
  wc -l "$B.host" "$B.kernel"
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cat > ~/sfa_real/$HSO" < "$B.host" || { echo "PUSH FAIL host"; continue; }
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cat > ~/sfa_real/$KKR" < "$B.kernel" || { echo "PUSH FAIL kernel"; continue; }
  # 每次推的都是本地纯净 op_host（只有 ascend910b），补一次 SoC 注册
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error|Error" | head -8
  timeout 900 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; for n in r1_min r8_heads p1; do printf '%-10s ' \$n; \
    timeout 300 ./test_sfa_dev cases/\$n.bin 3 none 2>&1 | $L1 | tr '\n' '|'; echo; done"
done
