#!/bin/bash
# 跑 p11_ptr_probe.py：kernel + harness 双向探针，判断 aicore 的 workspace 形参是不是调用方那块。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
rssh() { timeout "${T:-900}" ssh $C -o ConnectTimeout=25 "$H" "$1" 2>&1 | grep -av "^Warning:"; }

bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
T=120 ssh $C -o ConnectTimeout=25 "$H" 'cat > /tmp/p11_ptr.py' < "$REPO/code 3/probes/p11_ptr_probe.py"
T=300 rssh 'python3 /tmp/p11_ptr.py'
rssh "$ENVR; bash build.sh > /tmp/b7.log 2>&1; echo build_rc=\$?; grep -aE '构建 OK|error' /tmp/b7.log | head -5"
rssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1; \
  g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev \
    -I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include \
    -I\$HOME/sfa_real/vendor/custom/op_api/include \
    -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 \
    -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi 2>&1 | grep -aE 'error' | head -6; \
  ls -la test_sfa_dev | head -2"
T=600 rssh "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 1 none 2>&1 | grep -aE 'P11PROBE|HARNESS|FAIL|PASS|超差|批量' | head -8"

echo "=== 还原 ==="
bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
rssh "$ENVR; bash build.sh > /tmp/b7r.log 2>&1; echo restore_rc=\$?; grep -aE '构建 OK|error' /tmp/b7r.log | head -3"
