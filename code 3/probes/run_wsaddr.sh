#!/bin/bash
# 任务 #22：arch22 workspace 可写窗口矩阵。⚠️ 只改远端副本，trap 还原干净构建。
# 用法: run_wsaddr.sh [mode ...]     默认 addr wsbase0 wsbase120k wsbase1m wsself
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
HOST="code/op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
LINK="-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lcust_opapi"
# ⚠️ 必须和 refs/sfa/run.sh 一致：先试 -lnnopcapbase，失败才退回 -lnnopbase。
# 直接 -lnnopbase 会在链接期炸 `libnnopbase.so: undefined reference to gert::OppSoDesc::~OppSoDesc()`
# 并把已有的 test_sfa_dev 打成不存在（本轮踩过一次，5 档全 rc=127）。
HLINK="$LINK -lnnopcapbase"
HLINK2="$LINK -lnnopbase"
rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }

CLEANBUILD() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; \
  rssh "cd ~/sfa_real && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' code/CMakeLists.txt && \
        sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' $HOST"; \
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error" | head -4; \
  rssh "cat > /tmp/wsaddr_harness_patch.py" < "$REPO/code 3/probes/wsaddr_harness_patch.py"; \
  rssh "cd ~/sfa_real && python3 /tmp/wsaddr_harness_patch.py && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    { g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd $HLINK 2>/dev/null || \
      g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd $HLINK2 2>&1 | head -8; } ; \
    [ -x /tmp/tsd ] && mv -f /tmp/tsd test_sfa_dev; ls -la test_sfa_dev" 2>&1 | tail -3; }

restore() { echo "=== 还原干净构建 ==="; CLEANBUILD; \
  rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间:|==>" | tail -3; }
trap restore EXIT

echo "=== 0) 拉干净 + SoC 补丁 + harness 读数 + 声明 128 KB workspace ==="
CLEANBUILD
# host：把 op 声明的 workspace 从 0 提到 128 KB（探针口径，P11 若要跨核暂存也要声明）
rssh "cd ~/sfa_real && sed -i 's|if (currentWorkspace != nullptr) { currentWorkspace\[0\] = 0; }|if (currentWorkspace != nullptr) { currentWorkspace[0] = 128u * 1024u; }  // WSADDRPROBE|' $HOST && grep -n 'WSADDRPROBE' $HOST"

if [ "$#" -eq 0 ]; then set -- addr wsbase0 wsbase120k wsbase1m wsself; fi
for m in "$@"; do
  echo "########## wsaddr probe = $m ##########"
  python3 "$REPO/code 3/probes/mk_probe_wsaddr.py" "$m" | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -12
  rssh "$ENVR; timeout 300 ./test_sfa_dev cases/big1.bin 1 none >/tmp/wa.txt 2>&1; echo \"rc=\$?\"; \
    grep -aE '\[WSADDR\]|^case=' /tmp/wa.txt; \
    grep -aiE 'out of range|errorStr|errcode' /tmp/wa.txt | head -3"
done
