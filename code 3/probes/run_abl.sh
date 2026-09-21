#!/bin/bash
# P19 瓶颈定位（任务 #35）：把 FlushChunk 的计算逐段删掉，差分 = 该段在**平台形状**上的边际成本。
# 与 §15.10(b)（big1 / nb=8 / score 71 %）同方法、不同形状 ⇒ 决定 M1（score 上 Cube）还是
# M2（PV 上 Cube）先做。
# ⚠️ 纯计时：所有档输出都是错的 ⇒ 一律 act=none，绝不 write golden。
# 用法: run_abl.sh [full nosc nomax noexp nosum nopv nocalc ...]    默认全部
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
CASES="${CASES:-p1 p4 p6 q1h big1}"

rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SOC() { rssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"; }
HARNESS() {
    local o
    o=$(rssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real && \
        LINK=\"-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi\"; \
        g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd_new \$LINK 2>&1 | head -6; \
        test -x /tmp/tsd_new && mv /tmp/tsd_new test_sfa_dev && md5sum test_sfa_dev")
    printf '%s\n' "$o" | grep -aq "test_sfa_dev$" || { echo ">>> HARNESS FAIL: 重编未产出二进制"; return 1; }
    return 0
}
CLEANBUILD() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; SOC >/dev/null; \
    rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error" | head -6; \
    HARNESS || return 1; }
BUILD() { local o; o=$(rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -20); \
    printf '%s\n' "$o"; printf '%s\n' "$o" | grep -aq "构建 OK"; }
PERF() { local ks="$1"; rssh "$ENVR; for n in $CASES; do printf '  ks=%-4s %-6s ' \"$ks\" \"\$n\"; \
    SFA_MIXBD=0 SFA_FORCE_KS=$ks timeout 300 ./test_sfa_dev cases/\$n.bin 5 none >/tmp/ab.txt 2>&1; \
    printf 'rc=%s ' \$?; grep -aoE 'SFA_PICK.*|平均 [0-9.]+ ms' /tmp/ab.txt | tr '\n' '|'; echo; done"; }
restore() { echo "=== 还原干净构建 ==="; CLEANBUILD; \
    rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位" | tail -4; }
trap restore EXIT

MODES=("$@"); if [ "${#MODES[@]}" -eq 0 ]; then MODES=(full nosc nomax noexp nosum nopv nocalc); fi
echo "=== 0) 干净态 ==="; CLEANBUILD || { echo ">>> 前置构建失败"; exit 1; }
rssh "cat > /tmp/mixm0_host_patch.py" < "$REPO/code 3/probes/mixm0_host_patch.py"
for m in "${MODES[@]}"; do
  echo "########## abl = $m ##########"
  CLEANBUILD || { echo ">>> $m: 前置构建失败"; continue; }
  if [ "$m" != "full" ]; then
    python3 "$REPO/code 3/probes/mk_probe_abl.py" "$m" | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
    printf '  kernel md5 本地产物 vs 远端: %s\n' "$(python3 "$REPO/code 3/probes/mk_probe_abl.py" "$m" | md5sum | cut -d' ' -f1)"
  fi
  rssh "cd ~/sfa_real && python3 /tmp/mixm0_host_patch.py" || { echo ">>> $m: host 补丁失败"; continue; }
  echo "--- build ---"; BUILD || { echo ">>> $m: 构建失败，跳过（防旧 .so 读假数）"; continue; }
  echo "--- 计时（AUTO 档 ks=2  wherever 模型允许）---"; PERF "2"
  echo "--- 计时（强制 ks=1：一核一单元的纯临界路径）---"; PERF "1"
done
