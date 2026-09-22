#!/bin/bash
# P43：在**当前 P38 构建 + 平台量级形状**上重跑 §15.38 的段级消融。
# 与 run_abl.sh 的唯一区别：**不碰 SFA_FORCE_KS**（该旋钮挂在 mixm0 补丁的 `kvShard` 覆盖上，
# 在 P32/P38 的 tiling 路径下会让 `full` 档读出 0.357 ms = AUTO 真值 0.6575 的一半 ⇒ 假数），
# 只量 AUTO 档。判据只有时间：所有档输出必然错 ⇒ 一律 act=none，绝不锁 golden。
# 用法: p43_abl_auto.sh [full nosc nopv nocalc ...]     默认全部七档
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
CASES="${CASES:-big1 d2048 w4}"
rssh() { timeout "${T:-1600}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SOC() { rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"; }
CLEAN() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; SOC; }
TIME() { local m="$1"; for n in $CASES; do
    printf '  %-4s %-6s ' "$m" "$n"
    rssh "$ENVR; timeout 300 ./test_sfa_dev cases/$n.bin 5 none 2>&1 | grep -a '批量口径' | grep -aoE '平均 [0-9.]+ ms' | head -1" | tr -d '\r'
    echo
  done; }
restore() { echo "=== 还原干净构建 ==="; CLEAN; \
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error:" | head -3; \
  rssh "$ENVR; ./test_sfa_dev cases/big1.bin 3 diff" 2>&1 | grep -aE "超差|批量口径|逐位" | head -3; \
  rssh "cd ~/sfa_real && grep -ac ABL code/op_kernel/sparse_flash_attention.cpp; md5sum code/op_kernel/sparse_flash_attention.cpp"; }
trap restore EXIT

MODES=("$@"); if [ "${#MODES[@]}" -eq 0 ]; then MODES=(full nosc nopv nocalc nomax noexp nosum); fi
echo "=== 0) 干净态 ==="; CLEAN; rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error:" | head -3 || exit 1
for m in "${MODES[@]}"; do
  CLEAN
  if [ "$m" != "full" ]; then
    python3 "$REPO/code 3/probes/mk_probe_abl.py" "$m" | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL $m"; continue; }
  fi
  echo "########## AUTO abl = $m  kernel md5(远端)=$(rssh "md5sum ~/sfa_real/$KER" | cut -c1-8) ##########"
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aqE "构建 OK" || { echo ">>> $m 构建失败（跳过，防旧 .so 读假数）"; continue; }
  TIME "$m"
done
