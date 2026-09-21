#!/bin/bash
# 任务 #26：P11v2 前置门 —— **AIV→AIV** 跨核数据通道（输出张量当暂存 + `SyncAll` 当屏障）。
# ⚠️ 只改远端副本（kernel / test_sfa_dev.cpp），trap 一律还原成干净构建；本地提交源零改动。
# 用法: run_xchain.sh [both|nosync|scalar ...]      默认 both nosync
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
PROB="$REPO/code 3/probes"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
LINK="-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lcust_opapi"
HLINK="$LINK -lnnopcapbase"
HLINK2="$LINK -lnnopbase"
rssh() { timeout "${T:-900}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SOC() { rssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"; }
# harness 构建必须**响亮地**失败：g++ 挂掉时若 test_sfa_dev 还在，就会静默沿用上一次的旧二进制
# （读到的会是旧格式 + 旧行为，§15.20 踩过）。
BUILDH() { local o; o=$(rssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    rm -f /tmp/tsd; \
    { g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd $HLINK 2>/dev/null || \
      g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd $HLINK2 2>&1 | head -8; } ; \
    if [ -x /tmp/tsd ]; then mv -f /tmp/tsd test_sfa_dev; echo 'HARNESS OK'; \
    else echo 'HARNESS BUILD FAIL'; fi" 2>&1 | tail -4)
  printf '%s\n' "$o"
  case "$o" in *"HARNESS OK"*) return 0;; *) return 1;; esac; }
CLEANBUILD() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; SOC >/dev/null; \
    rssh "cat > ~/sfa_real/cube_harness_patch.py" < "$PROB/cube_harness_patch.py"; \
    rssh "cat > ~/sfa_real/xchain_screen.py" < "$PROB/xchain_screen.py"; \
    rssh "cd ~/sfa_real && python3 cube_harness_patch.py && python3 xchain_screen.py" || { echo 'HARNESS PATCH FAIL'; return 1; }; \
    BUILDH; }
# 🔴 trap 必须**连算子一起重建**：CLEANBUILD 只管 harness + 源码，跑完会把探针那份 `.so` 留在远端，
#   下一次"看起来是干净态"的计时量的其实是探针空转（实测 p1 批量口径 0.0098 ms vs 真值 0.292 ms）。
restore() { echo "=== 还原干净构建（含算子） ==="; \
    bash "$REPO/code 3/npu_debug/dev.sh" build 2>&1 | grep -aE "构建 OK|MAKE FAIL|error" | tail -3; \
    CLEANBUILD >/dev/null 2>&1; \
    rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位" | tail -3; }
trap restore EXIT

if [ "$#" -eq 0 ]; then set -- both nosync; fi

echo "=== 0) 拉回干净态 + 两段 harness 补丁 + 首次构建 ==="
CLEANBUILD || { echo "前置 harness 构建失败 ⇒ 全部档位放弃（否则会读到旧二进制的假数）"; exit 1; }

for m in "$@"; do
  echo "########## xchain probe = $m ##########"
  CLEANBUILD >/dev/null || { echo ">>> $m: 前置构建失败，跳过"; continue; }
  python3 "$PROB/mk_probe_xchain.py" "$m" | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
  BOUT=$(rssh "$ENVR; bash build.sh" 2>&1)
  printf '%s\n' "$BOUT" | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -20
  if ! printf '%s\n' "$BOUT" | grep -aq "构建 OK"; then
    echo ">>> $m: 算子构建失败 ⇒ 跳过（避免沿用上一次的 .so 读假数）"; continue
  fi
  # big1：nlse = B*S1*N1 = 1024 个 float ⇒ 放得下 40 块的发布区 + 读回区（p1 只有 16 格，不够）
  # ⚠️ 连跑 XC 次：一次 relayBad=0 可能只是"各块恰好都赶上了"，通道是否真被屏障兜住要重复采样。
  XC="${XC:-3}"
  rssh "$ENVR; for _r in \$(seq 1 $XC); do \
      SFA_CHAIN=40 SFA_QUIET_XC3=1 SFA_CUBE_EXIT=1 timeout ${TO:-300} stdbuf -o0 -e0 \
      ./test_sfa_dev cases/big1.bin 1 none >/tmp/xc.txt 2>&1; echo \"rc=\$?\"; \
      grep -aE '\[CUBE\] CHAIN' /tmp/xc.txt | head -1; \
      grep -aiE 'error|fail|exception' /tmp/xc.txt | head -2; done"
done
