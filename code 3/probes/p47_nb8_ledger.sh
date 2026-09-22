#!/bin/bash
# P47：量"§15.61(e)1 那道前置" —— M1 若要独占整行暂存就得 `nb=N1=8`，而 `nb=8` 在 UB 预算下
# 只能取 `n_blk<=40`（本地按 host 的 CalcUbNeed 复算：nb=8/48 = 204,512 B > ubSafe 186,534）。
# ⇒ 这一档同时付两笔：条纹量化（单元数减半 ⇒ 波数 7→4 但每轮 2 倍宽）+ 丢掉 P32 的 48 档。
# §15.60(d) 那张 M1 落点表是 nb=4 口径的，不重算就用不了。
#
# 三档（每档一次构建）：
#   auto      不动 host = P38 现状（AUTO 选档）
#   nb8       host 的 NB_CAND 砍成 {32,16,8} ⇒ qN=8 下唯一合法候选是 nb=8
#   nb8nosc   nb8 + 删 ComputeScores ⇒ **nb=8 口径的 score 段**（落点表要的就是这一格）
# 判据只有时间 ⇒ 所有档一律 act=none，绝不 act=write（探针纪律）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
HOST="op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
CASES="${CASES:-big1 d2048 w4}"
rssh() { timeout "${T:-1600}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SYNC() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; }
SOC() { rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"; }
FORCE8() { rssh "cd ~/sfa_real/code && \
  sed -i 's|constexpr uint32_t NB_CAND\[\] = {32, 16, 8, 4, 2, 1};|constexpr uint32_t NB_CAND[] = {32, 16, 8};  // P47 FORCE8|' $HOST && \
  grep -n 'P47 FORCE8' $HOST | head -1"; }
TIME() { local m="$1"; for n in $CASES; do
    printf '  %-8s %-6s ' "$m" "$n"
    rssh "$ENVR; timeout 300 ./test_sfa_dev cases/$n.bin 5 none 2>&1 | grep -a '批量口径' | grep -aoE '平均 [0-9.]+ ms' | head -1" | tr -d '\r'
    echo
  done; }
CFG() { rssh "$ENVR; timeout 300 ./test_sfa_dev cases/big1.bin 1 none 2>&1 | grep -aiE '\bnb\b|n_blk|shard|块数|分块' | head -4"; }
restore() { echo "=== 还原干净构建 ==="; SYNC; SOC; \
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error:" | head -3; \
  rssh "$ENVR; ./test_sfa_dev cases/big1.bin 3 diff" 2>&1 | grep -aE "超差|批量口径|逐位" | head -3; \
  rssh "cd ~/sfa_real && grep -ac 'P47 FORCE8\|ABL' code/op_kernel/sparse_flash_attention.cpp code/op_host/sparse_flash_attention.cpp; md5sum code/op_kernel/sparse_flash_attention.cpp"; }
trap restore EXIT INT TERM

MODES=("$@"); if [ "${#MODES[@]}" -eq 0 ]; then MODES=(auto nb8 nb8nosc); fi
echo "=== 0) 干净态 ==="; SYNC; SOC; rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error:" | head -3 || exit 1
for m in "${MODES[@]}"; do
  SYNC; SOC
  case "$m" in
    auto)                            ;;
    nb8)     FORCE8                 ;;
    nb8nosc) FORCE8
             python3 "$REPO/code 3/probes/mk_probe_abl.py" nosc | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL $m"; continue; } ;;
    *) echo "未知档 $m"; continue ;;
  esac
  echo "########## P47 = $m  kernel md5(远端)=$(rssh "md5sum ~/sfa_real/$KER" | cut -c1-8) ##########"
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aqE "构建 OK" || { echo ">>> $m 构建失败（跳过，防旧 .so 读假数）"; continue; }
  echo "  [生效档位] $(CFG)"
  TIME "$m"
done
