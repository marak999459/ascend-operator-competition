#!/bin/bash
# P6 第 1 号待钉项（任务 #23）：MIX 形态下 AIV 的 5~10× 到底是"索引墙"还是"并行度墙"。
# 依据（本轮头文件取证）：AIV 的 GetBlockIdx() = group*ratio + sub ∈ [0, 2*BD)，
#   而 GetBlockNum() = **组数 BD**（AIC/AIV 同值）⇒ 提交版 kernel 的 `coreIdx >= coreNum`
#   那道门在 MIX 下会砍掉一半 AIV 块，且活下来的块全挤在前 BD/2 个物理核上。
# 用法: run_mixgeo.sh [base pol1 pol2 ...]     默认 "pol1 pol2"
#   base = 不改 kernel 的 AIV-only 基准（同一会话对照，防"昨天 0.32 今天 0.40"这类漂移）
#   pol0 = 原样索引 + MIX（= §15.22 的复现档，只在需要重新钉基准时点名）
# ⚠️ 只改远端副本；trap 一律还原成干净构建（探针纪律 §15.14(f)）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
HOST="code/op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
BDS="${BDS:-8 16 20 27 40}"
CASES="${CASES:-p1 big1}"
rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SOC() { rssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
    grep -c ascend910_93 CMakeLists.txt op_host/sparse_flash_attention.cpp"; }
# 干净态 = 本地四文件 + 未打补丁的 harness（本探针不需要 harness 读数）
# ⚠️ harness 必须**每次重编**：npu.sh sync 用 tar 保留 mtime，磁盘上的 test_sfa_dev.cpp 是干净的，
#    但 test_sfa_dev **二进制**可能还是上一轮探针（run_cube_probe.sh）打补丁时编出来的那份。
#    本轮踩过：mixgeo 第一版沿用了 12:26 的补丁二进制 ⇒ 时间读数与 §15.22 的"8 ms"混在同一个
#    污染变量上，无法归因（code3.md §15.27 的归因自检）。
HARNESS() {
    local dirty o
    # 1) 磁盘 .cpp 必须无探针残留
    dirty=$(rssh "cd ~/sfa_real && grep -ac 'CUBEPROBE\|MIXPROBE' test_sfa_dev.cpp"); 
    if [ "${dirty:-1}" != "0" ]; then echo ">>> HARNESS FAIL: 远端 test_sfa_dev.cpp 仍有探针残留 ($dirty 行)"; return 1; fi
    # 2) 重编（g++ 失败绝不能静默沿用旧二进制）
    o=$(rssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real && \
        LINK=\"-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi\"; \
        g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd_new \$LINK 2>&1 | head -10; \
        test -x /tmp/tsd_new && mv /tmp/tsd_new test_sfa_dev && md5sum test_sfa_dev")
    printf '  harness: %s\n' "$o"
    printf '%s\n' "$o" | grep -aq "test_sfa_dev$" || { echo ">>> HARNESS FAIL: 重编未产出二进制"; return 1; }
    return 0
}
CLEANBUILD() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; SOC >/dev/null; \
    rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error" | head -8; \
    HARNESS || return 1; }
BUILD() { local o; o=$(rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -20); \
    printf '%s\n' "$o"; printf '%s\n' "$o" | grep -aq "构建 OK"; }
# 一个 BD 档：两个用例各跑 diff（正确性 + 平均时间一起出）
RUNBD() { local bd="$1"; rssh "$ENVR; for c in $CASES; do printf '  BD=%-3s %-6s ' \"$bd\" \"\$c\"; \
    SFA_BD=$bd timeout 300 ./test_sfa_dev cases/\$c.bin 3 diff >/tmp/mg_\$c.txt 2>&1; echo \"rc=\$?\"; \
    grep -aoE '超差 [0-9]+/[0-9]+|LSE 超差 [0-9]+/[0-9]+|平均 [0-9.]+ ms|逐位一致|不逐位' /tmp/mg_\$c.txt | tr '\n' '|'; echo; done"; }
restore() { echo "=== 还原干净构建 ==="; CLEANBUILD; \
    rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位" | tail -4; }
trap restore EXIT

POL=("$@"); if [ "${#POL[@]}" -eq 0 ]; then POL=(pol1 pol2); fi
echo "=== 0) 拉回干净态 + SoC 补丁 + 构建（含 harness 重编硬门）==="
CLEANBUILD || { echo ">>> 前置干净态构建失败：退出（绝不能拿旧二进制读数）"; exit 1; }

for p in "${POL[@]}"; do
  echo "########## mixgeo = $p ##########"
  CLEANBUILD || { echo ">>> $p: 前置构建失败"; continue; }
  if [ "$p" = "base" ]; then
    echo "--- AIV-only 基准（kernel 未改、块数由 tiling 自己定）---"
    rssh "$ENVR; for c in $CASES; do printf '  %-6s ' \"\$c\"; timeout 300 ./test_sfa_dev cases/\$c.bin 3 diff >/tmp/mg_\$c.txt 2>&1; \
      grep -aoE '超差 [0-9]+/[0-9]+|平均 [0-9.]+ ms|逐位一致|不逐位' /tmp/mg_\$c.txt | tr '\n' '|'; echo; done"
    continue
  fi
  python3 "$REPO/code 3/probes/mk_probe_mixgeo.py" "$p" | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
  rssh "cat > /tmp/mixgeo_host_patch.py" < "$REPO/code 3/probes/mixgeo_host_patch.py"
  rssh "cd ~/sfa_real && python3 /tmp/mixgeo_host_patch.py"
  echo "--- build ---"
  BUILD || { echo ">>> $p: 构建失败，跳过（防旧 .so 读假数）"; continue; }
  for bd in $BDS; do RUNBD "$bd"; done
done
