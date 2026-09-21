#!/bin/bash
# P19-M0（任务 #32）：P18 的 AIV 算法在 **MIX 构建**上零退化吗？
# 三档同场次对比：
#   base = AIV-only 提交态（只打 host 探针补丁 ⇒ 同一套运行期旋钮）
#   mxa  = MIX + 索引修正 + `SyncAll()`（AIV-only 硬件屏障，AIC 被 ks_=1 挡在屏障外）
#   mxb  = MIX + 索引修正 + `SyncAll<false>()`（框架全栅协议，AIC 一起调）
# 每档跑：fp16 正确性矩阵（AUTO 档 + 强制 ks=2 档）、fp32 第二遍、8 个性能点 ks=1/ks=2 各一屏。
# ⚠️ 只改远端副本；trap 一律还原成干净构建（探针纪律 §15.14(f)）。
# 用法: run_mixm0.sh [base mxa mxb ...]      默认 "base mxa mxb"
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
RCASES="${RCASES:-r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads}"
PCASES="${PCASES:-p1 p2 p4 p6 q1h q2h q3h big1}"
rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SOC() { rssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
    grep -c ascend910_93 CMakeLists.txt op_host/sparse_flash_attention.cpp"; }
HARNESS() {
    local dirty o
    dirty=$(rssh "cd ~/sfa_real && grep -ac 'CUBEPROBE\|MIXPROBE\|MIXM0' test_sfa_dev.cpp")
    if [ "${dirty:-1}" != "0" ]; then echo ">>> HARNESS FAIL: 远端 test_sfa_dev.cpp 有探针残留 ($dirty 行)"; return 1; fi
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
# ---- 正确性矩阵：$1 = ks（空=AUTO），$2 = MIXBD，$3 = f32(空/1) ----
GATE() { local ks="$1" mb="$2" f3="$3"; rssh "$ENVR; p=0; f=0; for n in $RCASES; do \
    SFA_MIXBD=$mb ${ks:+SFA_FORCE_KS=$ks} ${f3:+SFA_F32=$f3} timeout 300 ./test_sfa_dev cases/\$n.bin 3 diff >/tmp/m0.txt 2>&1; rc=\$?; \
    if [ \$rc -eq 0 ]; then p=\$((p+1)); st=PASS; else f=\$((f+1)); st=FAIL; fi; \
    printf '  %-6s %s ' \"\$n\" \"\$st\"; \
    grep -aoE '超差 [0-9]+/[0-9]+|不逐位|逐位一致' /tmp/m0.txt | tr '\n' '|'; \
    [ \$rc -ne 0 ] && { echo \"rc=\$rc\"; tail -3 /tmp/m0.txt | tr '\n' ' '; }; echo; done; \
    echo \"-------- ks=${ks:-AUTO} MIXBD=$mb f32=${f3:-0} PASS=\$p FAIL=\$f\""; }
# ---- 性能屏：$1 = ks，$2 = MIXBD ----
PERF() { local ks="$1" mb="$2"; rssh "$ENVR; for n in $PCASES; do printf '  ks=%-4s %-6s ' \"$ks\" \"\$n\"; \
    SFA_MIXBD=$mb SFA_FORCE_KS=$ks timeout 300 ./test_sfa_dev cases/\$n.bin 3 diff >/tmp/m0p.txt 2>&1; \
    printf 'rc=%s ' \$?; \
    grep -aoE 'SFA_PICK.*|超差 [0-9]+/[0-9]+|平均 [0-9.]+ ms|不逐位|逐位一致' /tmp/m0p.txt | tr '\n' '|'; echo; done"; }
restore() { echo "=== 还原干净构建 ==="; CLEANBUILD; \
    rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|时间|逐位" | tail -4; }
trap restore EXIT

MODES=("$@"); if [ "${#MODES[@]}" -eq 0 ]; then MODES=(base mxa mxb); fi
echo "=== 0) 拉回干净态 + SoC 补丁 + 构建（含 harness 重编硬门）==="
CLEANBUILD || { echo ">>> 前置干净态构建失败：退出（绝不能拿旧二进制读数）"; exit 1; }

for m in "${MODES[@]}"; do
  echo "########## mixm0 = $m ##########"
  CLEANBUILD || { echo ">>> $m: 前置构建失败"; continue; }
  MB=0
  if [ "$m" != "base" ]; then
    python3 "$REPO/code 3/probes/mk_probe_mix.py" "$m" | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL"; continue; }
    MB=1
  fi
  rssh "cat > /tmp/mixm0_host_patch.py" < "$REPO/code 3/probes/mixm0_host_patch.py"
  rssh "cd ~/sfa_real && python3 /tmp/mixm0_host_patch.py" || { echo ">>> $m: host 补丁失败"; continue; }
  echo "--- build ---"; BUILD || { echo ">>> $m: 构建失败，跳过（防旧 .so 读假数）"; continue; }
  rssh "md5sum ~/sfa_real/$KER ~/sfa_real/code/op_host/sparse_flash_attention.cpp"
  echo "--- 正确性 fp16 (AUTO) ---"; GATE ""  "$MB" ""
  echo "--- 正确性 fp16 (强制 ks=2，专打屏障) ---"; GATE "2" "$MB" ""
  echo "--- 正确性 fp32 (AUTO) ---"; GATE "" "$MB" "1"
  echo "--- 性能 ks=1 ---"; PERF "1" "$MB"
  echo "--- 性能 ks=2 ---"; PERF "2" "$MB"
done
