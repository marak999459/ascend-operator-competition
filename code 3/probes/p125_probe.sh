#!/bin/bash
# P125 发次前置：把"每单元一次的那段 ×N"计量器铺进**本地提交源** → sync → sed → build
# → 命中数门（§7 第 2 条）→ p32_gate 数值闸门（真 golden 双 dtype + 逐位）。
# 用法：bash p125_probe.sh q|w N        （q=prologue，w=epilogue）
# ⚠️ 前置①：p125_calib.sh + p125_read.py 必须已给出 **门 B 认线性**（每圈单价 u(N)=中位Δ/(N−1)
#           跨 N 稳定 ⇒ 重复段没被折叠/吸收；⛔ 旧的"N=1≠N=2"判法作废，理由见 p125_design.txt 第 4 条）
#           且 **门 C** 已把这一档 N 挑出来（本地最差档 Δ ≤ +15 %）并写回 p125_design.txt。
# ⚠️ 前置②：本地 kernel 必须 == pristine f815bf1e（⛔ 不许叠在别的探针上）。
# 退出 trap **不做还原**：探针态就是要发出去的字节，还原由 p125_reissue.sh 负责。
set -u
ARM="${1:-}"
N="${2:-}"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1800}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
K="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
BK="$REPO/code 3/probes/backup/p125_pre_probe/sparse_flash_attention.cpp"
SHA='f815bf1eaba0f8bc'
# 行差 = 段增量：HDR 3 行注释 + `for` 1 + `}` 1 = +5，减去被循环体顶掉的空行/注释行：
#   q：锚点 7 行（2 注释 + 1 空 + LoadQ + 3 Duplicate）→ 3+1+4+1=9 ⇒ +2
#   w：① 5→3+1+5+1=10 (+5)、② 9→3+1+9+1=14 (+5) ⇒ +10（锚点行数是 p125_pin.py 里的字面量，改注释要同改这里）
case "$ARM" in q) WANT=1; EXP_LINES=2;; w) WANT=2; EXP_LINES=10;; *) echo ">>> 用法 p125_probe.sh q|w N"; exit 1;; esac
case "$N" in (''|*[!0-9]*) echo ">>> N 必须是 1..255 的整数"; exit 1;; esac
[ "$N" -ge 1 ] && [ "$N" -le 255 ] || { echo ">>> N 要在 1..255"; exit 1; }

echo "########## 0) pristine 校验 + 备份 (ARM=$ARM N=$N) ##########"
[ "$(sha256sum "$K" | cut -c1-16)" = "$SHA" ] || { echo ">>> 本地 kernel != pristine ⇒ 中止（先跑 p125_reissue.sh 还原，避免叠补丁）"; exit 1; }
mkdir -p "$(dirname "$BK")"
cp -n "$K" "$BK"
[ "$(sha256sum "$BK" | cut -c1-16)" = "$SHA" ] || { echo ">>> 备份不是 pristine ⇒ 中止"; exit 1; }
sha256sum "$BK" | cut -c1-16,66-

echo "########## 1) 本地铺 ARM=$ARM N=$N ##########"
python3 "$REPO/code 3/probes/p125_pin.py" "$N" "$ARM" "$K" "$BK"
M=$(grep -c 'P125 计量器' "$K"); R=$(grep -c 'p125r = 0u' "$K")
F=$(grep -cE '(Set|Wait)Flag<HardEvent::' "$K"); MU=$(grep -c 'Muls(' "$K")
LQ=$(grep -c 'LoadQ(q, kb, kr,' "$K"); B=$(grep -c 'InitBuffer' "$K")
DL=$(( $(wc -l < "$K") - $(wc -l < "$BK") ))
FB=$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0' "$K")
echo "本地 MARK=$M(应=$WANT) REP=$R(应=$WANT) FLAG=$F(应=36) MULS=$MU(应=11) LOADQ=$LQ(应=1) BUFF=$B(应=13) 行差=$DL(应=$EXP_LINES) FORBID=$FB(应=0)"
[ "$M" = "$WANT" ] && [ "$R" = "$WANT" ] && [ "$F" = 36 ] && [ "$MU" = 11 ] && [ "$LQ" = 1 ] && \
[ "$B" = 13 ] && [ "$DL" = "$EXP_LINES" ] && [ "$FB" = 0 ] || { echo ">>> 本地命中数门不过 ⇒ 中止"; exit 1; }

echo "########## 2) sync + 远端 sha + SoC sed + build ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -2
nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16,66-" 2>&1 | grep -av Warning
nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp" >/dev/null
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error:' | head -5"

echo "########## 3) 远端命中数门（与本地同口径）+ 探活 ×6 ##########"
nssh "cd ~/sfa_real/code && echo MARK=\$(grep -c 'P125 计量器' op_kernel/sparse_flash_attention.cpp) \
  REP=\$(grep -c 'p125r = 0u' op_kernel/sparse_flash_attention.cpp) \
  FLAG=\$(grep -cE '(Set|Wait)Flag<HardEvent::' op_kernel/sparse_flash_attention.cpp) \
  MULS=\$(grep -c 'Muls(' op_kernel/sparse_flash_attention.cpp) \
  OLD=\$(grep -cE 'P12[0-4]|p12[0-4]|P11[0-9]|skP119|probeRep_|Muls\(oc, oc|Muls\(dk, dk|P124 死端' op_kernel/sparse_flash_attention.cpp) \
  FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0' op_kernel/sparse_flash_attention.cpp) \
  期望 MARK=$WANT REP=$WANT FLAG=36 MULS=11 OLD=0 FORBID=0" 2>&1 | grep -av Warning
ALIVE=0
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
[ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止（本地仍是探针态，人工决定还原）"; exit 1; }

echo "########## 4) 数值闸门 p32_gate.sh（真 golden 双 dtype + 逐位）##########"
bash "$REPO/code 3/probes/p32_gate.sh" 2>&1 | tail -45
echo "P125_PROBE_DONE ARM=$ARM N=$N ⇒ 下一步：bash p125_precheck.sh $ARM $N（最后一次 sync + 核四枚 + dry-run）"
