#!/bin/bash
# P124 本地标定：两臂 = pristine / 死端串 63 圈（同一份 pristine 现生成）—— 只要证明这把尺子在本地**没被擦掉**。
# 命门（比 P121 还硬一档）：① **编译器有没有把 `x*1.0f` 折成空** ⇒ 若折了，N=3 与 N=1 是同一份
#   机器码、本地 Δ 随 N 不动 ⇒ 尺子不存在，这一发**不许发**（P121 靠 `acc>>63` 防 CSE，这里没法防：
#   乘 1 就是乘 1，只能读出来再判）。② Δ vs N 线性 ⇒ 线性才允许把平台读数折回"一圈的价"。
# ③ 逐位惰性：八档 out/max/sum 对一次性 golden_p124/ 逐位比 —— 这一条不过就是补丁本身错了。
# ⚠️ 补丁只落远端副本；退出 trap = 只做一次 npu.sh sync（不 build、不 sed）⇒ 远端回到可发次态。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

CASES="${CASES:-g64c1 g64c4 g64c16 v64c8 r64c65 p6 big1 w3}"
REPS="${REPS:-5}"
ARMS="${ARMS:-0 63}"
# ⚠️ 双臂必须**同一份源**：默认 = 本地提交源（铺过 N 的那份）。若它 != f815bf1e，脚本自己拉回 pristine。
PRISTINE="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
if [ "$(sha256sum "$PRISTINE" | cut -c1-16)" != "f815bf1eaba0f8bc" ]; then PRISTINE="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"; fi

restore() {
  echo "=== 还原：只做 sync（远端源码 = 本地提交字节，无 sed、无 build） ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16; echo SEDHITS=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning | tail -2
}
trap restore EXIT

sha_local=$(sha256sum "$PRISTINE" | cut -c1-16)
cat "$PRISTINE" | nssh "cat > ~/sfa_real/kernel_pristine.cpp"
( cd "$REPO/code 3/probes" && tar cf - p124_deadrep.py ) | nssh "cd ~/sfa_real && tar xf -"
echo ">>> pristine sha 本地=$sha_local 远端应同"
nssh "cd ~/sfa_real && sha256sum kernel_pristine.cpp | cut -c1-16" 2>&1 | grep -av Warning | tail -1

for N in $ARMS; do
  echo "########## ARM N=$N (链上多串 $N 圈 / 每圈 nbCur 条 Muls(rowC)) ##########"
  nssh "cd ~/sfa_real && python3 p124_deadrep.py $N code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" 2>&1 | grep -av Warning | tail -1
  nssh "cd ~/sfa_real && echo SED_TMP=\$(grep -c ascend910_93 code/op_host/sparse_flash_attention.cpp) && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' code/CMakeLists.txt && \
    grep -q ascend910_93 code/op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' code/op_host/sparse_flash_attention.cpp" >/dev/null
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
  WANT=$((N>0 ? 1 : 0))
  nssh "cd ~/sfa_real/code && echo HITS_P124=\$(grep -c 'P124 死端孪生计量器' op_kernel/sparse_flash_attention.cpp) \
    LOOP=\$(grep -c 'p124r = 0u; p124r < ' op_kernel/sparse_flash_attention.cpp) \
    INNER=\$(grep -c 'Muls(dk, dk, 1.0f, rowC)' op_kernel/sparse_flash_attention.cpp) \
    MULS=\$(grep -c 'Muls(' op_kernel/sparse_flash_attention.cpp) \
    FLAG=\$(grep -c 'Flag<HardEvent' op_kernel/sparse_flash_attention.cpp) \
    BUFF=\$(grep -c InitBuffer op_kernel/sparse_flash_attention.cpp) 期望 HITS=$WANT LOOP=$WANT INNER=$WANT MULS=$((11 + WANT)) FLAG=36 BUFF=13" 2>&1 | grep -av Warning
  ALIVE=0
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
  # 惰性自证：pristine 臂写一次性 golden_p124/，探针臂与之逐位比
  ACT=write; [ "$N" = "0" ] || ACT=diff
  nssh "$ENVR; for c in $CASES; do printf '%-11s ' \$c; ./test_sfa_dev cases/\$c.bin 1 $ACT 2e-3 1e-2 golden_p124 2>&1 | grep -aoE '逐位一致 [^ ]+|不\*\*逐位一致\*\*|golden 写入 [^:]+: [A-Z]+' | tr '\n' ' '; echo; done" 2>&1 | grep -av "^Warning"
  nssh "$ENVR; for c in $CASES; do t=\$(timeout 1800 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-11s %10s\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning"
done
echo P124_CALIB_DONE
