#!/bin/bash
# P125 本地标定（两臂 × 阶梯）：prologue=q（LoadQ+3×Duplicate）/ epilogue=w（WriteOut 两段 DMA 尾巴）
# 命门三条，缺一条这一臂就**不许发货**：
#   ① N=1 与 N=2 的时间差 <0.3 % ⇒ 重复段被 CSE/流水吸收了 ⇒ 尺子不存在（P121 立的规矩）。
#   ② Δ 随 N 近似 log-线性 ⇒ 线性才允许把平台读数折回"一遍的价"。
#   ③ 逐位惰性：pristine 臂写一次性 golden_p125/，探针臂与之逐位比 ⇒ 不过就是补丁本身错了。
# ⚠️ 补丁只落远端副本（p125_pin.py 在远端从 kernel_pristine.cpp 现生成）；本地提交源一个字不动。
# ⚠️ 退出 trap = 只做一次 npu.sh sync（不 build、不 sed）⇒ 远端回到"可发次态"（P128 的教训：
#    任何探针 build 都会重打 SoC sed，所以发次前必须最后再 sync 一次并核 sha256）。
# 用法：CASES="..." LADDER="1 2 4 8 16" ARMS="q w" bash p125_calib.sh
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

CASES="${CASES:-p1 p2 p4 p6 big1 w3 r64c65}"
REPS="${REPS:-5}"
LADDER="${LADDER:-1 2 4 8 16}"
ARMS="${ARMS:-q w}"
PRISTINE="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
[ "$(sha256sum "$PRISTINE" | cut -c1-16)" = "f815bf1eaba0f8bc" ] || { echo ">>> 本地提交源不是榜上那份 kernel ⇒ 中止"; exit 1; }

restore() {
  echo "=== 还原：只做 sync（远端源码 = 本地提交字节，无 sed、无 build） ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16; echo SEDHITS=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning | tail -2
}
trap restore EXIT

cat "$PRISTINE" | nssh "cat > ~/sfa_real/kernel_pristine.cpp"
( cd "$REPO/code 3/probes" && tar cf - p125_pin.py ) | nssh "cd ~/sfa_real && tar xf -"
echo ">>> pristine：本地=$(sha256sum "$PRISTINE" | cut -c1-16) 远端应同 ↓"
nssh "cd ~/sfa_real && sha256sum kernel_pristine.cpp | cut -c1-16" 2>&1 | grep -av Warning | tail -1

for ARM in $ARMS; do
  echo "############ ARM=$ARM（$( [ "$ARM" = q ] && echo 'prologue：LoadQ + 3×Duplicate' || echo 'epilogue：WriteOut 两段 DMA 尾巴' )）############"
  for N in 0 $LADDER; do
    echo "##########   $ARM N=$N ##########"
    nssh "cd ~/sfa_real && python3 p125_pin.py $N $ARM code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" 2>&1 | grep -av Warning | tail -2
    nssh "cd ~/sfa_real && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' code/CMakeLists.txt && \
      grep -q ascend910_93 code/op_host/sparse_flash_attention.cpp || \
      sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' code/op_host/sparse_flash_attention.cpp" >/dev/null
    nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error:' | head -5"
    W=$([ "$N" = "0" ] && echo 0 || echo 1); [ "$ARM" = w ] && [ "$N" != "0" ] && W=2
    nssh "cd ~/sfa_real/code && echo MARK=\$(grep -c 'P125 计量器' op_kernel/sparse_flash_attention.cpp) \
      REP=\$(grep -c 'p125r = 0u' op_kernel/sparse_flash_attention.cpp) \
      FLAG=\$(grep -c 'Flag<HardEvent' op_kernel/sparse_flash_attention.cpp) \
      MULS=\$(grep -c 'Muls(' op_kernel/sparse_flash_attention.cpp) \
      LOADQ=\$(grep -c 'LoadQ(q, kb, kr,' op_kernel/sparse_flash_attention.cpp) \
      FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0' op_kernel/sparse_flash_attention.cpp) \
      期望 MARK=$W REP=$W FLAG=36 MULS=11 LOADQ=1 FORBID=0" 2>&1 | grep -av Warning
    ALIVE=0
    for try in 1 2 3 4 5 6; do
      if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
      echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
    done
    [ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
    ACT=write; [ "$N" = "0" ] || ACT=diff
    nssh "$ENVR; for c in $CASES; do printf '  %-9s ' \$c; ./test_sfa_dev cases/\$c.bin 1 $ACT 2e-3 1e-2 golden_p125 2>&1 | grep -aoE '逐位一致 [^ ]+|不\*\*逐位一致\*\*|golden 写入 [^:]+: [A-Z]+|超差 [0-9]+' | tr '\n' ' '; echo; done" 2>&1 | grep -av "^Warning"
    nssh "$ENVR; for c in $CASES; do t=\$(timeout 1800 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-9s %10s\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning"
  done
done
echo P125_CALIB_DONE
