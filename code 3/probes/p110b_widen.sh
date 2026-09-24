#!/bin/bash
# P110b：给"每头重做一次 K/K-rope 加宽"定价。ComputeScores 的 `for i in nbCur` 里那两条
# WidenToF32 因为**就地折叠**把 kf 用掉了 ⇒ 同一份 K 在 nb=4（P109 实测 w2/w3/w4/big1 全选 nb=4）
# 下要从 fp16 加宽 4 遍。
#   wid2   再加宽一遍（写回同一块 kf ⇒ 逐位惰性）⇒ 一遍加宽的**边际成本** S_w
#   hoist  i>0 时不加宽（**输出必错**，只看时间）⇒ 省 3 遍的**上界**
# 交叉校验：hoist 省的 ≈ 3·S_w ⇒ 对不上就说明加宽与后面的 Mul 有重叠、省下的时间不真能省。
# ⚠️ 补丁只落远端副本；每臂从 ../kernel_pristine.cpp 重生成；退出时 trap 还原干净构建。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
CASES="${CASES:-w3 w4 big1 w2 p6}"
REPS="${REPS:-5}"

soc_sed() {
  nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
}
restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; soc_sed
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 3 none" 2>&1 | grep -aE "超差|批量口径" | tail -2
}
trap restore EXIT

bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
soc_sed
nssh "cd ~/sfa_real && cp code/op_kernel/sparse_flash_attention.cpp ../kernel_pristine.cpp && \
  sha256sum ../kernel_pristine.cpp" 2>&1 | grep -av Warning | tail -1
( cd "$REPO/code 3/probes" && tar cf - p110b_widen.py ) | nssh "cd ~/sfa_real && tar xf -"

for M in ${MODES:-wid2 hoist rep1}; do
  echo "########## ARM $M ##########"
  # 命中数当门：wid2 标 2 行（kf+krf），hoist 只在 `if` 那行标 1 次，rep1 = 干净基线 0
  case "$M" in wid2) EXP=2 ;; hoist) EXP=1 ;; *) EXP=0 ;; esac
  nssh "cd ~/sfa_real && python3 p110b_widen.py $M code/op_kernel/sparse_flash_attention.cpp ../kernel_pristine.cpp && \
    echo HITS_MARK=\$(grep -c 'P110b' code/op_kernel/sparse_flash_attention.cpp) EXPECT=$EXP" 2>&1 | grep -av Warning | tail -2
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "${ALIVE:-0}" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -aE "超差" | head -3 | sed 's/^/  inert/'
  nssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
    export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
    for c in $CASES; do t=\$(timeout 900 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-14s %10s\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning"
done
