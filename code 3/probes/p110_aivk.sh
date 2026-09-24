#!/bin/bash
# P110「AIV 到没到向量墙」消融：同一份 kernel 把 ComputeScores 重跑 1/2/4 遍（**逐位惰性的**
# 计算量扰动），在 rows>=41 那一族上读时间 ⇒ 解出 t1 = A + S 里的 S。
#   S/t1 ≈ 申报的 51~55 % ⇒ 向量墙（要再快得改精度/架构）
#   S/t1 ≪ 51 %        ⇒ 流水线有空隙（微优化还有肉）
# ⚠️ 补丁只落远端副本；退出时 trap 还原干净构建。
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
( cd "$REPO/code 3/probes" && tar cf - p110_score_rep.py ) | nssh "cd ~/sfa_real && tar xf -"

for N in ${NS:-1 2 4}; do
  echo "########## ARM rep=$N ##########"
  # 每臂都从 pristine 重新生成 ⇒ 不会叠补丁；命中数当门（§7.2）
  nssh "cd ~/sfa_real && python3 p110_score_rep.py $N code/op_kernel/sparse_flash_attention.cpp ../kernel_pristine.cpp && \
    echo HITS_REP=\$(grep -c 'P110 rep' code/op_kernel/sparse_flash_attention.cpp) EXPECT=$((N-1))" 2>&1 | grep -av Warning | tail -2
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "${ALIVE:-0}" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
  # 惰性自证：rep>1 那一遍的 p6 输出必须与 golden 逐位差 == 基线那一发
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -aE "超差" | head -3 | sed 's/^/  inert/'
  nssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
    export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
    for c in $CASES; do t=\$(timeout 900 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-14s %10s\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning"
done
