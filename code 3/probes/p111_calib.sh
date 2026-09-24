#!/bin/bash
# P111 探针标定：同一份字节的两臂（probe vs pristine）× sbs 阶梯 ⇒ 读出
#   sbs=1 → +0 % / 2..8 → +25 % / 16..64 → +50 % / 128 → +75 %
# 外加逐位惰性自证（probe 臂的 p6 diff 必须与 pristine 臂**完全相同**的那行）。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
# case:sbs 期望
CASES="${CASES:-big1:1 w3:2 p1s8:8 k_n32s32:32 k_n16s64:64 p4s128c512:128 p16s128c128:128 p6:2}"
REPS="${REPS:-5}"
soc_sed() {
  nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
}
restore() {
  echo "=== 还原干净构建 ==="
  nssh "cd ~/sfa_real && cp ../kernel_pristine.cpp code/op_kernel/sparse_flash_attention.cpp"
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; soc_sed
  nssh "cd ~/sfa_real && cp ../kernel_pristine.cpp code/op_kernel/sparse_flash_attention.cpp && \
    grep -c probeRep_ code/op_kernel/sparse_flash_attention.cpp; $ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
}
trap restore EXIT

# pristine 快照（若上一轮留下的还在，就用 sha256 对一下当前提交源）
bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; soc_sed
nssh "cd ~/sfa_real && [ -f ../kernel_pristine.cpp ] || cp code/op_kernel/sparse_flash_attention.cpp ../kernel_pristine.cpp; sha256sum ../kernel_pristine.cpp" 2>&1 | grep -av Warning | tail -1

for ARM in probe pristine; do
  echo "########## ARM $ARM ##########"
  if [ "$ARM" = pristine ]; then
    nssh "cd ~/sfa_real && cp ../kernel_pristine.cpp code/op_kernel/sparse_flash_attention.cpp && grep -c probeRep_ code/op_kernel/sparse_flash_attention.cpp"
  else
    nssh "cd ~/sfa_real && grep -c probeRep_ code/op_kernel/sparse_flash_attention.cpp"   # 命中数当门，期望 4
  fi
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "${ALIVE:-0}" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -aE "超差" | sed "s/^/  [$ARM diff] /"
  nssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
    export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
    for cs in $CASES; do c=\${cs%:*}; t=\$(timeout 900 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-16s sbs=%-4s %10s\n' \$c \${cs#*:} \"\$t\"; done" 2>&1 | grep -av "^Warning"
done
