#!/bin/bash
# P114 本地标定：同一份 kernel 的两臂（N=1 基线 / N=2 gather 翻倍），读"本地靶族的 gather 占比"。
# 平台那一侧的 Δ% 由 p114 发次给；两边一对才知道 P112 那个"score 段只占 6 %"是不是搬运不对称。
# ⚠️ 补丁只落远端副本；退出 trap = 只做一次 npu.sh sync（**不 build、不 sed**）⇒ 远端源码回到
#    本地提交字节态，发次链可以直接接手（§7.5 要求最后一次 sync 之后不再 build）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
CASES="${CASES:-w3 w4 big1 w2 p6}"
REPS="${REPS:-5}"
PRISTINE="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"

restore() {
  echo "=== 还原：只做 sync（远端源码 = 本地提交字节，无 sed、无 build） ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16; echo SEDHITS=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning | tail -2
}
trap restore EXIT

# pristine = 未打探针的 P105 字节，从本地备份走 stdin 推（argv 上限，§push-bytes-via-stdin）
sha_local=$(sha256sum "$PRISTINE" | cut -c1-16)
cat "$PRISTINE" | nssh "cat > ~/sfa_real/kernel_pristine.cpp"
( cd "$REPO/code 3/probes" && tar cf - p114_gather_rep.py ) | nssh "cd ~/sfa_real && tar xf -"
nssh "cd ~/sfa_real && sha256sum kernel_pristine.cpp | cut -c1-16" 2>&1 | grep -av Warning | tail -1
echo ">>> pristine sha 本地=$sha_local 远端应同"

for N in ${NS:-1 2}; do
  echo "########## ARM gather=$N ##########"
  nssh "cd ~/sfa_real && python3 p114_gather_rep.py $N code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" 2>&1 | grep -av Warning | tail -1
  nssh "cd ~/sfa_real && echo SED_TMP=\$(grep -c ascend910_93 code/op_host/sparse_flash_attention.cpp) && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' code/CMakeLists.txt && \
    grep -q ascend910_93 code/op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' code/op_host/sparse_flash_attention.cpp" >/dev/null
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
  ALIVE=0
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
  # 惰性自证：N=2 那臂的输出必须仍与 golden 逐位一致（探针只买时间不买输出）
  nssh "$ENVR; for c in p6 big1 r2_chunk; do printf '  inert %-9s ' \$c; ./test_sfa_dev cases/\$c.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|不\*\*逐位一致\*\*' | tr '\n' ' '; echo; done" 2>&1 | grep -av Warning
  nssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
    export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
    for c in $CASES; do t=\$(timeout 900 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-14s %10s\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning"
done
