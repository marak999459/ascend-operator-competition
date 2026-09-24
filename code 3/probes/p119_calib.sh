#!/bin/bash
# P119 本地标定：四臂 = pristine / PV×2 / PV×3 / PV×5（同一份 pristine 现生成，只差重跑遍数 N）。
# 目的：① 惰性自证（三臂输出与 pristine 逐位一致，8 档 × out/max/sum）；② 本地 PV 段价目（Δ vs N 是否 1:1 涨）。
# ⚠️ 补丁只落远端副本；退出 trap = 只做一次 npu.sh sync（不 build、不 sed）⇒ 远端回到可发次态。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

CASES="${CASES:-g64c1 g64c4 g64c16 v64c8 r64c65 p6 big1 w3}"
REPS="${REPS:-5}"
ARMS="${ARMS:-0 1 2 4}"
PRISTINE="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"

restore() {
  echo "=== 还原：只做 sync（远端源码 = 本地提交字节，无 sed、无 build） ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16; echo SEDHITS=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning | tail -2
}
trap restore EXIT

sha_local=$(sha256sum "$PRISTINE" | cut -c1-16)
cat "$PRISTINE" | nssh "cat > ~/sfa_real/kernel_pristine.cpp"
( cd "$REPO/code 3/probes" && tar cf - p119_pvrep.py ) | nssh "cd ~/sfa_real && tar xf -"
echo ">>> pristine sha 本地=$sha_local 远端应同"
nssh "cd ~/sfa_real && sha256sum kernel_pristine.cpp | cut -c1-16" 2>&1 | grep -av Warning | tail -1

for N in $ARMS; do
  echo "########## ARM N=$N (PV x$((N+1))) ##########"
  nssh "cd ~/sfa_real && python3 p119_pvrep.py $N code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" 2>&1 | grep -av Warning | tail -1
  nssh "cd ~/sfa_real && echo SED_TMP=\$(grep -c ascend910_93 code/op_host/sparse_flash_attention.cpp) && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' code/CMakeLists.txt && \
    grep -q ascend910_93 code/op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' code/op_host/sparse_flash_attention.cpp" >/dev/null
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
  WANT=$((N>0 ? 2 : 0))
  nssh "cd ~/sfa_real/code && echo HITS_P119=\$(grep -c 'P119 PV meter' op_kernel/sparse_flash_attention.cpp) 期望=$WANT" 2>&1 | grep -av Warning
  ALIVE=0
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
  # 惰性自证：pristine 臂写一次性 golden_p119/，后三臂与之逐位比
  ACT=write; [ "$N" = "0" ] || ACT=diff
  nssh "$ENVR; for c in $CASES; do printf '%-11s ' \$c; ./test_sfa_dev cases/\$c.bin 1 $ACT 2e-3 1e-2 golden_p119 2>&1 | grep -aoE '逐位一致 [^ ]+|不\*\*逐位一致\*\*|golden 写入 [^:]+: [A-Z]+' | tr '\n' ' '; echo; done" 2>&1 | grep -av "^Warning"
  nssh "$ENVR; for c in $CASES; do t=\$(timeout 1800 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-11s %10s\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning"
done
echo P119_CALIB_DONE
