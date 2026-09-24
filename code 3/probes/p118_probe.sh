#!/bin/bash
# P118 发次前置：把 gather 计量器（MODE=cold|warm，N=1）铺进**本地提交源** → sync → sed → build
# → 命中数门（§7 第 2 条）→ p32_gate 数值闸门（真 golden 双 dtype）。
# ⚠️ 从 pristine（= P105 字节 f815bf1e）重生成，不叠在已有探针上；铺完的本地源就是待发字节。
# 退出 trap 不做还原：探针态是要发出去的东西，还原由 p118_reissue.sh 负责。
set -u
MODE="${1:-cold}"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1800}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
K="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
PRIST="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"
case "$MODE" in cold|warm) ;; *) echo ">>> MODE 必须 cold|warm"; exit 1;; esac

echo "########## 0) pristine 校验 (MODE=$MODE) ##########"
sha256sum "$PRIST" "$K" | cut -c1-16,76-
[ "$(sha256sum "$K" | cut -c1-16)" = "$(sha256sum "$PRIST" | cut -c1-16)" ] || \
  { echo ">>> 本地源 != pristine(P105) ⇒ 中止（先跑 p118_reissue.sh 还原，避免叠补丁）"; exit 1; }

echo "########## 1) 本地铺 N=1 MODE=$MODE ##########"
python3 "$REPO/code 3/probes/p118_coldcopy.py" 1 "$MODE" "$K" "$PRIST"
echo "本地 MARK=$(grep -c 'P118 gather meter' "$K") 目的条=$(grep -c 'CopyGm2Ub(dkP118' "$K") 栅栏=$(grep -c 'SetFlag<HardEvent::MTE2_V>(2)' "$K") 行差=$(($(wc -l < "$K") - $(wc -l < "$PRIST")))"
[ "$(grep -c 'P118 gather meter' "$K")" = "2" ] || { echo ">>> 标记命中数 != 2 ⇒ 中止"; exit 1; }
[ "$(grep -c 'CopyGm2Ub(dkP118' "$K")" = "2" ] || { echo ">>> 目的条 != 2（k 与 v）⇒ 中止"; exit 1; }
[ "$(grep -c 'SetFlag<HardEvent::MTE2_V>(2)' "$K")" = "1" ] || { echo ">>> 栅栏 != 1 ⇒ 中止"; exit 1; }

echo "########## 2) sync + sed + build ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -2
nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16,66-" 2>&1 | grep -av Warning
nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp" >/dev/null
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"

echo "########## 3) 远端命中数门 ##########"
nssh "cd ~/sfa_real/code && echo MARK=\$(grep -c 'P118 gather meter' op_kernel/sparse_flash_attention.cpp) \
  DSTK=\$(grep -c 'CopyGm2Ub(dkP118' op_kernel/sparse_flash_attention.cpp) \
  BAR=\$(grep -c 'SetFlag<HardEvent::MTE2_V>(2)' op_kernel/sparse_flash_attention.cpp) \
  MODE=$MODE \
  OLD=\$(grep -c 'P115 split\|P116 softmax-rep\|P114\|P112 探针\|probeRep_\|P110 rep' op_kernel/sparse_flash_attention.cpp) \
  PRINTF=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0' op_kernel/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning

echo "########## 4) p32_gate 数值闸门 ##########"
bash "$REPO/code 3/probes/p32_gate.sh" 2>&1 | tail -45
echo "P118_PROBE_DONE MODE=$MODE"
