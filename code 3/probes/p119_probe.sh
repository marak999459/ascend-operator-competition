#!/bin/bash
# P119 发次前置：把 PV 计量器（额外重跑 N 遍，目的端 = kfBuf_ 末行）铺进**本地提交源** → sync → sed
# → build → 命中数门（§7 第 2 条）→ p32_gate 数值闸门（真 golden 双 dtype）。
# ⚠️ 从 pristine（= P105 字节 f815bf1e）重生成，不叠在已有探针上；铺完的本地源就是待发字节。
# 退出 trap 不做还原：探针态是要发出去的东西，还原由 p119_reissue.sh 负责。
set -u
N="${1:-1}"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1800}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
K="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
PRIST="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"
case "$N" in 1|2|3|4|5|6) ;; *) echo ">>> N 必须 1..6"; exit 1;; esac

echo "########## 0) pristine 校验 (N=$N) ##########"
sha256sum "$PRIST" "$K" | cut -c1-16,76-
[ "$(sha256sum "$K" | cut -c1-16)" = "$(sha256sum "$PRIST" | cut -c1-16)" ] || \
  { echo ">>> 本地源 != pristine(P105) ⇒ 中止（先跑 p119_reissue.sh 还原，避免叠补丁）"; exit 1; }

echo "########## 1) 本地铺 N=$N ##########"
python3 "$REPO/code 3/probes/p119_pvrep.py" "$N" "$K" "$PRIST"
echo "本地 MARK=$(grep -c 'P119 PV meter' "$K") 复制体Axpy=$(grep -c 'Axpy<float, float>(skP119' "$K") 真Axpy=$(grep -c 'Axpy<float, float>(o\[i \* rowC\], vt,' "$K") REP=$(grep -c 'for (uint32_t rP119' "$K") 新buffer=$(grep -c 'InitBuffer' "$K")(应=$(grep -c 'InitBuffer' "$PRIST")) 行差=$(($(wc -l < "$K") - $(wc -l < "$PRIST")))"
[ "$(grep -c 'P119 PV meter' "$K")" = "2" ] || { echo ">>> 标记命中数 != 2 ⇒ 中止"; exit 1; }
[ "$(grep -c 'Axpy<float, float>(skP119' "$K")" = "1" ] || { echo ">>> 复制体 Axpy != 1 ⇒ 中止"; exit 1; }
[ "$(grep -c 'Axpy<float, float>(o\[i \* rowC\], vt,' "$K")" = "1" ] || { echo ">>> 真那段被动了 ⇒ 中止"; exit 1; }
[ "$(grep -c 'for (uint32_t rP119' "$K")" = "1" ] || { echo ">>> 重跑循环 != 1 ⇒ 中止"; exit 1; }
[ "$(grep -c 'InitBuffer' "$K")" = "$(grep -c 'InitBuffer' "$PRIST")" ] || { echo ">>> UB 预算动了（新增/删除 buffer）⇒ 中止"; exit 1; }

echo "########## 2) sync + sed + build ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -2
nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16,66-" 2>&1 | grep -av Warning
nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp" >/dev/null
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"

echo "########## 3) 远端命中数门 ##########"
nssh "cd ~/sfa_real/code && echo MARK=\$(grep -c 'P119 PV meter' op_kernel/sparse_flash_attention.cpp) \
  SK=\$(grep -c 'Axpy<float, float>(skP119' op_kernel/sparse_flash_attention.cpp) \
  REAL=\$(grep -c 'Axpy<float, float>(o\[i \* rowC\], vt,' op_kernel/sparse_flash_attention.cpp) \
  REP=\$(grep -c 'rP119 < ${N}u' op_kernel/sparse_flash_attention.cpp) N=$N \
  OLD=\$(grep -c 'P115 split\|P116 softmax-rep\|P114\|P112 探针\|probeRep_\|P110 rep\|P118 gather meter' op_kernel/sparse_flash_attention.cpp) \
  PRINTF=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0' op_kernel/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning

echo "########## 4) p32_gate 数值闸门 ##########"
bash "$REPO/code 3/probes/p32_gate.sh" 2>&1 | tail -45
echo "P119_PROBE_DONE N=$N"
