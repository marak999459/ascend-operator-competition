#!/bin/bash
# P116 发次前置：把 softmax 段 rep=2 铺进**本地提交源**（探针要真的发上平台）→ sync → sed → build
# → 命中数门（§7 第 2 条）→ p32_gate 数值闸门（真 golden 双 dtype）。
# ⚠️ 从 pristine（= P105 字节 f815bf1e）重生成，不叠在已有探针上；铺完的本地源就是待发字节。
# 退出 trap 不做还原：探针态是要发出去的东西，还原由 p116_reissue.sh 负责。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1800}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
K="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
PRIST="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"

echo "########## 0) pristine 校验 ##########"
sha256sum "$PRIST" "$K" | cut -c1-16,76-
[ "$(sha256sum "$K" | cut -c1-16)" = "$(sha256sum "$PRIST" | cut -c1-16)" ] || \
  { echo ">>> 本地源 != pristine(P105) ⇒ 中止（先还原再铺探针，避免叠补丁）"; exit 1; }

echo "########## 1) 本地铺 rep=2 ##########"
python3 "$REPO/code 3/probes/p116_softmax_rep.py" 2 "$K" "$PRIST"
echo "本地 HITS_P116=$(grep -c 'P116 softmax-rep' "$K")  循环头=$(grep -c 'for (uint32_t rP116_' "$K")  源码行数=$(wc -l < "$K") (pristine $(wc -l < "$PRIST"))"
[ "$(grep -c 'P116 softmax-rep' "$K")" = "1" ] || { echo ">>> 标记命中数 != 1 ⇒ 中止"; exit 1; }
[ "$(grep -c 'for (uint32_t rP116_' "$K")" = "1" ] || { echo ">>> 循环头 != 1 ⇒ 中止"; exit 1; }

echo "########## 2) sync + sed + build ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -2
nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16,66-" 2>&1 | grep -av Warning
nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp" >/dev/null
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"

echo "########## 3) 远端命中数门 ##########"
nssh "cd ~/sfa_real/code && echo HITS_P116=\$(grep -c 'P116 softmax-rep' op_kernel/sparse_flash_attention.cpp) \
  LOOP=\$(grep -c 'for (uint32_t rP116_' op_kernel/sparse_flash_attention.cpp) \
  P115=\$(grep -c 'P115 split\|pn_/N115\|N115u' op_kernel/sparse_flash_attention.cpp) \
  P114=\$(grep -c 'P114\|p114\|P113' op_kernel/sparse_flash_attention.cpp) \
  PRINTF=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0' op_kernel/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning

echo "########## 4) p32_gate 数值闸门 ##########"
bash "$REPO/code 3/probes/p32_gate.sh" 2>&1 | tail -45
echo "P116_PROBE_DONE"
