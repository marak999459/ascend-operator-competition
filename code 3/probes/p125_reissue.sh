#!/bin/bash
# P125 读完 ⇒ **回发 pristine**（这一族的臂全是"多加一遍活"的尺子，不可能成为最好字节 ⇒ 只有一条分支）。
# 链：backup/p125_pre_probe → 本地提交源（带 sha 断言）→ 命中数门（探针必须 0）→ 最后 sync（不再 build）
#     → 核四枚 sha + SoC 0 + 禁用词 0 → dry-run 逐个吻合 → submit
# ⚠️ 429：submit 前若刚发过，退避 ≥12 min；发完必须**当场核外部真值**（§7 #11）：
#     `p98_rank.py` 的 `sub` 变了没、六点到手没，⛔ 不许把"后台任务完成"当"发货成功"。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
K="$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
BK="$REPO/code 3/probes/backup/p125_pre_probe/sparse_flash_attention.cpp"
SHA='f815bf1eaba0f8bc'
FILES="op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp"
nssh() { timeout "${TMO:-900}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

[ "$(sha256sum "$BK" | cut -c1-16)" = "$SHA" ] || { echo ">>> 备份本身不是 pristine，拒绝覆盖"; exit 1; }
cp "$BK" "$K"
[ "$(sha256sum "$K" | cut -c1-16)" = "$SHA" ] || { echo ">>> 还原后 sha != pristine"; exit 1; }
echo "本地还原 OK：$(sha256sum "$K" | cut -c1-16,66-)"

echo "########## 最后一次 sync（此后不再 build）##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -2
( cd "$REPO/code 3/code" && sha256sum $FILES | cut -c1-16,66- )
nssh "cd ~/sfa_real/code && sha256sum $FILES | cut -c1-16,66- ; \
  echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ; \
  echo MARK=\$(grep -c 'P125 计量器' op_kernel/sparse_flash_attention.cpp) \
       REP=\$(grep -c 'p125r = 0u' op_kernel/sparse_flash_attention.cpp) \
       OLD=\$(grep -cE 'P12[0-4]|p12[0-4]|P11[0-9]|skP119|probeRep_' op_kernel/sparse_flash_attention.cpp) ; \
  echo FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
echo ">>> 期望 SOCEDEX=0 MARK=0 REP=0 OLD=0 FORBID=0；kernel=f815bf1e tiling= f2b28a86 / 1046b349 host=3a8f53050c8f15ae"

echo "########## dry-run ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -aE 'sha256|bytes' | cut -c1-100
echo "########## submit ##########"
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --max-wait 1200 | tee ~/sfa_real/p125_reissue.txt" 2>&1 | grep -av Warning
