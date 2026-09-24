#!/bin/bash
# P121 探针读完 ⇒ 回发当前最好的字节（P105，latest 计分）。
# 链：本地还原 → 最后 sync（不再 build）→ 核四枚 sha + SoC 0 + 禁用词 0 → dry-run 逐个吻合 → submit
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
PID=6a7c22d6a52e0f540a8a098d
CLI=/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit/cannjudge_cli.py
nssh() { timeout "${TMO:-900}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
cp "$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp" "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp"
echo "本地 kernel sha = $(sha256sum "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" | cut -c1-16)  期望 f815bf1eaba0f8bc"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | tail -1
nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16,66- ; echo SOCEDEX=\$(grep -rc ascend910_93 op_host/ op_kernel/ CMakeLists.txt | grep -v ':0' | wc -l) ALLMARK=\$(grep -c 'P112\|P114\|P115\|P116\|P118\|P119\|P120\|P121\|skP119\|rP119\|SetFlag<HardEvent::MTE2_V>(4)\|p121\|probeRep_' op_kernel/sparse_flash_attention.cpp) FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_kernel/sparse_flash_attention.cpp op_host/sparse_flash_attention.cpp | tr '\n' ' ')" 2>&1 | grep -av Warning
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --dry-run" 2>&1 | grep -aE 'sha256|bytes' | cut -c1-100
nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 $CLI submit --problem-id $PID --project-dir ~/sfa_real/code --max-wait 600" 2>&1 | grep -av Warning | cut -c1-90
