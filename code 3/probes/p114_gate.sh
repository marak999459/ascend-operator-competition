#!/bin/bash
# P114 探针（把每 chunk 三条 gather DataCopy 原地再发一遍）→ 上平台前的构建 + 数值闸门。
# 与 P112 同一种机制：同 src 同 dst ⇒ 输出逐位不变、只多花时间 ⇒
# "逐位一致"就是"增量只买时间、不买输出"的直接证明。
# ⚠️ 命中数门（§7 第 2 条）：kb/vb 那两条必须各 2 次、kr 1 次 + 探针 1 次 = 2 次，
#    P114 注释 2 处，P112/P111 残留 0。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1200}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
nssh "cd ~/sfa_real && echo \"HITS_KB=\$(grep -c 'CopyGm2Ub(kb\[done \* D_\]' code/op_kernel/sparse_flash_attention.cpp) HITS_KR=\$(grep -c 'CopyGm2Ub(kr\[done \* Dr_\]' code/op_kernel/sparse_flash_attention.cpp) HITS_VB=\$(grep -c 'CopyGm2Ub(vb\[done \* D_\]' code/op_kernel/sparse_flash_attention.cpp) HITS_P114=\$(grep -c 'P114' code/op_kernel/sparse_flash_attention.cpp) HITS_P112=\$(grep -c 'P112 探针' code/op_kernel/sparse_flash_attention.cpp) HITS_P111=\$(grep -c probeRep_ code/op_kernel/sparse_flash_attention.cpp)\"" | grep -av Warning
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
bash "$REPO/code 3/probes/p32_gate.sh"
