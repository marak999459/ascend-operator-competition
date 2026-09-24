#!/bin/bash
# P112 探针（无条件把 ComputeScores 原地再跑一遍）→ 上平台前的构建 + 数值闸门。
# 两道门：① 远端旋钮命中数（§7 第 2 条）② GATE1 的 13 个真 expect 档必须逐位一致
# —— 这一发尤其重要，因为 P112 在 sbs=1 的档（p6/big1）上**真的会执行**（P111 在那儿是惰性的），
# 逐位一致就是"增量只买时间、不买输出"的直接证明。
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
# 命中数当门：FlushChunk 里那句调用必须出现 2 次（原有 1 + 探针 1），P112 注释 1 次
nssh "cd ~/sfa_real && echo \"HITS_CALL=\$(grep -c 'ComputeScores(q, kb, kr, sc, nbCur, m);' code/op_kernel/sparse_flash_attention.cpp) HITS_P112=\$(grep -c 'P112 探针' code/op_kernel/sparse_flash_attention.cpp) HITS_P111=\$(grep -c probeRep_ code/op_kernel/sparse_flash_attention.cpp)\"" | grep -av Warning
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
bash "$REPO/code 3/probes/p32_gate.sh"
