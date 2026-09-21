#!/bin/bash
# P20（稀疏下标整批入 UB）的**同场次 A/B**：交替推 pre_p20 备份 vs 当前提交源，
# 每边各测两遍 —— 跨场次比时间已经被 §15.38(a) 咬过一次，这次不再犯。
# 判据：SFA_PICK 必须两边一致（否则 2 KB 预算改了选档，时间不可比），时间看批量口径。
# ⚠️ 只测不改 golden：act=none。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
C="-F $HOME/.atomgitdevenv/.ssh/config"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
KER="code/op_kernel/sparse_flash_attention.cpp"
TIL="code/op_kernel/sparse_flash_attention_tiling.h"
HOST="code/op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
CASES="${CASES:-p1 p4 p6 q1h big1}"
BK="$REPO/code 3/probes/backup"
rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }

push_variant() {  # $1 = A(pre_p20) | B(p20)
    if [ "$1" = A ]; then
        cat "$BK/kernel_sparse_flash_attention.cpp.bak_pre_p20"  | rssh "cat > ~/sfa_real/$KER"
        cat "$BK/host_sparse_flash_attention.cpp.bak_pre_p20"    | rssh "cat > ~/sfa_real/$HOST"
        cat "$BK/tiling_h_sparse_flash_attention.bak_pre_p20"    | rssh "cat > ~/sfa_real/$TIL"
    else
        cat "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" | rssh "cat > ~/sfa_real/$KER"
        cat "$REPO/code 3/code/op_host/sparse_flash_attention.cpp"   | rssh "cat > ~/sfa_real/$HOST"
        cat "$REPO/code 3/code/op_kernel/sparse_flash_attention_tiling.h" | rssh "cat > ~/sfa_real/$TIL"
    fi
    printf '  远端 kernel md5: '; rssh "md5sum ~/sfa_real/$KER | cut -d' ' -f1"
    rssh "cd ~/sfa_real/code && \
        sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
        sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
        python3 /tmp/mixm0_host_patch.py"
    rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -5
}

measure() {  # $1 = 标签
    rssh "$ENVR; for n in $CASES; do printf '  $1 %-6s ' \"\$n\"; \
        SFA_MIXBD=0 SFA_FORCE_KS=${KSF:-2} timeout 300 ./test_sfa_dev cases/\$n.bin 8 none >/tmp/ab.txt 2>&1; \
        grep -aoE 'SFA_PICK.*|平均 [0-9.]+ ms' /tmp/ab.txt | head -2 | tr '\n' '|'; echo; done"
}

restore() { echo "=== 还原：当前提交源 + 干净构建 ==="; push_variant B >/dev/null 2>&1; \
    rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|逐位" | tail -3; }
trap restore EXIT

rssh "cat > /tmp/mixm0_host_patch.py" < "$REPO/code 3/probes/mixm0_host_patch.py"
for round in 1 2; do
  for v in A B; do
    echo "########## 轮 $round 变体 $v ($([ $v = A ] && echo pre_p20 || echo P20)) ##########"
    push_variant "$v" || { echo ">>> $v 构建失败"; continue; }
    measure "$v"
  done
done
