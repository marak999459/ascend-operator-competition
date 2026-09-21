#!/bin/bash
# P21（扫描前置到上一次 flush 的阴影里）的**同场次 A/B**：交替推 pre_p21 备份 vs 当前提交源。
# 纪律照 §15.38(a)/§15.40：跨场次绝不比时间 —— A、B 各测两轮、同一场次内交替。
# 判据：① SFA_PICK 两边必须一致（选档变了时间不可比）② 时间只看批量口径 ③ act=none，绝不动 golden。
# ⚠️ mixm0_host_patch.py 用相对路径 code/op_host/… ⇒ 必须在 cwd=~/sfa_real 下执行（§15.40 的坑）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
C="-F $HOME/.atomgitdevenv/.ssh/config"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
KER="code/op_kernel/sparse_flash_attention.cpp"
TIL="code/op_kernel/sparse_flash_attention_tiling.h"
HOST="code/op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
CASES="${CASES:-p1 p2 p4 p6 q1h big1}"
BK="$REPO/code 3/probes/backup"
SRC="$REPO/code 3/code"
rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }

push_variant() {  # $1 = A(pre_p21 备份) | B(当前提交源)
    if [ "$1" = A ]; then
        cat "$BK/kernel_sparse_flash_attention.cpp.bak_pre_p21" | rssh "cat > ~/sfa_real/$KER"
        cat "$BK/host_sparse_flash_attention.cpp.bak_pre_p21"    | rssh "cat > ~/sfa_real/$HOST"
        cat "$BK/tiling_h_sparse_flash_attention.bak_pre_p21"    | rssh "cat > ~/sfa_real/$TIL"
    else
        cat "$SRC/op_kernel/sparse_flash_attention.cpp"       | rssh "cat > ~/sfa_real/$KER"
        cat "$SRC/op_host/sparse_flash_attention.cpp"         | rssh "cat > ~/sfa_real/$HOST"
        cat "$SRC/op_kernel/sparse_flash_attention_tiling.h"  | rssh "cat > ~/sfa_real/$TIL"
    fi
    printf '  远端 md5 kernel/host/tiling: '
    rssh "cd ~/sfa_real && md5sum $KER $HOST $TIL | cut -d' ' -f1 | tr '\n' ' '"
    echo
    # SoC 双注册是远端副本专属（提交源只有 ascend910b）；补丁必须在 ~/sfa_real 下跑
    rssh "cd ~/sfa_real/code && \
        sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
        sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp; \
        cd ~/sfa_real && python3 /tmp/mixm0_host_patch.py"
    rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -5
}

measure() {  # $1 = 标签
    rssh "$ENVR; for n in $CASES; do printf '  $1 %-6s ' \"\$n\"; \
        SFA_MIXBD=0 SFA_FORCE_KS=${KSF:-2} timeout 300 ./test_sfa_dev cases/\$n.bin 8 none >/tmp/ab.txt 2>&1; \
        grep -aoE 'SFA_PICK.*|平均 [0-9.]+ ms' /tmp/ab.txt | head -2 | tr '\n' '|'; echo; done"
}

restore() { echo "=== 还原：当前提交源 + 干净构建 + 正确性复验 ==="; push_variant B >/dev/null 2>&1; \
    rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|逐位" | tail -3; }
trap restore EXIT

rssh "cat > /tmp/mixm0_host_patch.py" < "$REPO/code 3/probes/mixm0_host_patch.py"
for round in 1 2; do
  for v in A B; do
    echo "########## 轮 $round 变体 $v ($([ $v = A ] && echo pre_p21 || echo P21)) ##########"
    push_variant "$v" || { echo ">>> $v 构建失败"; continue; }
    measure "$v"
  done
done
