#!/bin/bash
# P98：把一个本地目录（内含 code/ 子目录，即"提交态"的完整一份）推到远端 ~/sfa_real/code，
# 并当场打印远端四枚 sha256 核对手递进来的期望值。
#
# 为什么要单独一个脚本：§7 第 9 条 —— argv 单参数 128 KB 上限、tar 顶层名写错会静默落错
# 目录（P98 这一轮就是这么把 A′ 字节撒到 ~/sfa_real/sfa_real/ 的）。这里把顶层名钉死。
#
# 用法： bash code\ 3/probes/p98_pushcode.sh <staging父目录> [期望sha256前缀...]
#        期望前缀逐个 grep 命中才算成功（远端旋钮必须把命中数当门 —— §7 第 2 条）。
set -eu
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
SRC="${1:?用法: p98_pushcode.sh <含 code/ 的目录> [期望sha256前缀...]}"
shift || true
[ -d "$SRC/code/op_kernel" ] || { echo "!! $SRC/code/op_kernel 不存在"; exit 1; }

echo "=== 本地四枚 ==="
( cd "$SRC/code" && sha256sum op_kernel/sparse_flash_attention.cpp \
    op_kernel/sparse_flash_attention_tiling.h \
    op_kernel/tiling_key_sparse_flash_attention.h \
    op_host/sparse_flash_attention.cpp )
( cd "$SRC" && tar cf - code ) | timeout 300 ssh -F $S -o ConnectTimeout=25 $H \
  'cd ~/sfa_real && tar xf - && echo "=== 远端四枚 ===" && cd code && sha256sum \
     op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h \
     op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp' 2>&1 | grep -av Warning

REM=$(timeout 200 ssh -F $S $H 'cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp' 2>&1 | grep -av Warning)
LOC=$( cd "$SRC/code" && sha256sum op_kernel/sparse_flash_attention.cpp op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h op_host/sparse_flash_attention.cpp )
if [ "$REM" = "$LOC" ]; then echo "=== 逐字节一致 ✓ ==="; else echo "=== !! 本地/远端不一致 !! ==="; diff <(echo "$LOC") <(echo "$REM"); exit 1; fi
for hx in "$@"; do
  n=$(printf '%s' "$REM" | grep -c "$hx")
  echo "期望 $hx 命中 $n $( [ "$n" = 1 ] && echo '✓' || echo '✗ 门未过' )"
  [ "$n" = 1 ] || exit 1
done
echo "=== 远端 SoC sed 命中（提交态须为 0）==="
timeout 120 ssh -F $S $H 'cd ~/sfa_real/code && grep -c ascend910_93 CMakeLists.txt op_host/sparse_flash_attention.cpp || true' 2>&1 | grep -av Warning
