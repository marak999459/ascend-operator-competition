#!/bin/bash
# P32 干净构建复验 + golden 重锁（远端执行）
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 2
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
GOLD="r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads p1 p2 p4 p6 big1"
echo "=== V1) 与 P29 golden 差分（重锁前最后一次的对照）==="
for f in 0 1; do
  tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
  for cs in $GOLD; do
    if [ "$f" = 1 ]; then out=$(env SFA_F32=1 ./test_sfa_dev cases/$cs.bin 1 diff 2>&1); else out=$(./test_sfa_dev cases/$cs.bin 1 diff 2>&1); fi
    bad=$(printf '%s' "$out" | grep -ac "不\*\*逐位一致\*\*")
    tol=$(printf '%s' "$out" | grep -aoE "超差 [0-9]+/[0-9]+" | head -1)
    printf '%-5s %-11s 非逐位=%s %s\n' "$tag" "$cs" "$bad" "$tol"
  done
done
echo "=== V2) 计时复验（模型自选档）==="
for cs in p1 p4 p6 big1; do ./test_sfa_dev cases/$cs.bin 5 none 2>&1 | grep -aoE "平均 [0-9.]+ ms" | head -1 | sed "s/^/$cs /"; done
echo "=== V3) golden 重锁（旧基线整份备份到 golden_bak_p29）==="
if [ ! -d golden_bak_p29 ]; then cp -a golden golden_bak_p29 && echo "已备份 golden_bak_p29 ($(ls golden_bak_p29 | wc -l) 个文件)"; else echo "golden_bak_p29 已存在，跳过备份"; fi
for cs in $GOLD; do ./test_sfa_dev cases/$cs.bin 1 write >/dev/null 2>&1; env SFA_F32=1 ./test_sfa_dev cases/$cs.bin 1 write >/dev/null 2>&1; done
echo "=== V4) 重锁后复验（应全为 非逐位=0）==="
for f in 0 1; do
  tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
  for cs in $GOLD; do
    if [ "$f" = 1 ]; then out=$(env SFA_F32=1 ./test_sfa_dev cases/$cs.bin 1 diff 2>&1); else out=$(./test_sfa_dev cases/$cs.bin 1 diff 2>&1); fi
    bad=$(printf '%s' "$out" | grep -ac "不\*\*逐位一致\*\*")
    printf '%-5s %-11s 非逐位=%s\n' "$tag" "$cs" "$bad"
  done
done
echo P32_VERIFY_DONE
