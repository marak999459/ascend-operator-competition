#!/bin/bash
# P108 标定：P107 那发探针（rows<=40 -> n_blk=16）读数被噪声吃掉了 —— 同一份 P105 字节
# 连发两遍的逐点散布就有 2.3~7.8 %（C4 13.56 vs 12.58），而探针的签名只有 +17~28 %。
# 所以把幅度做大到噪声够不着：n_blk = 8（chunk 数 ×3 于 48、×2 于 16，且 NBLK_MIN=16
# 保证平台的任何一档都 >8 ⇒ 对 rows<=40 的点是**可证明非惰性**的）。
# ⚠️ 代价：8 在 NBLK_MIN 之下，真机上从没跑过。host 侧注释（op_host:68-73）说下限的
#    真实理由是"行起点必须落在 32B 块上"= 8 个 float 的整数倍 ⇒ 8 理论安全，但要先
#    撞一发正确性再扫时间（下面第一件事就是两发快检，挂了就整批不跑）。
# 只跑在远端（由 p107_calib.sh 用 BODY=p108_calib_body.sh 推过来后调用）。
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 1
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
REPS="${REPS:-5}"

spec_of() {
  case "$1" in
    p1) echo 4:4 ;;  p2) echo 8:2 ;;  p4) echo 16:4 ;;  p6) echo 32:4 ;;
    pm41) echo 4:4 ;; pm81) echo 8:4 ;; pm161) echo 16:4 ;;
    p4s1c16384) echo 4:4 ;; p16s1c16384) echo 16:4 ;; p32s1c16384) echo 32:4 ;;
    p4s128c512) echo 4:4 ;;  p32s128c512) echo 32:4 ;;  p16s128c128) echo 16:4 ;;
    big1) echo 128:4 ;; w3) echo 128:4 ;; w4) echo 128:8 ;; w1) echo 4:4 ;; w2) echo 32:4 ;;
    *) echo "?:?" ;;
  esac
}

# arm <case> <ROWS> <K>   （ROWS=0 ⇒ 探针关）
arm() {
  local c="$1" rw="$2" kk="$3" out pk eff bm rows qn
  rows=$(spec_of "$c" | cut -d: -f1); qn=$(spec_of "$c" | cut -d: -f2)
  if [ "$rw" = "0" ]; then
    out=$(timeout 900 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  else
    out=$(SFA_PROBE_ROWS="$rw" SFA_PROBE_K="$kk" timeout 900 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  fi
  eff=$(printf '%s' "$out" | grep -a SFA_EFF | head -1 | sed 's/.*SFA_EFF //')
  bm=$(printf '%s' "$out" | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//')
  printf '%-14s r=%-4s n=%-2s %-11s 批量 %10s  eff[%s]\n' "$c" "$rows" "$qn" \
         "$([ "$rw" = 0 ] && echo off || echo "on/$rw/$kk")" "$bm" "$eff"
  printf '%s' "$out" | grep -aq "超差" && printf '%s' "$out" | grep -aE "超差 [1-9]" | head -2
}

# 0) 两发快检：8 这一档在真机上到底跑不跑得起来（ADDR_MISALIGN 会在第一发就炸）
echo "########## P108 前置快检：n_blk=8 可行性 ##########"
for c in p6 big1; do
  SFA_PROBE_ROWS=100000 SFA_PROBE_K=8 timeout 300 ./test_sfa_dev "cases/$c.bin" 1 diff 2>&1 \
    | grep -aE 'SFA_EFF|超差|ADDR|ERR|FAIL' | head -5
done
if SFA_PROBE_ROWS=100000 SFA_PROBE_K=8 timeout 300 ./test_sfa_dev cases/p6.bin 1 none 2>&1 \
     | grep -aq 批量口径; then echo ">>> n_blk=8 可跑 ⇒ 继续扫"; else echo ">>> n_blk=8 跑不起来 ⇒ 中止"; exit 1; fi

ALL="p1 p2 p4 p6 pm41 pm81 pm161 w1 w2 big1 w3 w4"
echo "########## P108 幅度标定：off vs n_blk=8 vs n_blk=16（ROWS=100000 ⇒ 全部行都夹） ##########"
for c in ${ONLY:-$ALL}; do
  arm "$c" 0 0
  arm "$c" 100000 8
  arm "$c" 100000 16
  echo "---"
done

echo "########## n_blk=8 下的正确性（判据只认 超差；golden 逐位一致在换 chunk 宽度后必挂） ##########"
for c in r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads \
         p1 p2 p4 p6 big1 w3 q1h q2h q3h p_n64 p_n512 p_n1024 \
         e1empty e2one e3two e4odd e5s2one e6many e7padq e8padkv; do
  for f in 0 1; do
    tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
    line=$(SFA_PROBE_ROWS=100000 SFA_PROBE_K=8 env $([ "$f" = 1 ] && echo SFA_F32=1) \
           timeout 300 ./test_sfa_dev "cases/$c.bin" 1 diff 2>&1 \
           | grep -aoE '超差 [0-9]+/[0-9]+' | tr '\n' ' ')
    printf '%-10s %-5s %s\n' "$c" "$tag" "$line"
  done
done
echo "P108_CALIB_DONE"
