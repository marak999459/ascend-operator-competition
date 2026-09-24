#!/bin/bash
# P107 探针的**本地标定**：把"只按 rows 夹 n_blk"这一扰动在真机上量出倍数，
# 并且证明它在 rows>THRESH 的那一族上【逐字节不触发】。
# ⚠️ 只跑在远端（由 p107_calib.sh 推过来后调用）。
set -u
source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
cd ~/sfa_real || exit 1
export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom
export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH
REPS="${REPS:-5}"

# rows:  1/4/16/32 的 token 阶梯（每行 16384 token，最接近按 8~15 ms 反解出来的量级）
#        + p1~p6（每行 2048）+ big1/w3/w4（rows>=128）+ sbs=128 两档
spec_of() {
  case "$1" in
    p1) echo 4:4 ;;  p2) echo 8:2 ;;  p4) echo 16:4 ;;  p6) echo 32:4 ;;
    pm41) echo 4:4 ;; pm81) echo 8:4 ;; pm161) echo 16:4 ;;
    p1s1c16384) echo 4:1 ;;  p4s1c16384) echo 4:4 ;;
    p16s1c16384) echo 16:4 ;; p32s1c16384) echo 32:4 ;;
    p4s128c512) echo 4:4 ;;  p32s128c512) echo 32:4 ;;  p16s128c128) echo 16:4 ;;
    big1) echo 128:4 ;; w3) echo 128:4 ;; w4) echo 128:8 ;; w1) echo 4:4 ;; w2) echo 32:4 ;;
    *) echo "?:?" ;;
  esac
}

# arm <case> <PROBE_ROWS> <PROBE_K>   （ROWS=0 且 K=0 ⇒ 探针关）
arm() {
  local c="$1" rw="$2" kk="$3" out pk bm bd rows qn
  rows=$(spec_of "$c" | cut -d: -f1); qn=$(spec_of "$c" | cut -d: -f2)
  if [ "$rw" = "0" ]; then
    out=$(timeout 600 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  else
    out=$(SFA_PROBE_ROWS="$rw" SFA_PROBE_K="$kk" timeout 600 ./test_sfa_dev "cases/$c.bin" "$REPS" none 2>&1)
  fi
  pk=$(printf '%s' "$out" | grep -a SFA_PICK | head -1 | sed 's/.*SFA_PICK //')
  # SFA_PICK 打印在覆盖【之前】，所以它不是生效凭据 —— eff 才代表 kernel 真正拿到的 n_blk
  eff=$(printf '%s' "$out" | grep -a SFA_EFF | head -1 | sed 's/.*SFA_EFF //')
  bm=$(printf '%s' "$out" | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//')
  printf '%-14s r=%-4s n=%-2s %-4s 批量 %10s  eff[%s]  pick[%s]\n' "$c" "$rows" "$qn" \
         "$([ "$rw" = 0 ] && echo off || echo "on/$rw/$kk")" "$bm" "$eff" "$pk"
}

echo "########## P107 标定：扰动 off vs (rows<=40 -> n_blk=16) vs (rows<=20 -> n_blk=16) ##########"
ALL="p1 p2 p4 p6 pm41 pm81 pm161 p1s1c16384 p4s1c16384 p16s1c16384 p32s1c16384 \
     p4s128c512 p16s128c128 p32s128c512 w1 w2 big1 w3 w4"
for c in ${ONLY:-$ALL}; do
  arm "$c" 0 0
  arm "$c" 40 16
  arm "$c" 20 16
  echo "---"
done

echo "########## 探针 ON(rows<=40) 下的正确性：只读有真 expect 的档 ##########"
for c in r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads \
         q1h q2h q3h p_n64 p_n512 p_n1024 e1empty e2one e3two e4odd e5s2one e6many e7padq e8padkv p1 p2 p4 p6 big1; do
  for f in 0 1; do
    tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
    line=$(SFA_PROBE_ROWS=40 SFA_PROBE_K=16 env $([ "$f" = 1 ] && echo SFA_F32=1) \
           timeout 300 ./test_sfa_dev "cases/$c.bin" 1 diff 2>&1 \
           | grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|不\*\*逐位一致\*\*|FAIL|PASS' | tr '\n' ' ')
    printf '%-10s %-5s %s\n' "$c" "$tag" "$line"
  done
done
echo "P107_CALIB_DONE"
