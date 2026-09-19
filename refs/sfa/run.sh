#!/bin/bash
# SFA 真机：编译 harness + 生成用例 + 跑对拍
# 用法: run.sh [用例名...]     不传则跑全部
set -u
source /home/developer/Ascend/cann-9.0.0/set_env.sh 2>/dev/null
CANN=/home/developer/Ascend/cann-9.0.0
R=/home/developer/sfa_real
V=$R/vendor/custom

# ---- 编译 harness ----
cd "$R" || exit 2
echo "=== 编译 harness ==="
g++ -std=c++17 -O2 test_sfa_real.cpp -o test_sfa \
  -I$CANN/aarch64-linux/include -I$V/op_api/include \
  -L$CANN/aarch64-linux/lib64 -L$V/op_api/lib \
  -lascendcl -lnnopcapbase -lcust_opapi 2>/dev/null \
 || g++ -std=c++17 -O2 test_sfa_real.cpp -o test_sfa \
  -I$CANN/aarch64-linux/include -I$V/op_api/include \
  -L$CANN/aarch64-linux/lib64 -L$V/op_api/lib \
  -lascendcl -lnnopbase -lcust_opapi 2>&1 | head -8
[ -f test_sfa ] || { echo "HARNESS 编译失败"; exit 1; }
echo "harness OK"

# ---- 生成用例（在 NPU 上用 python+numpy 跑参考实现）----
GEN="${GEN:-1}"
if [ "$GEN" = "1" ]; then
  echo "=== 生成用例 ==="
  cd "$R/cases" 2>/dev/null || { mkdir -p "$R/cases"; cd "$R/cases"; }
  python3 - <<'PYEOF'
import sys, os
sys.path.insert(0, '/home/developer/sfa_real')
import sfa_ref as S
cases = [
    # 名称,        B, S1, S2, N1, D, SBS, MODE, COUNT, nblk
    ("r1_min",      1, 1, 16, 1, 512, 1, 0, 16, 4),
    ("r2_chunk",    1, 1, 64, 1, 512, 1, 0, 64, 8),
    ("r3_mode3",    1, 4, 32, 1, 512, 1, 3, 32, 4),
    ("r4_shortkv",  1, 4,  4, 1, 512, 1, 3, 16, 2),
    ("r5_blocks",   1, 2, 64, 1, 512, 4, 0, 32, 4),
    ("r6_multiB",   2, 2, 32, 2, 512, 1, 3, 32, 4),
    ("r7_norope",   1, 1, 32, 1, 512, 1, 0, 32, 4),
    ("r8_heads",    1, 2, 32, 8, 512, 1, 3, 32, 4),
]
for (name, B, S1, S2, N1, D, SBS, MODE, CN, nblk) in cases:
    zr = (name == "r7_norope")
    c = S.gen_case(B, S1, S2, N1, D, SBS, MODE, CN, seed=hash(name) % 10000,
                   nblk=nblk, zero_rope=zr)
    p = "/home/developer/sfa_real/cases/%s.bin" % name
    S.write_case(p, c)
    print("  生成 %-12s B=%d S1=%d S2=%d N1=%d SBS=%d MODE=%d" % (name, B, S1, S2, N1, SBS, MODE))
PYEOF
fi

# ---- 跑 ----
export ASCEND_CUSTOM_OPP_PATH=$V
export LD_LIBRARY_PATH=$V/op_api/lib:$CANN/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
cd "$R"

LIST="${*:-r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads}"
echo ""
echo "############ 真机对拍 ############"
pass=0; fail=0
for n in $LIST; do
  f="$R/cases/$n.bin"
  [ -f "$f" ] || { printf "%-12s SKIP(无用例)\n" "$n"; continue; }
  timeout 300 ./test_sfa "$f" > "/tmp/rt_$n.txt" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then st="PASS"; pass=$((pass+1)); else st="FAIL($rc)"; fail=$((fail+1)); fi
  line=$(grep -a -E "最大相对误差|超差元素|墙钟" "/tmp/rt_$n.txt" | tr '\n' ' ')
  printf "%-12s %-9s %s\n" "$n" "$st" "$line"
done
echo "-----------------------------------------------------------"
echo "PASS=$pass FAIL=$fail"
