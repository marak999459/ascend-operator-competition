#!/bin/bash
# P92b：cube 路径的**每 chunk 固定税**标定 —— 同一场三次只换 NBLK_CAND 档（16/32/64），
# 单元数与总计算量不变、chunk 条数按 1/2/4 倍变 ⇒ 时间对 chunk 条数的斜率就是
# "旗标往返 + Fixpipe + 环往返"的单价。⚠️ 只动远端副本；trap 还原干净构建。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
CS="${CS:-w3 w4 k_n8s128 k_n16s64 k_n64s16 p6}"
HOSTREL=code/op_host/sparse_flash_attention.cpp
OLD='constexpr uint32_t NBLK_CAND[] = {128, 64, 48, 40, 32, 16, 8, 4, 2, 1};'
ns() { timeout "${T:-1800}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

restore() {
  echo "########## 还原提交源构建 $(date +%H:%M:%S) ##########"
  ( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) 2>&1 | grep -aE 'op_host/sparse' | head -1
  ( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -3
  ns "cd ~/sfa_real && grep -ac '\[MEM\]' $HOSTREL; grep -a 'NBLK_CAND\[\] =' $HOSTREL"
}
trap restore EXIT

for v in "$@"; do
  echo "########## n_blk = $v  $(date +%H:%M:%S) ##########"
  ( cd "$REPO" && timeout 300 bash "code 3/npu_debug/npu.sh" sync ) 2>&1 | grep -a 'op_host/sparse' | head -1
  # 旋钮生效证据当门：命中数必须 == 1，否则这一臂作废
  ns "cd ~/sfa_real && python3 - $v <<'PY'
import io, sys
v = sys.argv[1]
p = 'code/op_host/sparse_flash_attention.cpp'
old = 'constexpr uint32_t NBLK_CAND[] = {128, 64, 48, 40, 32, 16, 8, 4, 2, 1};'
new = 'constexpr uint32_t NBLK_CAND[] = {%s};' % v
s = io.open(p, encoding='utf-8').read()
assert s.count(old) == 1, 'ANCHOR MISS %d' % s.count(old)
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
PY
      grep -ac 'NBLK_CAND\[\] = {$v}' ~/sfa_real/$HOSTREL"
  ( cd "$REPO" && timeout 1800 bash "code 3/npu_debug/npu.sh" build ) 2>&1 | grep -aE '构建 OK|error:' | head -5
  ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; \
      export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
      export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
      for r in 1 2; do for c in $CS; do \
        o=\$(timeout 300 ./test_sfa_dev cases/\$c.bin 1 diff 2>&1); \
        t=\$(printf '%s' \"\$o\" | grep -aoE '平均 [0-9.]+ ms' | head -1); \
        [ -z \"\$t\" ] && t=\"FAIL | \$(printf '%s' \"\$o\" | tail -2 | tr '\n\r' '  ')\"; \
        d=\$(printf '%s' \"\$o\" | grep -aoE '超差 [0-9]+/[0-9]+' | head -1); \
        printf 'r%s nblk=%-3s %-10s %s | %s\n' \$r $v \$c \"\$t\" \"\$d\"; \
      done; done"
done
