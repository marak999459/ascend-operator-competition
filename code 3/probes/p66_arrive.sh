#!/bin/bash
# P66：p65 之后，嫌疑只剩一条线 —— AIV→AIC 的到达。
#
# 实测矩阵（同一份内核，只改握手）：
#   free6   通  ：门关 + 排空删 + AIV 不等 READY + set 常量 6 + 收尾 wait(6)
#   noack   通  ：门关 + 排空删 + **AIV 照等 READY** + credit 照发       ← p65
#   perdir  挂  ：完整协议 + READY 按 sub 分号(4/8) + credit 常量分支(5/7) ← p65
#   c57     挂  ：完整协议，只把 set 的号写成编译期常量 5/7               ← p64
#   dbl     挂  ：两颗共用常量 6、AIC 每片等两次                          ← p61
#   full    挂  ：完整协议
#   p63     —   ：实测两侧片数逐块相等（43=43=43），且 sub0 确实走到 set 那行
# ⇒ ① AIC→AIV 的 READY 广播到得了两颗（noack 直接证死 H-A）；②编号写法无罪（c57）；
#   ③"谁没跑"无罪（p63）。**唯一还没被钉死的自变量：AIV→AIC 的那次 arrive 到底落没落到
#   AIC 的等待上**。而 p65 之后所有"通"的形态都是"AIC 一个 credit 都不等"。
#
# 关键旁证（这一轮从 refs/sfa/cann_builtin_900 里读出来的）：官方 MIX 的流水是
#   AIC: Mm1 每片 set C1V1(7) → Mm2 每片 **wait V1C2(8) 一次**，再 set C2V2(9)+C2V1(4)
#   AIV: 两颗【都】跑同样的 nBufferLoopTimes 次循环（片内按 GetBlockIdx()%2 分 M 行），
#        每片 wait C1V1 → set V1C2 → wait C2V1
#   ⇒ 官方是"两颗 AIV 每片各 set 一次、AIC 每片只 wait 一次"，净富余 +1/片 还跑得动。
#   也就是说**官方从来不依赖"某一特定颗"的到达**。如果硬件上 sub1 的 AIV→AIC arrive
#   根本落不到 AIC（或落到一条 AIC 看不见的线），官方照样对，我们照样挂 —— 因为
#   我们的收支表是"每条线各 1 张、一线一颗"，线 7 的唯一来源就是 sub1。
#
# 四发定向（判据只有 rc：124=挂，1/0=跑到了收尾）：
#   `sub0only` 只有 sub0 set(5)，AIC 每片只等 5   ⇒ 通的形态基线
#   `sub1only` 只有 sub1 set(7)，AIC 每片只等 7   ⇒ 挂 = sub1 的 arrive 或 线 7 是死的
#   `sub1on5`  只有 sub1 set(5)，AIC 每片只等 5   ⇒ 挂而 sub0only 通 ⇒ 死的是【颗】不是【号】
#   `ign7`     两颗照自然号 set(5/7)，AIC 只等 5  ⇒ 通 = 这就是可用的修法形状
# ⚠️ 补丁只打在远端 ~/sfa_real，跑完必 RESTORE + 重建成净版；本地提交源不动。
# ⚠️ 输出超差不看：这些形态的数据时序本就没保。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-sub0only}"
CASES="${CASES:-r6_multiB p1}"
TMO="${TMO:-70}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ($VARIANT) ---"
ns "$ENVR; cp -f $K /tmp/kernel_p66_orig.cpp; md5sum /tmp/kernel_p66_orig.cpp"

echo "--- PATCH ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p66_orig.cpp', encoding='utf-8').read()

GATE  = ('        if (cp_ >= sfa::SFA_RING) {\n'
         '            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED);\n'
         '            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED + 2u);\n'
         '        }')
DRAIN = ('            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
         '            for (uint32_t i = 0u; i < d; ++i) {\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED);\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED + 2u);\n'
         '            }')
CSET  = '            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED + 2u * sub_);'

def rep(t, a, b, n):
    assert t.count(a) == 1, 'anchor %s x%d' % (n, t.count(a))
    return t.replace(a, b, 1)

def gate(line):
    return ('        if (cp_ >= sfa::SFA_RING) {                       // [p66]\n'
            '            CrossCoreWaitFlag<2, PIPE_MTE2>(%su);\n'
            '        }' % line)

def drain(line):
    return ('            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
            '            for (uint32_t i = 0u; i < d; ++i) {             // [p66]\n'
            '                CrossCoreWaitFlag<2, PIPE_S>(%su);\n'
            '            }' % line)

def cset(which, line):
    return ('            if (sub_ == %su) {                             // [p66]\n'
            '                CrossCoreSetFlag<2, PIPE_MTE3>(%su);\n'
            '            }' % (which, line))

if v == 'sub0only':   t = rep(rep(src, CSET, cset(0, 5), 'cset'), GATE, gate(5), 'gate'); t = rep(t, DRAIN, drain(5), 'drain')
elif v == 'sub1only': t = rep(rep(src, CSET, cset(1, 7), 'cset'), GATE, gate(7), 'gate'); t = rep(t, DRAIN, drain(7), 'drain')
elif v == 'sub1on5':  t = rep(rep(src, CSET, cset(1, 5), 'cset'), GATE, gate(5), 'gate'); t = rep(t, DRAIN, drain(5), 'drain')
elif v == 'ign7':     t = rep(rep(src, GATE, gate(5), 'gate'), DRAIN, drain(5), 'drain')
else: raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(t)
print('PATCHED_%s lines %d -> %d' % (v.upper(), len(src.splitlines()), len(t.splitlines())))
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p66_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p66_build.log | head -6; tail -2 /tmp/p66_build.log")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 读数是上一个二进制的 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p66_orig.cpp $K; md5sum $K"
     ns "$ENVR; bash build.sh > /tmp/p66_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P66_ABORT"; exit 1;;
esac

for cs in $CASES; do
  ns "$ENVR; rm -rf /tmp/p66_gold; mkdir -p /tmp/p66_gold; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 write 2e-3 1e-2 /tmp/p66_gold > /tmp/p66_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|PASS|FAIL|error[A-Za-z]*:? ?[A-Za-z0-9_ .:-]{0,40}' /tmp/p66_$cs.log | sort -u | head -3 | tr '\n' ' '); \
      printf '%-12s rc=%-4s 落地:%s %s\n' $cs \$rc \"\$(cd /tmp/p66_gold 2>/dev/null && ls | wc -l)\" \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p66_orig.cpp $K; md5sum $K"
echo "--- REBUILD CLEAN ---"
ns "$ENVR; bash build.sh > /tmp/p66_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P66_DONE"
