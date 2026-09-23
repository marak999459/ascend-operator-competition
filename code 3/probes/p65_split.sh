#!/bin/bash
# P65：p64 把"编号写法"判死之后，两侧耦合只剩两条独立的路要分头量。
#
# 到 p64 为止的完整实测表（同一份内核，只改握手的开关）：
#   free6   通  ：门关 + 排空删 + **AIV 不等 READY** + set 常量 6 + 收尾 wait(6) 一次
#   nostale 挂 ：同 free6 但一颗 AIV 都不 set ⇒ free6 那条到达是真的
#   c57     挂 ：**完整协议**，只把 set 的号写成编译期常量 5/7 ⇒ 编号写法无罪
#   dbl     挂 ：门关? 不，完整协议 + 两颗共用常量 6 + 每片等两次
#   cred6   挂 ：完整协议，运行时号 {6,8}
#   p63     —   ：门关 + AIV 不等 READY + set 保持运行时原样 ⇒ 实测两侧片数【逐块相等】
#                （p1 每组 AIV sub0/sub1 各 telSets_=43 = AIC cp_=43；r6 各 1 = 1）
# ⇒ "谁没跑/跑几片"这条整个翻篇了。剩下的两个自变量在 free6 里是**一起**被关掉的：
#     ① AIC 等 credit（门 + 排空）    ② AIV 等 READY
#   所以现在是两条假设，各自能独立解释全部数据：
#     H-A：**AIC→AIV 的 READY 只到得了半颗核**（mode 2 在 cube 侧不是"发给组里两颗 AIV"）
#          ⇒ AIV1 永挂 READY ⇒ 它那次 credit 也永远交不出 ⇒ c57/dbl/full 全挂；
#             而 free6 里 AIV 根本不等 READY ⇒ 通。r6（门不开、只在收尾各等一条）也挂
#             ⇒ 与 (h) 的实测"sub0 确实走到 set 那一行"不冲突：那一份是 aivfree 形态。
#     H-B：AIV→AIC 的 credit 只能被"每片一条线各一次"以外的形态吃掉（等待数/时序）。
#   分头钉：
#     `noack`   关掉 AIC 一切等待（门 + 排空都删），**AIV 照等 READY**，credit 照发。
#               ⇒ 挂 = READY 到不了两颗（H-A 立）；通 = READY 无罪，转 H-B。
#     `perdir`  完整协议，但 READY 按 sub 分成两个号（AIC 每片发 4 和 8，sub0 等 4、
#               sub1 等 8），credit 走 c57 的常量分支（5/7）。
#               ⇒ 通 = H-A 成立且这就是修法；挂 = 广播/定向都不是问题，回到 H-B。
# ⚠️ 判据只有 rc（这些形态的输出注定是错的，超差/PASS-FAIL 一概不看）。
# ⚠️ 补丁只打在远端 ~/sfa_real 上，跑完一定 RESTORE + 重建成净版；本地提交源不动。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-noack}"
CASES="${CASES:-r6_multiB p1}"
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ($VARIANT) ---"
ns "$ENVR; cp -f $K /tmp/kernel_p65_orig.cpp; md5sum /tmp/kernel_p65_orig.cpp"

echo "--- PATCH ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p65_orig.cpp', encoding='utf-8').read()

GATE  = '        if (cp_ >= sfa::SFA_RING) {'
DRAIN = ('            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
         '            for (uint32_t i = 0u; i < d; ++i) {\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED);\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED + 2u);\n'
         '            }')
R_WAIT = '        CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_READY);'
A_SET  = '        CrossCoreSetFlag<2, PIPE_FIX>(sfa::CF_READY);'
CSET   = '            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED + 2u * sub_);'

def rep(t, a, b, n):
    assert t.count(a) == 1, 'anchor %s x%d' % (n, t.count(a))
    return t.replace(a, b, 1)

if v == 'noack':                                  # 只关 AIC 侧的一切等待，AIV 保持自然协议
    t = rep(src, GATE, '        if (false) {   // [p65] 门关', 'gate')
    t = rep(t, DRAIN, '            // [p65] 排空删掉：这一档不要求任何 credit 到达', 'drain')
    out = t
elif v == 'perdir':                               # READY 按 sub 分号 + credit 走常量分支
    t = rep(src, R_WAIT,
            '        if (sub_ == 0u) {                     // [p65] 两颗各等自己的号\n'
            '            CrossCoreWaitFlag<2, PIPE_MTE2>(4u);\n'
            '        } else {\n'
            '            CrossCoreWaitFlag<2, PIPE_MTE2>(8u);\n'
            '        }', 'rwait')
    t = rep(t, A_SET,
            '        CrossCoreSetFlag<2, PIPE_FIX>(4u);    // [p65] 两条定向\n'
            '        CrossCoreSetFlag<2, PIPE_FIX>(8u);', 'aset')
    out = rep(t, CSET,
              '            if (sub_ == 0u) {                 // [p65] 常量分支\n'
              '                CrossCoreSetFlag<2, PIPE_MTE3>(5u);\n'
              '            } else {\n'
              '                CrossCoreSetFlag<2, PIPE_MTE3>(7u);\n'
              '            }', 'cset')
else:
    raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(out)
print('PATCHED_%s lines %d -> %d' % (v.upper(), len(src.splitlines()), len(out.splitlines())))
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p65_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p65_build.log | head -6; tail -2 /tmp/p65_build.log")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 读数是上一个二进制的 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p65_orig.cpp $K; md5sum $K"
     ns "$ENVR; bash build.sh > /tmp/p65_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P65_ABORT"; exit 1;;
esac

for cs in $CASES; do
  ns "$ENVR; rm -rf /tmp/p65_gold; mkdir -p /tmp/p65_gold; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 write 2e-3 1e-2 /tmp/p65_gold > /tmp/p65_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|PASS|FAIL|error[A-Za-z]*:? ?[A-Za-z0-9_ .:-]{0,40}' /tmp/p65_$cs.log | sort -u | head -3 | tr '\n' ' '); \
      printf '%-12s rc=%-4s 落地:%s %s\n' $cs \$rc \"\$(cd /tmp/p65_gold 2>/dev/null && ls | wc -l)\" \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p65_orig.cpp $K; md5sum $K"
echo "--- REBUILD CLEAN ---"
ns "$ENVR; bash build.sh > /tmp/p65_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P65_DONE"
