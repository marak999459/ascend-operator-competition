#!/bin/bash
# P62：把 p61 的判决落成"可直接上线的那一档"。
#
# 到 p61 为止的三条实测（probes/p60_direction.sh / p61_ids.sh）：
#   free5  挂   ：AIV `set(CF_CRED + 2*sub_)`（**运行时算出来的 id**）+ AIC `wait(5)`
#   free6  通   ：两颗 AIV `set(6u)`（**编译期常量**）+ AIC `wait(6)`
#   nostale 挂  ：AIV 一个 set 都不发 + AIC `wait(6)` ⇒ id 6 上没有残留，free6 的"通"是真到达
# ⇒ 变量还剩两维纠缠着：(id 的**值** 5/7 vs 6) 与 (id 是**常量**还是**运行时算出来的**)。
#   p61 的 `cred6` 正好钉在第一维上：它保留运行时表达式、只把 `CF_CRED` 5u 挪到 6u
#   ⇒ id 变成运行时的 {6,8}。**若它挂而 `dbl`（常量 6、每片等两次）通过**，就坐实第二条：
#     arch22 的 `ffts_cross_core_sync` 需要**立即数 id**，运行时算出来的 id 静默打歪。
#   （官方 `kernel_mla.h:873` 那里传的是 `constInfo.syncV0C1` —— 看着像运行时成员，但它由
#     `static constexpr`（:88-93）在同一个模板实例里赋定值，内联后必然折成立即数 ⇒ 与
#     "运行时不可用"不矛盾。反过来我的 `sub_ = GetBlockIdx() & 1` 是**真正的运行时量**，折不动。）
# 本档就是那一发的两个上线候选（都走完整协议，判据 = rc + 超差）：
#   cc68    四处长 id 全用**编译期常量**，保留"每颗 AIV 一条自己的 credit 线"：
#             set 侧按 `sub_` 二选一（两条都是常量）、gate/drain 侧 `wait(6)`+`wait(8)`
#   cc68b   同 cc68，但 READY 也从 4 挪到 **常量 4**（本来已是常量）+ credit 线换成 {7,9}
#           —— 用来复核"通过"不是某个特定数字的运气（两组不同的常量号都通才安心）
# ⚠️ 补丁只打在远端 ~/sfa_real/code 上，跑完一定 RESTORE；本地提交源不动。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-cc68}"
CASES="${CASES:-r2_chunk r6_multiB p1 p2}"
TMO="${TMO:-60}"
ns() { timeout "${T:-$((TMO + 240))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ($VARIANT) ---"
ns "$ENVR; cp -f $K /tmp/kernel_p62_orig.cpp; md5sum /tmp/kernel_p62_orig.cpp $K"

echo "--- PATCH ($VARIANT) ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p62_orig.cpp', encoding='utf-8').read()

CRED = 'constexpr uint32_t CF_CRED  = 5u;'
GW0  = '            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED);'
GW1  = '            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED + 2u);'
CSET = '            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED + 2u * sub_);'
DRAIN = ('            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
         '            for (uint32_t i = 0u; i < d; ++i) {\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED);\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED + 2u);\n'
         '            }')

def one(text, anchor, name):
    assert text.count(anchor) == 1, '%s x%d' % (name, text.count(anchor))
    return text
def rep(text, anchor, new, name):
    return one(text, anchor, name).replace(anchor, new, 1)

if v in ('cc68', 'cc68b'):
    a, b = ('6u', '8u') if v == 'cc68' else ('7u', '9u')
    # set 侧：两条常量分支，运行时只选边、不算 id
    t = rep(src, CSET,
            '            if (sub_ == 0u) {        // [p62] id 必须是编译期常量\n'
            '                CrossCoreSetFlag<2, PIPE_MTE3>(%s);\n'
            '            } else {\n'
            '                CrossCoreSetFlag<2, PIPE_MTE3>(%s);\n'
            '            }' % (a, b), 'cset')
    t = rep(t, GW0,  '            CrossCoreWaitFlag<2, PIPE_MTE2>(%s);   // [p62] 常量' % a, 'gw0')
    t = rep(t, GW1,  '            CrossCoreWaitFlag<2, PIPE_MTE2>(%s);   // [p62] 常量' % b, 'gw1')
    t = rep(t, DRAIN,
            '            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
            '            for (uint32_t i = 0u; i < d; ++i) {\n'
            '                CrossCoreWaitFlag<2, PIPE_S>(%s);   // [p62] 常量\n'
            '                CrossCoreWaitFlag<2, PIPE_S>(%s);   // [p62] 常量\n'
            '            }' % (a, b), 'drain')
    out = t
else:
    raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(out)
print('PATCHED_%s  lines %d -> %d' % (v.upper(), len(src.splitlines()), len(out.splitlines())))
PYX"

echo "--- BUILD ---"
ns "$ENVR; bash build.sh > /tmp/p62_build.log 2>&1; echo build_rc=\$?; tail -3 /tmp/p62_build.log"

for cs in $CASES; do
  ns "$ENVR; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 diff > /tmp/p62_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [A-Za-z0-9]+|PASS|FAIL|errorStr: [A-Za-z0-9_ .:-]+' /tmp/p62_$cs.log | sort -u | tr '\n' ' '); \
      printf '%-12s rc=%-4s %s\n' $cs \$rc \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p62_orig.cpp $K; md5sum $K"
echo "P62_DONE"
