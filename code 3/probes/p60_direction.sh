#!/bin/bash
# P60：把"AIV→AIC 的 credit 到底能不能到"这一件事单独量出来。
#
# 已排除（本轮真机）：
#   · 换 notify 管道 PIPE_V -> PIPE_MTE3（官方 arch22 SFA 的 AIV→AIC 方向无例外用 MTE3，
#     见 refs/sfa/cann_builtin_900/..._service_vector_mla.h:1102,1107）⇒ 6/6 仍 rc=124。
#   · 数学段：`gatecut`（剪掉 gather 之后全部 cube 活、只留握手）**照样挂**，而且
#     **r6_multiB 这种单片用例也挂** ⇒ 挂点根本不在 Mmad/Fixpipe，而在"AIC 等 credit"。
#   · PIPE_ALL 栅栏：p45 cubexfer 的 256 轮里 AIC 侧一直是 `PipeBarrier<PIPE_ALL>` +
#     `set(5)`/`wait(6)`，跑通 ⇒ MIX 下 AIC 的 PIPE_ALL 不会等 AIV（那条怀疑判死）。
# ⇒ 剩下的候选只有三条，这一档脚本一条条钉：
#   free5   AIV 完全不等 READY（自己跑完）+ AIC 门关 + 排空只留【一条】wait(id 5)
#           ⇒ 通过 = AIV→AIC 链路通、问题在计数/时序；挂 = 链路本身不通
#   free6   同 free5，但两颗 AIV 都 set(id 6)、AIC 只 wait(id 6) 一次
#           ⇒ 与 p45 探针**完全同一套旗标编号与管道**，只是搬进真内核
#   shared6 旗标全改成 p45 那一套（id 6 共用、AIV set 前加 PIPE_ALL 栅栏、AIC 的
#           wait 紧跟在自己的 set 之后），其余代码不动 ⇒ "已知好的协议"整体移植
# ⚠️ 判据只有一项：**能不能收尾**（rc!=124）。这几档的输出都注定是错的（AIV 不等
#    READY / 数学被剪），超差与 PASS/FAIL 一概不看。
# ⚠️ 补丁只打在远端 ~/sfa_real/code 上，跑完一定 RESTORE；本地提交源不动。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-free5}"
CASES="${CASES:-r6_multiB p1}"
TMO="${TMO:-60}"
ns() { timeout "${T:-$((TMO + 240))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p60_orig.cpp; md5sum /tmp/kernel_p60_orig.cpp $K"

echo "--- PATCH ($VARIANT) ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p60_orig.cpp', encoding='utf-8').read()

GATE = '        if (cp_ >= sfa::SFA_RING) {'
DRAIN = ('            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
         '            for (uint32_t i = 0u; i < d; ++i) {\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED);\n'
         '                CrossCoreWaitFlag<2, PIPE_S>(sfa::CF_CRED + 2u);\n'
         '            }')
CSET = '            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED + 2u * sub_);'
R_WAIT = '        CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_READY);'
A_SET = '        CrossCoreSetFlag<2, PIPE_FIX>(sfa::CF_READY);'

def one(text, anchor, name):
    assert text.count(anchor) == 1, '%s x%d' % (name, text.count(anchor))
    return text

def gateoff(t):
    return one(t, GATE, 'gate').replace(GATE, '        if (false) {', 1)
def aivfree(t):
    return one(t, R_WAIT, 'rwait').replace(R_WAIT, '        // [p60] AIV 不等 READY', 1)
def drain1(t, id_expr):
    return one(t, DRAIN, 'drain').replace(
        DRAIN, '            CrossCoreWaitFlag<2, PIPE_S>(%s);   // [p60] 只等一条' % id_expr, 1)
def creditset(t, expr):
    return one(t, CSET, 'cset').replace(CSET, '            CrossCoreSetFlag<2, PIPE_MTE3>(%s);' % expr, 1)
def creditwait(t, expr):
    # 门关着的时候把门里两条也换掉（shared6 要用同一套编号）
    t = one(t, '            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED);', 'gw0')
    t = t.replace('            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED);',
                  '            CrossCoreWaitFlag<2, PIPE_MTE2>(%s);' % expr, 1)
    t = one(t, '            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED + 2u);', 'gw1')
    return t.replace('            CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_CRED + 2u);',
                     '            CrossCoreWaitFlag<2, PIPE_MTE2>(%s);' % expr, 1)

if v == 'free5':
    out = drain1(aivfree(gateoff(src)), 'sfa::CF_CRED')
elif v == 'free6':
    out = creditset(drain1(aivfree(gateoff(src)), '6u'), '6u')
elif v == 'shared6':
    t = creditset(src, '6u')
    t = creditwait(t, '6u')
    # 探针里验过的两处收口：AIV set 前把整条流水排空；AIC 的 wait 紧跟自己的 set
    t = t.replace(CSET, '            PipeBarrier<PIPE_ALL>();    // [p60] 探针同式\n'
                        '            CrossCoreSetFlag<2, PIPE_MTE3>(6u);', 1)
    t = t.replace(A_SET, A_SET + '\n'
                  '        if (cp_ >= sfa::SFA_RING) {\n'
                  '            CrossCoreWaitFlag<2, PIPE_FIX>(6u);   // [p60] 紧跟 set\n'
                  '        }', 1)
    out = drain1(gateoff(t), '6u')
else:
    raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(out)
print('PATCHED_%s  lines %d -> %d' % (v.upper(), len(src.splitlines()), len(out.splitlines())))
PYX"

echo "--- BUILD ---"
ns "$ENVR; bash build.sh > /tmp/p60_build.log 2>&1; echo build_rc=\$?; tail -3 /tmp/p60_build.log"

for cs in $CASES; do
  ns "$ENVR; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 diff > /tmp/p60_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|PASS|FAIL|errorStr: [A-Za-z0-9_ .:-]+' /tmp/p60_$cs.log | sort -u | tr '\n' ' '); \
      printf '%-12s rc=%-4s %s\n' $cs \$rc \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p60_orig.cpp $K; md5sum $K"
echo "P60_DONE"
