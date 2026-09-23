#!/bin/bash
# P61：P60 把挂点钉在"旗标编号"上 —— 同一份代码、同一套管道，只把 AIV→AIC 的 id
#      从 {5,7}（每颗 AIV 自己的号）换成 {6}（两颗共用）就从 rc=124 变成 rc=1。
#      这一档把"编号"这条轴量尽，并顺手试两个可直接上线的编号方案。
#
# p60 已证 / 待证的分工：
#   free5  挂  ：AIV 按 sub_ 分别 set 5 / 7，AIC 只 wait(5) 一次 —— 连"至少有一颗
#                sub_=0 的 AIV 在 set 5"都没成立（官方 arch22 内核 AIV 块号 = group*2+sub，
#                aiCoreIdx = tmpBlockIdx/2，见 refs/.../kernel_mla.h:422-428，与我同一口径）
#   free6  通  ：两颗都 set 6、AIC wait(6) 一次
#   ⇒ 还剩两种解释，必须分开：
#     (i)  id 5/7 这一条边在 arch22 MIX 下不通，id 6 通；
#     (ii) id 6 的 wait 是【白过】的（上一发探针/框架残留了一个未消费的 set）——
#         那 p60 的全部结论作废。
#   nostale 就是钉 (ii) 的控制组：把 CSET 整条删掉（AIV 一个 set 都不发），
#           其余与 free6 逐字节相同。预期【必须挂】；若它反而通过 ⇒ 判 (ii)，p60 作废。
# 上线候选两式（都跑完整协议：AIV 等 READY、门关不许关、排空按 cp_ 计数）：
#   cred6  CF_CRED 5u -> 6u ⇒ id {6,8}，每颗 AIV 一条自己的 credit 线（+2*sub_）
#   dbl    两颗 AIV 共用 id 6，AIC 每片 wait 两次（= 官方 kernel_mla.h:794-797 那个
#          "同一个 id 连发四条 / 连等四条"的计数式握手，官方就是这么用单 id 计数的）
# ⚠️ 判据：rc + 超差。cred6/dbl 走的是完整协议 ⇒ 数学对了就应当 PASS；超差非零就是
#    真 bug，别再当握手问题查。
# ⚠️ 补丁只打在远端 ~/sfa_real/code 上，跑完一定 RESTORE；本地提交源不动。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-cred6}"
CASES="${CASES:-r6_multiB p1}"
TMO="${TMO:-60}"
ns() { timeout "${T:-$((TMO + 240))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ($VARIANT) ---"
ns "$ENVR; cp -f $K /tmp/kernel_p61_orig.cpp; md5sum /tmp/kernel_p61_orig.cpp $K"

echo "--- PATCH ($VARIANT) ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p61_orig.cpp', encoding='utf-8').read()

CRED = 'constexpr uint32_t CF_CRED  = 5u;'
GATE = '        if (cp_ >= sfa::SFA_RING) {'
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

if v == 'cred6':
    out = rep(src, CRED, 'constexpr uint32_t CF_CRED  = 6u;   // [p61] 5->6', 'cred')
elif v == 'dbl':
    t = rep(src, CSET, '            CrossCoreSetFlag<2, PIPE_MTE3>(6u);   // [p61] 两颗共用 6', 'cset')
    W6 = '            CrossCoreWaitFlag<2, PIPE_MTE2>(6u);   // [p61] 共用 id：两片各等一次'
    t = rep(t, GW0, W6 + '\n' + W6, 'gw0')
    t = rep(t, GW1, '            // [p61] 共用 id ⇒ 不需要第二条线', 'gw1')
    t = rep(t, DRAIN,
            '            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;\n'
            '            for (uint32_t i = 0u; i < d; ++i) {\n'
            '                CrossCoreWaitFlag<2, PIPE_S>(6u);   // [p61] 每片两颗 AIV 各一次\n'
            '                CrossCoreWaitFlag<2, PIPE_S>(6u);\n'
            '            }', 'drain')
    out = t
elif v == 'nostale':
    # 与 free6 逐字节同形，只把 AIV 的 set 删干净：drain 的 wait(6) 若还能过，
    # 说明 id 6 上有未消费的残留 ⇒ p60 的"编号说"作废。
    t = rep(src, CSET, '            // [p61] 不发 credit（控制组）', 'cset')
    t = rep(t, GATE, '        if (false) {   // [p61] 门关', 'gate')
    t = rep(t, DRAIN, '            CrossCoreWaitFlag<2, PIPE_S>(6u);   // [p61] 只等一条', 'drain')
    out = t
else:
    raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(out)
print('PATCHED_%s  lines %d -> %d' % (v.upper(), len(src.splitlines()), len(out.splitlines())))
PYX"

echo "--- BUILD ---"
ns "$ENVR; bash build.sh > /tmp/p61_build.log 2>&1; echo build_rc=\$?; tail -3 /tmp/p61_build.log"

for cs in $CASES; do
  ns "$ENVR; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 diff > /tmp/p61_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|PASS|FAIL|errorStr: [A-Za-z0-9_ .:-]+' /tmp/p61_$cs.log | sort -u | tr '\n' ' '); \
      printf '%-12s rc=%-4s %s\n' $cs \$rc \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p61_orig.cpp $K; md5sum $K"
echo "P61_DONE"
