#!/bin/bash
# P58 生死判：M1d 的 cube 形态整体挂死（ping-pong 后 6/6 rc=124，无异常）。
# 在**远端副本**上把 CubeOneChunk 从某个切点之后剪掉，并在那里补一条 READY，
# 于是"能不能跑完"就把挂点夹在两段之间：
#   handshake   扫完片立刻 set READY（只测旗标流；输出必错，但要求能收尾）
#               ⚠️ 这一档把 credit 门**一起跳过**了 ⇒ AIC 领先无上限。真机实测（本轮）：
#                  无门时多片必挂、单片档通过。两条候选解释：①AIV 收件箱只有 ~RING 深，
#                  超出的 set 是丢的不是等的（与 §15.71 cubexfer"领先 ≤2 轮跑到 256 轮"自洽）；
#                  ②两侧的片数本来就不等（AIC 少产）。gatecut 就是来分这两条的。
#                  ⇒ 判旗标流要用 gatecut（门留着），不能用 handshake。
#   gatecut     留着 credit 门、剪掉 gather 之后的全部 cube 活 ⇒ 纯握手全通路（READY +
#               两条 credit + 排空）；这一档通过就说明交接协议本身没问题
#   gatecut1    同 gatecut，但 SFA_RING 2->1（锁步档）
#   aftergather 保留 K/K-rope 的 ND2NZ gather 进 L1，剪掉 L0 搬运/Mmad/Fixpipe
#   aftermmad   保留 gather + LoadData + 5 片 Mmad，剪掉 Fixpipe（L0C 不排空）
#   nowait      AIV 不等 READY（≡ 只测"AIC 自己能不能走完流"）
#   rdyonly     handshake + 关排空（本轮已用：证明 AIC 侧扫描/旗标自身不挂）
#   full        不打补丁（当前实现，已知挂）
# ⚠️ 另记一条读头文件得到的口径（dav_c220/kernel_operator_sync_impl.h:432-440）：
#    `CrossCoreWaitFlag<mode,pipe>` 的 **pipe 模板参数在 arch22 被忽略**
#    （实现就是 `(void)modeId; wait_flag_dev(flagId);`）⇒ 换等待管道不可能改变行为，
#    `vwait` 档真机复现了这一点（与 PIPE_MTE2 同挂）。pipe 只对 **set** 侧有效。
# ⚠️ aftermmad 的档只跑单片用例：Fixpipe 不排空时，下一片的 Mmad 可能因 L0C 占着而停，
#    那是这一档自己的副作用，不能当成挂点证据。
# 每案一条独立 ssh：单条 ssh 有 900 s 窗口，6 案 ×120 s 会把它撑爆（上一轮就是这么丢读数的）。
# 结束一定 RESTORE：远端 ~/sfa_real/code 是提交源的副本，补丁只活在这里。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-handshake}"
CASES="${CASES:-r6_multiB p1 big1}"
TMO="${TMO:-120}"
ns() { timeout "${T:-$((TMO + 200))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p58_orig.cpp; md5sum /tmp/kernel_p58_orig.cpp $K"

echo "--- PATCH ($VARIANT) ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p58_orig.cpp', encoding='utf-8').read()
READY = '        { CrossCoreSetFlag<2, PIPE_FIX>(sfa::CF_READY); ++cp_; return true; }'
def after(text, anchor, name):
    assert text.count(anchor) == 1, '%s x%d' % (name, text.count(anchor))
    return text.replace(anchor, anchor + '\n' + READY, 1)
def nodrain(text):
    a = '            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;'
    assert text.count(a) == 1, 'drain anchor x%d' % text.count(a)
    out = text.replace(a, '            const uint32_t d = 0u;', 1)
    g = '        if (cp_ >= sfa::SFA_RING) {'
    assert out.count(g) == 1, 'gate anchor x%d' % out.count(g)
    return out.replace(g, '        if (false) {', 1)
def nowait(text):
    w = '        CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_READY);'
    assert text.count(w) == 1, 'ready-wait anchor x%d' % text.count(w)
    return text.replace(w, '        // [p58] AIV 不等 READY ⇒ 这一档只测"AIC 能不能走完自己的流"', 1)
def flood(text):
    a = '            const uint32_t d = (cp_ < sfa::SFA_RING) ? cp_ : sfa::SFA_RING;'
    assert text.count(a) == 1, 'drain anchor x%d' % text.count(a)
    return text.replace(a, '            for (uint32_t z = 0u; z < 1024u; ++z) {\n'
                           '                PipeBarrier<PIPE_ALL>();\n'
                           '                CrossCoreSetFlag<2, PIPE_FIX>(sfa::CF_READY);\n'
                           '            }\n' + a, 1)
def before(text, anchor, name):
    assert text.count(anchor) == 1, '%s x%d' % (name, text.count(anchor))
    return text.replace(anchor, READY + '\n' + anchor, 1)
def vwait(text):
    w = '        CrossCoreWaitFlag<2, PIPE_MTE2>(sfa::CF_READY);'
    assert text.count(w) == 1, 'ready-wait anchor x%d' % text.count(w)
    return text.replace(w, '        CrossCoreWaitFlag<2, PIPE_V>(sfa::CF_READY);   // [p58] 换管道', 1)
def ring1(text):
    a = 'constexpr uint32_t SFA_RING = 2u;'
    assert text.count(a) == 1, 'ring anchor x%d' % text.count(a)
    return text.replace(a, 'constexpr uint32_t SFA_RING = 1u;', 1)
GATHERC = '        // 3) 逐段 ND2NZ 收 K / K-rope 进 L1 的 B tile'
if v == 'full':
    out = src
elif v == 'gatecut':
    out = before(src, GATHERC, 'gate')
elif v == 'gatecut1':
    out = ring1(before(src, GATHERC, 'gate'))
elif v == 'handshake':
    out = after(src, '        if (cnt == 0u) { return false; }', 'scan')
elif v == 'rdyonly':
    out = nodrain(after(src, '        if (cnt == 0u) { return false; }', 'scan'))
elif v == 'nowait':
    out = nowait(nodrain(after(src, '        if (cnt == 0u) { return false; }', 'scan')))
elif v == 'flood':
    out = nodrain(flood(after(src, '        if (cnt == 0u) { return false; }', 'scan')))
elif v == 'vwait':
    out = vwait(nodrain(after(src, '        if (cnt == 0u) { return false; }', 'scan')))
elif v == 'mathonly':
    out = nodrain(src)
elif v == 'aftergather':
    out = after(src, '        WaitFlag<HardEvent::MTE2_MTE1>(0);', 'gather')
elif v == 'aftermmad':
    out = after(src, '        SetFlag<HardEvent::M_FIX>(3);', 'mmad')
else:
    raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(out)
print('PATCHED_%s  lines %d -> %d' % (v.upper(), len(src.splitlines()), len(out.splitlines())))
PYX"

if [ "$VARIANT" != full ]; then
  echo "--- BUILD ---"
  ns "$ENVR; bash build.sh > /tmp/p58_build.log 2>&1; echo build_rc=\$?; tail -3 /tmp/p58_build.log"
fi

for cs in $CASES; do
  ns "$ENVR; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 diff > /tmp/p58_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|FAIL|PASS|errorStr: [A-Za-z0-9_ .-]+' /tmp/p58_$cs.log | sort -u | tr '\n' ' '); \
      printf '%-12s rc=%-4s %s\n' $cs \$rc \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p58_orig.cpp $K; md5sum $K"
echo "LIVENESS_DONE"
