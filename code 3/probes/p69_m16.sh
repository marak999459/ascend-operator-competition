#!/bin/bash
# P69：M 轴定则实验 —— A tile 的 nValue 从 N1(=4) 补齐到 16 个【真实行】，别的都不动。
#
# 已知（p68 环读回）：末片 score 里只有 head3 与本地参考逐位相同，head0 读回 0x7fff
#   （fp16 NaN ⇒ 那一 lane 里是未初始化/巨大值，乘出来 inf），head1/head2 是 std≈2.6
#   的碎屑 ⇒ 环布局、Fixpipe 行距、AIV 行号都是对的，错的是【L0A 的 m 轴内容】。
# 与官方 arch22 mm1 逐字段对齐后（Nd2Nz/LoadData2D/Mmad/FixpipeParamsV220 全部同式，
#   见 refs/…/sparse_flash_attention_service_cube_mla.h:356-437, 441-489, 800-835），
#   唯一的形状差别是：官方 A 侧 nValue 恒等于 M_SPLIT_SIZE=16，从不给不满 16 的
#   nValue（B 侧才用任意行数）。所以这一档就赌这一个变量。
# 做法：qGm_ 起读 16 行（p1 的 Q 总共正好 16 行 ⇒ 首单元零越界；其余单元把基址夹到
#   qTot-16·D ⇒ 只会读到别的行的头，不会出张量）⇒ 判据只看 row(0,0) 的四个头。
# ⚠️ 只打远端 ~/sfa_real；跑完 RESTORE + 重建成净版。判据是**读回来的数**，不是 PASS/FAIL。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p69_orig.cpp; md5sum /tmp/kernel_p69_orig.cpp"

echo "--- PATCH ---"
ns "$ENVR; python3 - <<'PYX'
import io
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open(p, encoding='utf-8').read()

OLD1 = '''                const float l = ml.GetValue(halfOff_ + i);
                LocalTensor<float> oi = o[i * rowC];
                if (l > 0.0f) {
                    Muls(oi, oi, 1.0f / l, rowC);
                } else {
                    Duplicate(oi, 0.0f, rowC);      // 整行被 mask：输出 0
                }'''
NEW1 = '''                const float l = ml.GetValue(halfOff_ + i);   // [p68] LSE 照旧
                LocalTensor<float> oi = o[i * rowC];
                Adds(oi, sBuf_.Get<float>()[i * nBlk_], 0.0f, 64u);   // [p68] 末片原值
                Duplicate(oi[64], 0.0f, rowC - 64u);                  // [p68]'''

OLD2 = '''        DataCopy(ctx.l1qa, qGm_[cs1_], ctx.nzA);
        ctx.nzAr.nValue = N1_;
        DataCopy(ctx.l1qr, qrGm_[ropeBase], ctx.nzAr);'''
NEW2 = '''        {                                                                   // [p69]
            const uint64_t qTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)D_;   // [p69]
            const uint64_t rTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)Dr_;  // [p69]
            const uint64_t aNeed = 16ull * (uint64_t)D_;                     // [p69]
            const uint64_t rNeed = 16ull * (uint64_t)Dr_;                    // [p69]
            const bool bigA = (qTot >= aNeed);                               // [p69]
            const bool bigR = (rTot >= rNeed);                               // [p69]
            uint64_t ab = (uint64_t)cs1_;                                    // [p69]
            if (bigA && ab + aNeed > qTot) { ab = qTot - aNeed; }            // [p69]
            uint64_t rb = ropeBase;                                          // [p69]
            if (bigR && rb + rNeed > rTot) { rb = rTot - rNeed; }            // [p69]
            ctx.nzA.nValue = bigA ? 16u : (uint32_t)N1_;                     // [p69] M 轴补齐 16 行
            DataCopy(ctx.l1qa, qGm_[ab], ctx.nzA);                           // [p69]
            ctx.nzAr.nValue = bigR ? 16u : (uint32_t)N1_;                    // [p69]
            DataCopy(ctx.l1qr, qrGm_[rb], ctx.nzAr);                         // [p69]
        }                                                                     // [p69]'''

for old, new, tag in ((OLD1, NEW1, 'ringdump'), (OLD2, NEW2, 'm16')):
    assert src.count(old) == 1, '%s anchor x%d' % (tag, src.count(old))
    src = src.replace(old, new, 1)
    print('PATCHED_' + tag)
io.open(p, 'w', encoding='utf-8').write(src)
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p69_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p69_build.log | head -6")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p69_orig.cpp $K; bash build.sh > /tmp/p69_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P69_ABORT"; exit 1;;
esac

echo "--- RUN (write -> /tmp/p69_act) ---"
ns "$ENVR; mkdir -p /tmp/p69_act; for cs in p1; do timeout $TMO ./test_sfa_dev cases/\$cs.bin 1 write 2e-3 1e-2 /tmp/p69_act > /tmp/p69r_\$cs.log 2>&1; echo \"\$cs rc=\$?\"; done; ls /tmp/p69_act"

echo "--- PULL ---"
timeout 300 ssh -F $S $H "cd /tmp/p69_act && tar cf - ." 2>/dev/null | (mkdir -p /tmp/p69/dump && tar xf - -C /tmp/p69/dump) && ls -la /tmp/p69/dump

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p69_orig.cpp $K; md5sum $K"
ns "$ENVR; bash build.sh > /tmp/p69_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P69_DONE"
