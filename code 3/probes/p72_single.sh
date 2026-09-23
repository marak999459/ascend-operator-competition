#!/bin/bash
# P72 = P70 的 Gram 探针 + **环改单槽（奇数片不再写 query 张量）** —— 因果判决 + 候选修法。
#
# p70/p71 把毒钉成一个"列 j 与环行 r 同时 ≡ 0 (mod N1)"的集合，而 p71 进一步证明它是
# **lane（m 轴）**级别的：把可见行挪到 {0,4,8,12} 之后这四行**整行全 NaN**。
# 被毒的 lane 号 = GM 平铺行 ≡ 0 (mod N1) = **每条 query 行的第 0 个头那一行**，因为
# A tile 的 lane n ↔ qGm_[ab + n*D_]，而 p1/p2 的 Q 总量正好 16 行 ⇒ ab 被夹到 0。
# 这正好是**奇数片环** ringQP_Gm_[fbase] 的落点：fbase = cs1_>>1 个 float = fp16 偏移
# cs1_ = (b*S1+s)*N1*D = 本单元 query 行的第 0 个头 ⇒ 每产一片奇数片，AIC 就往"别的单元
# 还要拿来当 A tile 的那些 head-0 行"上盖 16×nTile 个 fp32 分数；fp32 位模式按 fp16 读回
# 必撞 NaN 指数 ⇒ 一条 lane 里只要有一个 NaN，它参与的每一格都是 NaN。
# 这一档把奇偶两槽并成一槽、且永远落在 attention_out 行上（Gram 形态 16×16 float = 1 KB
# <= N1*D*2 B，p2 也放得下）⇒ **预言：16/16 格全中、四条头行都干净**。
# 成立即同时给出 M1d 的第一版修法：环单槽 + 只住输出行，不再碰 query。
# ⚠️ 只打远端 ~/sfa_real；跑完 RESTORE + 重建成净版。判据是**读回来的数**，不是 PASS/FAIL。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p72_orig.cpp; md5sum /tmp/kernel_p72_orig.cpp"

echo "--- PATCH ---"
ns "$ENVR; python3 - <<'PYX'
import io
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open(p, encoding='utf-8').read()

# 1) 末片 sc 原值印到 attention_out 的前 16 格（Gram 一行就 16 列）
OLD1 = '''                const float l = ml.GetValue(halfOff_ + i);
                LocalTensor<float> oi = o[i * rowC];
                if (l > 0.0f) {
                    Muls(oi, oi, 1.0f / l, rowC);
                } else {
                    Duplicate(oi, 0.0f, rowC);      // 整行被 mask：输出 0
                }'''
NEW1 = '''                const float l = ml.GetValue(halfOff_ + i);   // [p72] LSE 照旧
                LocalTensor<float> oi = o[i * rowC];
                Adds(oi, sBuf_.Get<float>()[i * nBlk_], 0.0f, 16u);   // [p72] Gram 行
                Duplicate(oi[16], 32768.0f, rowC - 16u);              // [p72] 32768=哨兵'''

# 2) A tile 的 nValue 补齐 16 个真实行（p69 那一档，让 lane↔GM 行 的映射唯一）
OLD2 = '''        DataCopy(ctx.l1qa, qGm_[cs1_], ctx.nzA);
        ctx.nzAr.nValue = N1_;
        DataCopy(ctx.l1qr, qrGm_[ropeBase], ctx.nzAr);'''
NEW2 = '''        {                                                                   // [p72]
            const uint64_t qTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)D_;   // [p72]
            const uint64_t rTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)Dr_;  // [p72]
            const uint64_t aNeed = 16ull * (uint64_t)D_;                     // [p72]
            const uint64_t rNeed = 16ull * (uint64_t)Dr_;                    // [p72]
            const bool bigA = (qTot >= aNeed);                               // [p72]
            const bool bigR = (rTot >= rNeed);                               // [p72]
            uint64_t ab = (uint64_t)cs1_;                                    // [p72]
            if (bigA && ab + aNeed > qTot) { ab = qTot - aNeed; }            // [p72]
            uint64_t rb = ropeBase;                                          // [p72]
            if (bigR && rb + rNeed > rTot) { rb = rTot - rNeed; }            // [p72]
            ctx.nzA.nValue = bigA ? 16u : (uint32_t)N1_;                     // [p72]
            DataCopy(ctx.l1qa, qGm_[ab], ctx.nzA);                           // [p72]
            ctx.nzAr.nValue = bigR ? 16u : (uint32_t)N1_;                    // [p72]
            DataCopy(ctx.l1qr, qrGm_[rb], ctx.nzAr);                         // [p72]
        }                                                                     // [p72]'''

# 3) 五个 k 切片：B 侧从 A tile 同址装载 ⇒ C = A·Aᵀ
OLD3 = '''            ctx.ldB.repeatTimes = static_cast<uint8_t>((nTile / 16u) * nF);
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1kr, ctx.ldB);
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1ka[j * nTile * kk], ctx.ldB);
            }'''
NEW3 = '''            ctx.ldB.repeatTimes = static_cast<uint8_t>(nF);              // [p72] B = A tile
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1qr, ctx.ldB);                        // [p72]
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1qa[j * 16u * kk], ctx.ldB);          // [p72]
            }'''

OLD4 = '''            ctx.mp.n = static_cast<uint16_t>(nTile);'''
NEW4 = '''            ctx.mp.n = 16u;                     // [p72] n 轴 = A tile 的 16 条 lane'''

OLD5 = '''        ctx.fx.nSize = static_cast<uint16_t>(nTile);
        ctx.fx.dstStride = nTile;'''
NEW5 = '''        ctx.fx.nSize = 16u;                     // [p72]
        ctx.fx.dstStride = 16u;                 // [p72] 环行距跟着变成 16 个 float'''

OLD6 = '''        const uint32_t nTile = (m + 15u) & ~15u;'''
NEW6 = '''        const uint32_t nTile = 16u; (void)m;    // [p72] 与 AIC 的 fx.nSize 同口径'''

# 7)+8) 【本轮的判决位，也是候选修法】环只用【一条槽】，且永远落在本单元自己的
#        attention_out 行上 ⇒ 奇数片不再写 query 张量（两侧同改：只改一侧就变成读/写不同址）。
OLD7 = '''        const GlobalTensor<float> &rg = ((aChunk_ & 1u) != 0u) ? ringQP_Gm_ : ringOutGm_;'''
NEW7 = '''        const GlobalTensor<float> &rg = ringOutGm_;   // [p72] 只读输出行'''
OLD8 = '''        if ((cp_ & 1u) != 0u) {
            Fixpipe(ringQP_Gm_[fbase], ctx.l0c, ctx.fx);
        } else {
            Fixpipe(ringOutGm_[fbase], ctx.l0c, ctx.fx);
        }'''
NEW8 = '''        Fixpipe(ringOutGm_[fbase], ctx.l0c, ctx.fx);  // [p72] 单槽：绝不写 query'''

for idx, (old, new) in enumerate(((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3),
                                  (OLD4, NEW4), (OLD5, NEW5), (OLD6, NEW6),
                                  (OLD7, NEW7), (OLD8, NEW8)), 1):
    assert src.count(old) == 1, 'anchor %d x%d' % (idx, src.count(old))
    src = src.replace(old, new, 1)
    print('PATCHED_%d' % idx)
io.open(p, 'w', encoding='utf-8').write(src)
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p72_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p72_build.log | head -6")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p72_orig.cpp $K; bash build.sh > /tmp/p72_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P70_ABORT"; exit 1;;
esac

echo "--- RUN (write -> /tmp/p72_act) ---"
ns "$ENVR; mkdir -p /tmp/p72_act; for cs in p1 p2; do timeout $TMO ./test_sfa_dev cases/\$cs.bin 1 write 2e-3 1e-2 /tmp/p72_act > /tmp/p72r_\$cs.log 2>&1; echo \"\$cs rc=\$?\"; done; ls /tmp/p72_act"

echo "--- PULL ---"
timeout 300 ssh -F $S $H "cd /tmp/p72_act && tar cf - ." 2>/dev/null | (mkdir -p /tmp/p72/dump && tar xf - -C /tmp/p72/dump) && ls -la /tmp/p72/dump

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p72_orig.cpp $K; md5sum $K"
ns "$ENVR; bash build.sh > /tmp/p72_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P70_DONE"
