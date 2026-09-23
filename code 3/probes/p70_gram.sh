#!/bin/bash
# P70：把 B 侧操作数换成 A tile 自己 ⇒ Mmad 算的是 **Gram = Q·Qᵀ**（K 完全退出数据流）。
#
# 为什么要这一发：p68/p69 的环读回说明 L0C 里有 1~3 条 lane 是 NaN，但两种解释还分不清：
#   (H-A) 某个操作数读到【tile 外】的未初始化 L1（fp16 位模式 = NaN）；
#   (H-B) A/B 的 lane↔(头, k) 映射与我们推的 H2 式子不一致（结果有限但错行/错列）。
# 换成 Gram 之后，期望值可以只用 p1.bin 的 Q/Qrope 纯本地算出来（不需要 K、不需要 idx），
# 而且 Gram 有两条**先验不变量**：对称、对角 = scale·‖q‖² > 0。
#   NaN 还在（同一批 lane）              ⇒ H-A，而且既然 K 已经不在数据流里，越界点必在 A 侧；
#   有限但不对称 / 对角非正            ⇒ H-B，映射模型要重推；
#   与本地 Gram 逐格吻合              ⇒ A 侧无罪，NaN 来自 K 的 ND2NZ-gather（n 轴），
#                                     下一步就把 B 侧换回 l1ka 但只喂【一段连续】token。
# 叠加 p69 的 nValue=16（A tile 填 16 个真实行 ⇒ lane n = qGm_ 的第 n 行，期望唯一）：
#   p1 的 Q 总量正好 16 行 ⇒ 基址被夹到 0 ⇒ **每个单元的 A tile 都一样**，
#   所以预期"任意 (s,h) 的读回 = scale·Gram[h][j]"，与 s 无关（这本身就是一条可判性质）。
# ⚠️ 只打远端 ~/sfa_real；跑完 RESTORE + 重建成净版。判据是**读回来的数**，不是 PASS/FAIL。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p70_orig.cpp; md5sum /tmp/kernel_p70_orig.cpp"

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
NEW1 = '''                const float l = ml.GetValue(halfOff_ + i);   // [p70] LSE 照旧
                LocalTensor<float> oi = o[i * rowC];
                Adds(oi, sBuf_.Get<float>()[i * nBlk_], 0.0f, 16u);   // [p70] Gram 行
                Duplicate(oi[16], 0.0f, rowC - 16u);                  // [p70]'''

# 2) A tile 的 nValue 补齐 16 个真实行（p69 那一档，让 lane↔GM 行 的映射唯一）
OLD2 = '''        DataCopy(ctx.l1qa, qGm_[cs1_], ctx.nzA);
        ctx.nzAr.nValue = N1_;
        DataCopy(ctx.l1qr, qrGm_[ropeBase], ctx.nzAr);'''
NEW2 = '''        {                                                                   // [p70]
            const uint64_t qTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)D_;   // [p70]
            const uint64_t rTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)Dr_;  // [p70]
            const uint64_t aNeed = 16ull * (uint64_t)D_;                     // [p70]
            const uint64_t rNeed = 16ull * (uint64_t)Dr_;                    // [p70]
            const bool bigA = (qTot >= aNeed);                               // [p70]
            const bool bigR = (rTot >= rNeed);                               // [p70]
            uint64_t ab = (uint64_t)cs1_;                                    // [p70]
            if (bigA && ab + aNeed > qTot) { ab = qTot - aNeed; }            // [p70]
            uint64_t rb = ropeBase;                                          // [p70]
            if (bigR && rb + rNeed > rTot) { rb = rTot - rNeed; }            // [p70]
            ctx.nzA.nValue = bigA ? 16u : (uint32_t)N1_;                     // [p70]
            DataCopy(ctx.l1qa, qGm_[ab], ctx.nzA);                           // [p70]
            ctx.nzAr.nValue = bigR ? 16u : (uint32_t)N1_;                    // [p70]
            DataCopy(ctx.l1qr, qrGm_[rb], ctx.nzAr);                         // [p70]
        }                                                                     // [p70]'''

# 3) 五个 k 切片：B 侧从 A tile 同址装载 ⇒ C = A·Aᵀ
OLD3 = '''            ctx.ldB.repeatTimes = static_cast<uint8_t>((nTile / 16u) * nF);
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1kr, ctx.ldB);
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1ka[j * nTile * kk], ctx.ldB);
            }'''
NEW3 = '''            ctx.ldB.repeatTimes = static_cast<uint8_t>(nF);              // [p70] B = A tile
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1qr, ctx.ldB);                        // [p70]
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1qa[j * 16u * kk], ctx.ldB);          // [p70]
            }'''

OLD4 = '''            ctx.mp.n = static_cast<uint16_t>(nTile);'''
NEW4 = '''            ctx.mp.n = 16u;                     // [p70] n 轴 = A tile 的 16 条 lane'''

OLD5 = '''        ctx.fx.nSize = static_cast<uint16_t>(nTile);
        ctx.fx.dstStride = nTile;'''
NEW5 = '''        ctx.fx.nSize = 16u;                     // [p70]
        ctx.fx.dstStride = 16u;                 // [p70] 环行距跟着变成 16 个 float'''

OLD6 = '''        const uint32_t nTile = (m + 15u) & ~15u;'''
NEW6 = '''        const uint32_t nTile = 16u; (void)m;    // [p70] 与 AIC 的 fx.nSize 同口径'''

for idx, (old, new) in enumerate(((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3),
                                  (OLD4, NEW4), (OLD5, NEW5), (OLD6, NEW6)), 1):
    assert src.count(old) == 1, 'anchor %d x%d' % (idx, src.count(old))
    src = src.replace(old, new, 1)
    print('PATCHED_%d' % idx)
io.open(p, 'w', encoding='utf-8').write(src)
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p70_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p70_build.log | head -6")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p70_orig.cpp $K; bash build.sh > /tmp/p70_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P70_ABORT"; exit 1;;
esac

echo "--- RUN (write -> /tmp/p70_act) ---"
ns "$ENVR; mkdir -p /tmp/p70_act; for cs in p1 p2; do timeout $TMO ./test_sfa_dev cases/\$cs.bin 1 write 2e-3 1e-2 /tmp/p70_act > /tmp/p70r_\$cs.log 2>&1; echo \"\$cs rc=\$?\"; done; ls /tmp/p70_act"

echo "--- PULL ---"
timeout 300 ssh -F $S $H "cd /tmp/p70_act && tar cf - ." 2>/dev/null | (mkdir -p /tmp/p70/dump && tar xf - -C /tmp/p70/dump) && ls -la /tmp/p70/dump

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p70_orig.cpp $K; md5sum $K"
ns "$ENVR; bash build.sh > /tmp/p70_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P70_DONE"
