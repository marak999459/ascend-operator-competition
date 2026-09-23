#!/bin/bash
# P71 = P70 的 Gram 探针 + **环行号 ×4**：把 AIV 读回的环行换成 lane {0,4,8,12} 再看一次。
#
# p70 实测（见 code3.md §15.72(j)）：dump 出来的 16 格与"只用 p*.bin 的 Q/Qrope 本地算的
# Gram = Q·Qᵀ"**逐格吻合**（p1 每头 12/16 格、p2 8/16 格，对角 8.24/8.42/8.84 全中）
# ⇒ 布局模型（ND2NZ 的 lane=C0 轴、LoadData 分形序、Mmad 不转置、Fixpipe 行距）全部无罪。
# 剩下的 NaN 集合是"列 j ≡ 0 (mod N1) + 环行 0 整行"，两种解释仍然同形（它们给出的观测集
# 完全一样），必须换坐标才分得开：
#   (L) tile 的 lane {0,4,8,12} 里是 NaN ⇒ 被毒的 lane 做 m 轴时**整行**该是 NaN；
#   (C) 写出/读回侧按列丢格（Fixpipe 或环被覆盖）⇒ lane 4/8/12 只有"列 ≡0 mod N1"该 NaN。
# 本轮就是把可见行从 {0,1,2,3} 挪到 {0,4,8,12}：L 预言四行全 NaN，C 预言只有行 0 全 NaN。
# （沿用 p70 的 nValue=16 ⇒ lane 4/8/12 是真实 GM 行，期望值仍是本地可算的 Gram。）
# ⚠️ 只打远端 ~/sfa_real；跑完 RESTORE + 重建成净版。判据是**读回来的数**，不是 PASS/FAIL。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p71_orig.cpp; md5sum /tmp/kernel_p71_orig.cpp"

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
NEW1 = '''                const float l = ml.GetValue(halfOff_ + i);   // [p71] LSE 照旧
                LocalTensor<float> oi = o[i * rowC];
                Adds(oi, sBuf_.Get<float>()[i * nBlk_], 0.0f, 16u);   // [p71] Gram 行
                Duplicate(oi[16], 32768.0f, rowC - 16u);              // [p71] 32768=哨兵'''

# 2) A tile 的 nValue 补齐 16 个真实行（p69 那一档，让 lane↔GM 行 的映射唯一）
OLD2 = '''        DataCopy(ctx.l1qa, qGm_[cs1_], ctx.nzA);
        ctx.nzAr.nValue = N1_;
        DataCopy(ctx.l1qr, qrGm_[ropeBase], ctx.nzAr);'''
NEW2 = '''        {                                                                   // [p71]
            const uint64_t qTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)D_;   // [p71]
            const uint64_t rTot = (uint64_t)B_ * S1_ * N1_ * (uint64_t)Dr_;  // [p71]
            const uint64_t aNeed = 16ull * (uint64_t)D_;                     // [p71]
            const uint64_t rNeed = 16ull * (uint64_t)Dr_;                    // [p71]
            const bool bigA = (qTot >= aNeed);                               // [p71]
            const bool bigR = (rTot >= rNeed);                               // [p71]
            uint64_t ab = (uint64_t)cs1_;                                    // [p71]
            if (bigA && ab + aNeed > qTot) { ab = qTot - aNeed; }            // [p71]
            uint64_t rb = ropeBase;                                          // [p71]
            if (bigR && rb + rNeed > rTot) { rb = rTot - rNeed; }            // [p71]
            ctx.nzA.nValue = bigA ? 16u : (uint32_t)N1_;                     // [p71]
            DataCopy(ctx.l1qa, qGm_[ab], ctx.nzA);                           // [p71]
            ctx.nzAr.nValue = bigR ? 16u : (uint32_t)N1_;                    // [p71]
            DataCopy(ctx.l1qr, qrGm_[rb], ctx.nzAr);                         // [p71]
        }                                                                     // [p71]'''

# 3) 五个 k 切片：B 侧从 A tile 同址装载 ⇒ C = A·Aᵀ
OLD3 = '''            ctx.ldB.repeatTimes = static_cast<uint8_t>((nTile / 16u) * nF);
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1kr, ctx.ldB);
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1ka[j * nTile * kk], ctx.ldB);
            }'''
NEW3 = '''            ctx.ldB.repeatTimes = static_cast<uint8_t>(nF);              // [p71] B = A tile
            if (rp) {
                LoadData(l0a, ctx.l1qr, ctx.ldA);
                LoadData(l0b, ctx.l1qr, ctx.ldB);                        // [p71]
            } else {
                LoadData(l0a, ctx.l1qa[j * 16u * kk], ctx.ldA);
                LoadData(l0b, ctx.l1qa[j * 16u * kk], ctx.ldB);          // [p71]
            }'''

OLD4 = '''            ctx.mp.n = static_cast<uint16_t>(nTile);'''
NEW4 = '''            ctx.mp.n = 16u;                     // [p71] n 轴 = A tile 的 16 条 lane'''

OLD5 = '''        ctx.fx.nSize = static_cast<uint16_t>(nTile);
        ctx.fx.dstStride = nTile;'''
NEW5 = '''        ctx.fx.nSize = 16u;                     // [p71]
        ctx.fx.dstStride = 16u;                 // [p71] 环行距跟着变成 16 个 float'''

OLD6 = '''        const uint32_t nTile = (m + 15u) & ~15u;'''
NEW6 = '''        const uint32_t nTile = 16u; (void)m;    // [p71] 与 AIC 的 fx.nSize 同口径'''

# 7) 【本轮的判决位】把 AIV 读的环行号从 headBase_+i 改成 4*(headBase_+i)
#    ⇒ p1 的四条可见行变成 lane {0,4,8,12} = p70 里"可疑的那四条"。
#      · 若这四行**整行全 NaN**  => 毒在 tile 的 lane 上（读回的列图案只是同一条 lane 的另一半）；
#      · 若 lane 4/8/12 只是**列 ≡0 mod N1 为 NaN**、对角 8.x 正常
#                                => 毒在 n 轴/写出侧，tile 无罪，row 0 的整行 NaN 另有原因。
OLD7 = '''            DataCopy(sc[i * nBlk_],
                     rg[base + static_cast<uint64_t>(headBase_ + i) * nTile], dcp);'''
NEW7 = '''            DataCopy(sc[i * nBlk_],
                     rg[base + static_cast<uint64_t>(4u * (headBase_ + i)) * nTile],   // [p71]
                     dcp);                                                              // [p71]'''

for idx, (old, new) in enumerate(((OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3),
                                  (OLD4, NEW4), (OLD5, NEW5), (OLD6, NEW6),
                                  (OLD7, NEW7)), 1):
    assert src.count(old) == 1, 'anchor %d x%d' % (idx, src.count(old))
    src = src.replace(old, new, 1)
    print('PATCHED_%d' % idx)
io.open(p, 'w', encoding='utf-8').write(src)
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p71_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p71_build.log | head -6")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p71_orig.cpp $K; bash build.sh > /tmp/p71_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P70_ABORT"; exit 1;;
esac

echo "--- RUN (write -> /tmp/p71_act) ---"
ns "$ENVR; mkdir -p /tmp/p71_act; for cs in p1 p2; do timeout $TMO ./test_sfa_dev cases/\$cs.bin 1 write 2e-3 1e-2 /tmp/p71_act > /tmp/p71r_\$cs.log 2>&1; echo \"\$cs rc=\$?\"; done; ls /tmp/p71_act"

echo "--- PULL ---"
timeout 300 ssh -F $S $H "cd /tmp/p71_act && tar cf - ." 2>/dev/null | (mkdir -p /tmp/p71/dump && tar xf - -C /tmp/p71/dump) && ls -la /tmp/p71/dump

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p71_orig.cpp $K; md5sum $K"
ns "$ENVR; bash build.sh > /tmp/p71_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P70_DONE"
