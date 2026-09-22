// [题3/P0.5 离线核对，非提交面] 复刻 host 的 CalcUbNeed/CalcBlocking 与 kernel Init 的真实占用，
// 用来判"host 预算式写死 2 字节"在 fp32 实例上会不会把 UB 撑过物理容量。纯算术，不上机。
// 用法: node ub_budget_check.js
const UB_PHYS = 196608n, UB_PCT = 95n;
const NB = [32, 16, 8, 4, 2, 1], NBLK = [128, 64, 32, 16, 8, 4, 2, 1];

// host 现行（op_host/sparse_flash_attention.cpp:39-49）—— K/V/kr 写死 2 字节、没有 lseBuf_
const hostNeed = (nb, nBlk, qD, dr) =>
    BigInt(nb) * BigInt(qD + dr) * 4n
    + BigInt(nb) * BigInt(qD) * 4n
    + BigInt(nBlk) * BigInt(qD) * 2n * 2n
    + BigInt(nBlk) * BigInt(dr) * 2n
    + BigInt(nb) * BigInt(nBlk) * 4n * 2n
    + BigInt(nb) * 12n
    + BigInt(nBlk) * 4n;

// kernel Init 真实占用（op_kernel/sparse_flash_attention.cpp:168-177）—— sz = sizeof(DT_QUERY)，且多一格 lseBuf_
const trueNeed = (nb, nBlk, qD, dr, sz) =>
    BigInt(nb) * BigInt(qD + dr) * 4n
    + BigInt(nb) * BigInt(qD) * 4n
    + BigInt(nBlk) * BigInt(qD) * BigInt(sz) * 2n
    + BigInt(nBlk) * BigInt(dr) * BigInt(sz)
    + BigInt(nb) * BigInt(nBlk) * 4n * 2n
    + BigInt(nb) * 12n
    + BigInt(nBlk) * 4n
    + BigInt(nb) * 8n;

// 修好的 host 式：dtype 感知 + 补 lseBuf_（= trueNeed）
const needFor = (mode) => (nb, nBlk, qD, dr, sz) => (mode === 'host' ? hostNeed(nb, nBlk, qD, dr) : trueNeed(nb, nBlk, qD, dr, sz));

function choose(need, ubSafe, sbs, qD, dr, sz) {
    let best = { nb: 1, nBlk: 1, score: 0n };
    for (const nb of NB) for (const nBlk of NBLK) {
        if (nb * nBlk < sbs) continue;
        if (need(nb, nBlk, qD, dr, sz) > ubSafe) continue;
        const score = BigInt(nb) * 100000n + BigInt(nBlk);
        if (score > best.score) best = { nb, nBlk, score };
    }
    return best;
}

const ubSafe = (UB_PHYS / 100n) * UB_PCT;
console.log(`ubSafe = ${ubSafe} B (物理 ${UB_PHYS} B 的 95%)`);
for (const [qD, dr] of [[512, 64], [128, 64], [64, 64]]) {
    console.log(`\n=== Q_D=${qD}, Dr=${dr} ===`);
    console.log('dtype sbs  | 现行:nb/nblk  现行按host式  真实占用  超物理? | 修后:nb/nblk  修后占用  仍超?');
    for (const sz of [2, 4]) {
        for (const sbs of [1, 16, 32, 64, 128]) {
            const c = choose(needFor('host'), ubSafe, sbs, qD, dr, sz);
            const t = trueNeed(c.nb, c.nBlk, qD, dr, sz);
            const f = choose(needFor('fixed'), ubSafe, sbs, qD, dr, sz);
            const ft = trueNeed(f.nb, f.nBlk, qD, dr, sz);
            const tag = sz === 2 ? 'fp16' : 'fp32';
            const hn = String(hostNeed(c.nb, c.nBlk, qD, dr));
            console.log(
                `${tag} ${String(sbs).padStart(3)}  | ${String(c.nb).padStart(3)}/${String(c.nBlk).padStart(3)}  ` +
                `${hn.padStart(11)} ${String(t).padStart(11)}  ${(t > UB_PHYS ? '★超' : 'ok').padEnd(5)} | ` +
                `${String(f.nb).padStart(3)}/${String(f.nBlk).padStart(3)}  ${String(ft).padStart(11)}  ${(ft > UB_PHYS ? '★超' : 'ok').padEnd(4)}`);
        }
    }
}

// ==== 第二张表：P1/P3 真正要问的问题 —— 真实预算下 n_blk 的天花板在哪一格 ====
// 对每个 nb 求"真实占用 <= ubSafe"的最大 n_blk（离散候选 + 连续上界），
// 并单独判一条硬事实：sbs=128 的整块（K+V 各 128 token）在 Q_D=512 下物理上装不装得下。
console.log('\n\n########## n_blk 天花板（按 kernel 真实占用，ubSafe 口径）##########');
for (const [qD, dr] of [[512, 64], [128, 64]]) {
    for (const sz of [2, 4]) {
        const tag = sz === 2 ? 'fp16' : 'fp32';
        console.log(`\n--- Q_D=${qD}, Dr=${dr}, ${tag} ---`);
        console.log(' nb  | 最大候选n_blk  连续上界  真实占用  余量(B)  | nb*n_blk(连续token窗口)');
        for (const nb of NB) {
            let bestBlk = 0;
            for (const nBlk of NBLK) {
                if (trueNeed(nb, nBlk, qD, dr, sz) <= ubSafe) { if (nBlk > bestBlk) bestBlk = nBlk; }
            }
            const per = BigInt(qD) * BigInt(sz) * 2n + BigInt(dr) * BigInt(sz) + BigInt(nb) * 8n + 4n;
            const fixedN = BigInt(nb) * BigInt(qD + dr) * 4n + BigInt(nb) * BigInt(qD) * 4n + BigInt(nb) * 12n + BigInt(nb) * 8n;
            const cap = per > 0n ? (ubSafe - fixedN) / per : 0n;
            const used = bestBlk ? trueNeed(nb, bestBlk, qD, dr, sz) : 0n;
            console.log(
                String(nb).padStart(3) + '  |' + (bestBlk ? String(bestBlk) : '  无  ').padStart(13) +
                String(cap).padStart(10) + String(used).padStart(11) + String(ubSafe - used).padStart(9) +
                '  |  ' + String(nb * bestBlk));
        }
        const kvOnly = 128n * BigInt(qD) * BigInt(sz) * 2n + 128n * BigInt(dr) * BigInt(sz);
        console.log(`   · 128 token 的 K+V+kr 净字节 = ${kvOnly} B → ` +
            (kvOnly <= UB_PHYS ? '还能与别的缓冲共存' : '★ 光这三样已超物理 UB，"整块 128 token 一次进 UB"不可能'));
    }
}
