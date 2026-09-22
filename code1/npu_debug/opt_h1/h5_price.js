// [题1/p20 第 1.5 阶离线判读工装，非提交面] H5（反向行合批）的价签：不在纸上算，
// 全部从两份真机日志现推 —— 先重拟合 (A,B,k0,k1)，再按"行项被摊掉多少"分三档给
// ΔT(本地) → ΔT(平台，按落档的两条同格读数定比例) → Δscore(两式精确差分)。
// 用法: node h5_price.js <p20b.log> <p20_stage1.log> [tbest5=1.48] [our5=4.16]
const fs = require('fs');
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const cells = {};
for (const f of process.argv.slice(2, 4)) {
    for (const l of fs.readFileSync(f, 'utf8').split(/\r?\n/)) {
        if (!l.startsWith('S ') || /arm=IDENT|arm=SELFTEST/.test(l)) continue;
        const blk = +l.match(/ b(\d+)/)[1];
        const shp = l.match(/shape=(\S+)/)[1].split(',');
        (cells[shp.slice(0, 5).join(',') + '|b' + blk] = cells[shp.slice(0, 5).join(',') + '|b' + blk] || []).push(+l.match(/mean=([0-9.]+)/)[1]);
    }
}
const C = Object.entries(cells).map(([k, v]) => {
    const [shp, b] = k.split('|'); const s = shp.split(',');
    return { k, S: +s[2], D: +s[3], m: +s[4], blk: +b.slice(1), T: mean(v), dir: s[0] };   // key 里的 blk 段带 'b' 前缀，必须剥掉
});
// 反向价签只能吃反向格。第二份日志给错（比如给了 p20c 那份前向 A/B）时，前向格会被
// 当作反向样本进最小二乘，且合并/不合并两臂在 shape key 上撞成同一格、被平均掉 ——
// 症状是格数不对 + 系数离谱（实测 k1 从 107 掉到 62）。宁可停，不要给一个能看的数。
const bad = C.filter(c => c.dir !== 'bwd');
if (bad.length) {
    console.error(`ABORT: ${bad.length} 格不是反向（${[...new Set(bad.map(c => c.k))].slice(0, 4).join('  ')} …）`);
    console.error('       用法 node h5_price.js <p20b 反向日志> <p20_ladder stage-1 反向日志> —— 别把 p20c(前向合批) 当第二参。');
    process.exit(7);
}
function solve(M, v) {
    const n = v.length;
    for (let i = 0; i < n; i++) {
        let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
        [M[i], M[p]] = [M[p], M[i]]; [v[i], v[p]] = [v[p], v[i]];
        const d = M[i][i]; for (let c = i; c < n; c++) M[i][c] /= d; v[i] /= d;
        for (let r = 0; r < n; r++) { if (r === i) continue; const f = M[r][i]; if (!f) continue;
            for (let c = i; c < n; c++) M[r][c] -= f * M[i][c]; v[r] -= f * v[i]; }
    }
    return v;
}
const F = [c => 1, c => c.blk, c => c.S / c.blk, c => c.m * c.S / c.blk];
const n = F.length, M = Array.from({ length: n }, () => new Array(n).fill(0)), v = new Array(n).fill(0);
for (const c of C) { const x = F.map(f => f(c)); for (let i = 0; i < n; i++) { v[i] += x[i] * c.T; for (let j = 0; j < n; j++) M[i][j] += x[i] * x[j]; } }
const [A, B, K0, K1] = solve(M, v);
const KL = K0 + K1;                                   // m=1 每行单价
console.log(`重拟合（${C.length} 格）：A=${A.toFixed(4)}µs  B=${(B * 1000).toFixed(1)}ns/核  每行=${(K0 * 1000).toFixed(0)}+${(K1 * 1000).toFixed(0)}m ns  ⇒ m=1 行项 ${(KL * 1000).toFixed(0)}ns`);
const Tnow = (blk, S = 64, m = 1) => A + B * blk + (K0 + K1 * m) * S / blk;
const Tmer = (blk, f, S = 64, m = 1) => A + B * blk + f * (K0 + K1 * m) * S / blk;   // f = 行项残余比例
// 比例：平台 c5 与本地同格读数之比（两个数都取自已落档文件，不在这里发明）
const REF_LOCAL = Tnow(16), RATIO = (+process.argv[5] || 4.16) / (C.find(c => c.k === 'bwd,fp16,64,384,1|b16') || { T: REF_LOCAL }).T;
console.log(`本地基准格 bwd,fp16,64,384,1@b16 = ${(C.find(c => c.k === 'bwd,fp16,64,384,1|b16') || { T: REF_LOCAL }).T.toFixed(2)}µs；平台 c5 = ${process.argv[5] || 4.16} ⇒ 比例 ${RATIO.toFixed(3)}×（⚠️ 单点推断）`);
const tb = +process.argv[4] || 1.48, our = +process.argv[5] || 4.16;
const lin = t => 12.5 * tb * (1 / t - 1 / our);
const pow = t => 12.5 * (Math.pow(tb / t, 1.5) - Math.pow(tb / our, 1.5));
console.log(`\n两式（线性 = 12.5·tbest·(1/Tn−1/To)；幂 = 12.5·[(tbest/Tn)^1.5−(tbest/To)^1.5]）；tbest5=${tb} 我方 c5=${our}`);
console.log('校核：幂式取"挤到全场 min=1.52"给 ' + pow(1.52).toFixed(2) + ' 分（§24.9 表上记的是 9.36，差 ' + (100 * (pow(1.52) - 9.36) / 9.36).toFixed(1) + '%）\n');
const BASE = Tnow(16);
const row = (name, blk, f, note) => {
    const tl = Tmer(blk, f), tp = tl * RATIO;
    console.log(`  ${name.padEnd(30)} blk=${String(blk).padStart(2)} 行项残余 ${String((f * 100).toFixed(0) + '%').padStart(4)} | 本地 ${BASE.toFixed(2)}→${tl.toFixed(2)}（ΔT ${(BASE - tl).toFixed(2)}µs）| 平台 ${our}→${tp.toFixed(2)} | ${lin(tp).toFixed(2)} / ${pow(tp).toFixed(2)} 分 | ${note}`);
};
console.log(`=== 三档（现值一律取当前工作点 blk=16 = 本地 ${BASE.toFixed(2)}µs；激进档才允许 blk 左移）===`);
row('保守：只摊掉 DMA 发起', 16, 0.40, '留下 Cast/Add/Duplicate 那几次');
row('中位：L=tpc=4 行项变加数', 16, 1 / 4, '26.4-② 前后向之差支撑');
row('激进：整核一次搬运', 8, 0, '行项→0，只剩 A+B·blk+bytes');
row('激进 blk=4', 4, 0, '同上，B 再省 0.26µs');
row('激进 blk=2', 2, 0, '下界：再小就要吃 io/核 的字节项（本轮测不出来）');
console.log('\n=== 门槛反解：过 target 分需要平台 c5 ≤ ?（线性式）===');
for (const target of [1.6, 2.0]) {
    const need = 1 / (1 / our + target / (12.5 * tb));
    console.log(`  ${target} 分 ⇒ 平台 ≤ ${need.toFixed(2)}µs ⇒ 本地 ≤ ${(need / RATIO).toFixed(2)}µs ⇒ ΔT ≥ ${(our - need).toFixed(2)}µs（平台口径）`);
}
console.log('\n=== S 轴敏感性（26.7-④：判分档的 S 未知，倍数挂在 tpc=S/blk 上；一律取 blk=16 比较）===');
console.log('    ⚠️ 只看 ΔT 两列：末列分数把"平台现值"一律钉在 our=4.16（那是 S=64 那一格的榜面值），');
console.log('       S≠64 时现值并非 4.16 ⇒ 分数列在四行里恒等于 1.72/1.68，**是钉出来的假象，不是结论**。');
for (const S of [32, 64, 128, 256]) {
    const t0 = Tnow(16, S), t1 = Tmer(16, 0, S);
    console.log(`  S=${String(S).padStart(3)}（tpc=${(S / 16).toFixed(0)} 行/核）: 现值(本地) ${t0.toFixed(2)} → 行项清零 ${t1.toFixed(2)} ⇒ ΔT ${(t0 - t1).toFixed(2)}µs（本地）/ ${((t0 - t1) * RATIO).toFixed(2)}µs（平台口径）⇒ ${lin(t1 * RATIO).toFixed(2)} / ${pow(t1 * RATIO).toFixed(2)} 分`);
}
