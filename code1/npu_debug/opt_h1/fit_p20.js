// [题1/p20 离线判读工装，非提交面] 反向小档阶梯的三点式拟合：T = A + B·blk + k·(S/blk) [+ C·io/blk]
// 用法: node fit_p20.js <p20_ladder.log> [tpcS=64]
// 目的：把 §24.5 那句"≥2.2µs 是与 io 无关的固定开销"换成**实测可分解的三项**，
//       并直接回答两问：① io/字节项在 [48,160]KiB 里到底有没有贡献；② 三项里还有没有 ≥1.6 分（≈1.5µs）的可压项。
const fs = require('fs');
const lines = fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/).filter(l => l.startsWith('S '));
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;

const cells = {};
for (const l of lines) {
    const arm = l.match(/arm=(\w+)/)[1];
    if (arm === 'REF' || arm === 'SELFTEST') continue;      // REF/IDENT 只用于相位干净性，不进拟合
    const blk = +l.match(/ b(\d+)/)[1];
    const shp = l.match(/shape=(\S+)/)[1].split(',');       // bwd,fp16,S,D,m
    const S = +shp[2], D = +shp[3], m = +shp[4];
    const ioK = (m + 1) * S * D * 2 / 1024;
    const key = `${ioK}|${blk}`;
    (cells[key] = cells[key] || { ioK, blk, S, D, m, y: [], raw: [] }).y.push(+l.match(/mean=([0-9.]+)/)[1]);
}
const R = Object.values(cells).map(c => ({ ...c, y: mean(c.y) })).sort((a, b) => a.blk - b.blk || a.ioK - b.ioK);
console.log(`拟合点 ${R.length}（每点 = 该格 X 臂 ${Object.values(cells)[0].y.length} 个原始点均值；相位已由 8-run 块平衡掉）`);

// 最小二乘（正规方程 + 列主元高斯消元），列 = [1, blk, S/blk, (io/blk)]
function lsq(cols, y) {
    const n = cols.length;
    const M = Array.from({ length: n }, (_, i) => Array.from({ length: n },
        (_, j) => cols[i].reduce((s, x, k2) => s + x * cols[j][k2], 0)));      // X^T X
    const v = Array.from({ length: n }, (_, i) => cols[i].reduce((s, x, k2) => s + x * y[k2], 0));
    for (let i = 0; i < n; i++) {
        let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
        [M[i], M[p]] = [M[p], M[i]]; [v[i], v[p]] = [v[p], v[i]];
        const d = M[i][i]; for (let c = i; c < n; c++) M[i][c] /= d; v[i] /= d;      // 主行归一
        for (let r = 0; r < n; r++) {
            if (r === i) continue;
            const f = M[r][i]; if (!f) continue;
            for (let c = i; c < n; c++) M[r][c] -= f * M[i][c]; v[r] -= f * v[i];
        }
    }
    return v;
}
const y = R.map(c => c.y);
const fit3 = lsq([R.map(() => 1), R.map(c => c.blk), R.map(c => c.S / c.blk)], y);
const fit4 = lsq([R.map(() => 1), R.map(c => c.blk), R.map(c => c.S / c.blk), R.map(c => c.ioK / c.blk)], y);
const pred = (f, c, four) => f[0] + f[1] * c.blk + f[2] * c.S / c.blk + (four ? f[3] * c.ioK / c.blk : 0);
const rms = f => Math.sqrt(mean(R.map(c => (c.y - pred(f, c, f.length === 4)) ** 2)));

console.log('\n=== ⑤ 三项拟合（µs）：T = A + B·blk + k·(S/blk) ===');
console.log(`  A=${fit3[0].toFixed(3)}µs  B=${(fit3[1] * 1000).toFixed(1)}ns/核  k=${(fit3[2] * 1000).toFixed(1)}ns/(核·行)   RMS=${rms(fit3).toFixed(3)}µs`);
console.log('  逐格残差（|r|>0.1 才算模型没吃住）：');
for (const c of R) console.log(`   io${String(c.ioK).padStart(3)}k blk=${String(c.blk).padStart(2)}  实测${c.y.toFixed(2)}  三式${pred(fit3, c).toFixed(2)}(r${(c.y - pred(fit3, c) >= 0 ? '+' : '')}${(c.y - pred(fit3, c)).toFixed(2)})  四式${pred(fit4, c, 1).toFixed(2)}(r${(c.y - pred(fit4, c, 1) >= 0 ? '+' : '')}${(c.y - pred(fit4, c, 1)).toFixed(2)})`);
console.log(`\n=== ⑥ 字节项要不要？===`);
console.log(`  三式 RMS=${rms(fit3).toFixed(3)}µs  →  四式 RMS=${rms(fit4).toFixed(3)}µs，加进来的 C=${(fit4[3] * 1000).toFixed(2)}ns/KiB（前向标定值 19ns/KiB）`);
console.log(`  ⇒ C 相对零的可辨识度：|C|/19 = ${Math.abs(fit4[3] * 1000 / 19).toFixed(2)}，四式把 A 挪到 ${fit4[0].toFixed(3)}（三式 ${fit3[0].toFixed(3)}）`);
for (const b of [...new Set(R.map(c => c.blk))].sort((x, z) => x - z)) {
    const g = R.filter(c => c.blk === b);
    const slope = (g[g.length - 1].y - g[0].y) / (g[g.length - 1].ioK - g[0].ioK);
    console.log(`   blk=${String(b).padStart(2)}: 实测 ${g.map(c => `io${c.ioK}k=${c.y.toFixed(2)}`).join(' ')}   ⇒ 跨 io 斜率 ${(slope * 1000).toFixed(2)}ns/KiB`);
}

const [A, B, k] = fit3;
const blkStar = Math.sqrt(k * R[0].S / B);
const Tmin = A + 2 * Math.sqrt(B * k * R[0].S);
console.log(`\n=== ⑦ 连续最优与"还剩多少"===`);
console.log(`  blk* = ${blkStar.toFixed(1)} ⇒ T_min = ${Tmin.toFixed(2)}µs（自然 blk=16 实测 ${mean(R.filter(c => c.blk === 16).map(c => c.y)).toFixed(2)}µs ⇒ 距连续最优 ${(mean(R.filter(c => c.blk === 16).map(c => c.y)) - Tmin).toFixed(2)}µs）`);
console.log(`  三项在 blk=16 处的占比：A=${A.toFixed(2)}(${(100 * A / Tmin).toFixed(0)}%)  B·blk=${(B * 16).toFixed(2)}  k·S/blk=${(k * 64 / 16).toFixed(2)}  可压预算合计 ${(B * 16 + k * 64 / 16).toFixed(2)}µs`);
const TBEST = 1.48;
console.log(`\n=== ⑧ 要够到平台 c5 的 tbest=${TBEST}µs，工作项预算只剩 ${(TBEST - A).toFixed(2)}µs ===`);
console.log(`  当前工作项 = ${(B * 16 + k * 64 / 16).toFixed(2)}µs ⇒ 需压到 ${(100 * (TBEST - A) / (B * 16 + k * 64 / 16)).toFixed(0)}%；`);
console.log(`  等价说法：在 A=${A.toFixed(2)} 不动的前提下，64 行反向的"每核每行"成本要从 ${(k * 1000).toFixed(0)}ns 降到 ${((TBEST - A) * 1000 / (64 / 16)).toFixed(0)}ns（blk=16、4 行/核）`);
