// [题1/p20 离线判读工装，非提交面] c5 固定开销分解第 1 阶：反向小档 io×blk 阶梯 + 相位平衡块。
// 用法: node agg_p20.js <p20_ladder.log>
// 块结构（run_p20_ladder.sh 的 pair()）：每 8 行一组 = R,X,X,R | X,R,R,X ⇒ 每臂各占慢位{1,4}与快位{2,3}一次。
// 报告三件事：① 每格 X 与同块 REF 的 Δ（对相位无偏）；② 观测量对 §23.5 地板模型的**超出量**是否随 io 走；
//            ③ 构造性恒等对照（IDENT ≡ REF）给出的 residual 相位 p —— 它必须是 0 附近，否则整组判脏。
const fs = require('fs');
const lines = fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/).filter(l => l.startsWith('S '));
const pts = lines.map(l => ({
    arm: l.match(/arm=(\w+)/)[1],
    blk: +(l.match(/ b(\d+)/) || [0, 0])[1],
    round: +l.match(/ r(\d+)/)[1],
    slot: +l.match(/slot(\d)/)[1],
    got: +l.match(/got=(\d+)/)[1],
    ver: /ALL PASS/.test(l) ? 'PASS' : (/FAIL=\d+/.test(l) ? l.match(/FAIL=\d+/)[0] : 'NOVERDICT'),
    mean: +(l.match(/mean=([0-9.]+)/) || [0, NaN])[1],
    shape: (l.match(/shape=(\S+)/) || [0, ''])[1],
}));
const self = pts.filter(p => p.arm === 'SELFTEST');
const cells = pts.filter(p => p.arm !== 'SELFTEST');
if (self.length) console.log(`自检：${self[0].shape} got=${self[0].got} ${self[0].ver} mean=${self[0].mean}`);
const bad = cells.filter(p => p.ver !== 'PASS');
console.log(`读数点 ${cells.length}（非 PASS ${bad.length}${bad.length ? ' ⇒ ' + [...new Set(bad.map(b => b.arm + ':' + b.ver))].join(' ') : ''}）`);

// 每 8 点一组 = 一个 pair（X 对 REF）
const chunks = [];
for (let i = 0; i + 8 <= cells.length; i += 8) chunks.push(cells.slice(i, i + 8));
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const spread = a => a.length ? Math.max(...a) - Math.min(...a) : NaN;
const f1 = x => (isNaN(x) ? '  —  ' : x.toFixed(2).padStart(6));

const IoK = shp => {   // io(KiB) = (m+1)*S*D*2 / 1024，本扫描固定 m=1,S=64 ⇒ = D/4
    const D = +shp.split(',')[3];
    const m = +shp.split(',')[4], S = +shp.split(',')[2];
    return (m + 1) * S * D * 2 / 1024;
};

console.log('\n=== ① 每格 vs 同块 REF（相位平衡后的 Δ）===');
console.log('  io(KiB) blk | X均值  散布  n | 同块REF均值  Δ(X−REF) 逐块Δ        | X原始点');
const byCell = {};
for (const c of chunks) {
    const X = c.filter(p => p.arm !== 'REF'), R = c.filter(p => p.arm === 'REF');
    if (!X.length || !R.length) continue;
    const key = `${IoK(X[0].shape)}|${X[0].blk}`;
    (byCell[key] = byCell[key] || { arms: [], refs: [], deltas: [], tag: X[0].arm }).arms.push(...X.map(p => p.mean));
    byCell[key].refs.push(...R.map(p => p.mean));
    byCell[key].deltas.push(mean(X.map(p => p.mean)) - mean(R.map(p => p.mean)));
    byCell[key].raw = (byCell[key].raw || []).concat(X.map(p => p.mean.toFixed(1)));
}
const rowsOut = [];
for (const [key, v] of Object.entries(byCell)) {
    const [io, blk] = key.split('|').map(Number);
    const mx = mean(v.arms), mr = mean(v.refs);
    rowsOut.push({ io, blk, mx, mr, d: mx - mr, spread: spread(v.arms), n: v.arms.length, deltas: v.deltas, raw: v.raw, tag: v.tag });
}
rowsOut.sort((a, b) => a.io - b.io || a.blk - b.blk);
for (const r of rowsOut) {
    console.log(`  ${String(r.io).padStart(5)}   b${String(r.blk).padStart(2)} | ${f1(r.mx)} ${f1(r.spread)} ${String(r.n).padStart(3)} | ${f1(r.mr)}   ${f1(r.d)}  [${r.deltas.map(x => (x >= 0 ? '+' : '') + x.toFixed(2)).join(' ')}] | ${r.raw.join(' ')}`);
}

console.log('\n=== ② 对 §23.5 地板模型 T = A + B·blk + C·io/blk 的超出量 ===');
console.log('   （A=1.10~1.20µs、B=75ns/核、C=19ns/KiB —— 三个常数都是**前向**标定的，见 §24.7-③）');
console.log('  io(KiB) blk   实测    模型(1.15)  超出   |  同 blk 跨 io 的超出量是否相等？');
for (const A of [1.10, 1.15, 1.20]) {
    const ex = r => r.mx - (A + 0.075 * r.blk + 0.019 * r.io / r.blk);
    if (A !== 1.15) continue;
    for (const b of [...new Set(rowsOut.map(r => r.blk))].sort((x, y) => x - y)) {
        const g = rowsOut.filter(r => r.blk === b && r.tag !== 'IDENT');
        console.log(`  全部 blk=${String(b).padStart(2)}: ` + g.map(r => `io${r.io}k 实测${r.mx.toFixed(2)} 模型${(A + 0.075 * b + 0.019 * r.io / b).toFixed(2)} 超出${ex(r) >= 0 ? '+' : ''}${ex(r).toFixed(2)}`).join('   '));
    }
}
console.log('\n  以 A 为自由参数反解"每格刚好对上实测"所需的最小固定项：');
for (const b of [...new Set(rowsOut.map(r => r.blk))].sort((x, y) => x - y)) {
    const g = rowsOut.filter(r => r.blk === b && r.tag !== 'IDENT');
    if (!g.length) continue;
    const need = g.map(r => r.mx - 0.075 * b - 0.019 * r.io / b);
    console.log(`   blk=${String(b).padStart(2)}  A_needed = ${need.map(x => x.toFixed(2)).join(' / ')}  （极差 ${f1(Math.max(...need) - Math.min(...need))}）`);
}

console.log('\n=== ③ 构造性恒等对照 IDENT ≡ REF（真增益必为 0）===');
const ident = rowsOut.filter(r => r.tag === 'IDENT');
if (!ident.length) console.log('  （本轮还没跑到）');
for (const r of ident) console.log(`  p = ${r.d >= 0 ? '+' : ''}${r.d.toFixed(3)}µs   逐块 [${r.deltas.map(x => (x >= 0 ? '+' : '') + x.toFixed(2)).join(' ')}]  ⇒ |p| ${Math.abs(r.d) < 0.1 ? '< 0.1µs，本组相位干净' : '≥0.1µs，上面所有 Δ 要先减掉它'}`);

console.log('\n=== ④ 槽位相位本体（每臂 4 点按 slot 摊开）===');
for (const t of ['REF', ...new Set(rowsOut.map(r => r.tag))]) {
    const g = cells.filter(p => p.arm === t);
    if (!g.length) continue;
    const byslot = [1, 2, 3, 4].map(s => { const v = g.filter(p => p.slot === s).map(p => p.mean); return v.length ? mean(v).toFixed(2) + `(${v.length})` : '—'; });
    console.log(`  ${t.padEnd(8)} slot1..4 = ${byslot.join('  ')}   极差 ${f1(spread(g.map(p => p.mean)))}`);
}
