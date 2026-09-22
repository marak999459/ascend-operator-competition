// [题1/R21 p21 离线判读工装，非提交面] H5a 同构建 A/B 的读表器。
// 输入 = run_p21_h5a.sh 的日志（每格 8 发：块1 off,on,on,off | 块2 on,off,off,on）。
// 三件事，按 §26.8 的判据排：
//   ① 两条"由构造不可能走新分支"的对照臂（m=2 / 前向）必须先 ≈0，否则整轮读数作废（§23.27-②）；
//   ② 逐格 ΔT = mean(off) − mean(on)，并按轮次拆开看符号是否逐轮一致（12/12 那种硬度的证据）；
//   ③ 把 ΔT 折回 §26.5 的行项口径：f = 行项残余 = (T_on − A − B·blk) / ((k0+k1·m)·S/blk)，
//      再拿 h5_price.js 那两式给平台侧分数（比例 1.234 是单点推断，见 §26.7-③）。
// 用法: node agg_p21.js <p21.log> [A=1.3897] [B_ns=65.0] [k0_ns=145.1] [k1_ns=107.0] [ratio=1.234] [tbest=1.48] [our=4.16]
const fs = require('fs');
const P = [+process.argv[3] || 1.3897, +(process.argv[4] || 65.0) / 1000,
          +(process.argv[5] || 145.1) / 1000, +(process.argv[6] || 107.0) / 1000];
const RATIO = +process.argv[7] || 1.234, TB = +process.argv[8] || 1.48, OUR = +process.argv[9] || 4.16;
const [A, B, K0, K1] = P;
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const cells = {};
let nbad = 0;
for (const l of fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/)) {
    if (!l.startsWith('S ')) continue;
    const arm = (l.match(/arm=(\w+)/) || [, '?'])[1];
    const mm = l.match(/mean=([0-9.]+)/);
    if (!mm) { nbad++; continue; }
    const tag = l.match(/S (\S+)/)[1];
    const blk = +l.match(/ b(\d+)/)[1];
    const shp = (l.match(/shape=(\S+)/) || [, ''])[1];
    const key = `${tag}|b${blk}|${shp}`;
    const c = cells[key] = cells[key] || { tag, blk, shp, off: [], on: [], rounds: { off: {}, on: {} } };
    const r = +(l.match(/ r(\d+) /) || [, '0'])[1], slot = +(l.match(/slot(\d)/) || [, '0'])[1];
    (c[arm] = c[arm] || []).push(+mm[1]);
    (c.rounds[arm][r] = c.rounds[arm][r] || []).push({ slot, T: +mm[1] });
    c.verdict = (c.verdict || '') + ((/ALL PASS/.test(l)) ? 'P' : (/FAIL=\d+/.test(l) ? 'F' : '?'));
    c.mm = (c.mm || '') + (/mismatch=0\b/.test(l) ? '0' : (/mismatch=[1-9]/.test(l) ? 'X' : '-'));
}
const s = shp => { const q = shp.split(','); return { dir: q[0], S: +q[2], D: +q[3], m: +q[4] }; };
const R = Object.values(cells).filter(c => c.off && c.on && c.off.length && c.on.length).map(c => {
    const g = s(c.shp), tpc = g.S / c.blk, rowTerm = (K0 + K1 * g.m) * tpc;
    const To = mean(c.off), Tn = mean(c.on);
    return { ...c, ...g, tpc, L: Math.min(tpc, Math.floor(16384 / (g.D * 2))), To, Tn, D: g.D, dT: To - Tn, rowTerm,
             f: (Tn - A - B * c.blk) / rowTerm, rounds: [1, 2].map(r => {
                 const o = c.rounds.off[r], n = c.rounds.on[r];
                 return (o && n) ? mean(o.map(x => x.T)) - mean(n.map(x => x.T)) : NaN; }) };
}).sort((x, y) => x.tag.localeCompare(y.tag) || x.blk - y.blk);

const ctrl = R.filter(c => c.tag[0] === 'x'), test = R.filter(c => c.tag[0] === 'c');
console.log(`常数取自 §26.2 的 29 格拟合：A=${A.toFixed(4)}µs B=${(B * 1e3).toFixed(1)}ns/核 k0=${(K0 * 1e3).toFixed(1)}ns/行 k1=${(K1 * 1e3).toFixed(1)}ns/(行·副本)；平台比例 ${RATIO}×（单点推断）`);
console.log(`无 mean 的行：${nbad}   每格应为 8 发（off×4 / on×4）\n`);

console.log('=== ① 构造恒等对照（必须 |Δ| ≤ 0.1µs，否则整轮作废）===');
for (const c of ctrl) {
    console.log(`  ${c.tag} ${c.shp.padEnd(20)} blk=${String(c.blk).padStart(2)}  off ${c.To.toFixed(2)}  on ${c.Tn.toFixed(2)}  Δ=${c.dT >= 0 ? '+' : ''}${(c.dT * 1000).toFixed(0)}ns  数值 ${/[^0-]/.test(c.mm) ? '有非零 mismatch!' : '全 mismatch=0'}`);
}
const ctrlBad = ctrl.filter(c => Math.abs(c.dT) > 0.1);
console.log(ctrl.length ? `  ⇒ ${ctrl.length - ctrlBad.length}/${ctrl.length} 条通过` : '  ⛔ 一条对照都没有 ⇒ 本轮没有功效，读数不得引用');

console.log('\n=== ② m=1 各族：ΔT 与行项残余 f（判据 ΔT ≥ 0.96µs ⇔ f ≤ 20%，§26.8-①）===');
console.log('  格   形状                  blk tpc  L | off    on    ΔT(µs)  逐轮Δ        预测off  残差 | f      平台on  两式分');
for (const c of test) {
    const pred = A + B * c.blk + c.rowTerm;
    const tp = c.Tn * RATIO;
    const lin = 12.5 * TB * (1 / tp - 1 / OUR), pw = 12.5 * (Math.pow(TB / tp, 1.5) - Math.pow(TB / OUR, 1.5));
    const rd = c.rounds.map(x => (isNaN(x) ? ' —  ' : (x >= 0 ? '+' : '') + (x * 1000).toFixed(0))).join('/');
    console.log(`  ${c.tag.padEnd(3)} ${c.shp.padEnd(20)} ${String(c.blk).padStart(3)} ${String(c.tpc).padStart(3)} ${String(c.L).padStart(3)} | ${c.To.toFixed(2)}  ${c.Tn.toFixed(2)}  ${(c.dT >= 0 ? '+' : '') + c.dT.toFixed(2)}   ${rd.padEnd(11)} ${pred.toFixed(2)}  ${(c.To - pred >= 0 ? '+' : '') + (c.To - pred).toFixed(2)} | ${(c.f * 100).toFixed(0).padStart(4)}%  ${tp.toFixed(2)}   ${lin.toFixed(2)}/${pw.toFixed(2)}`);
    if (/[^0-]/.test(c.mm)) console.log(`      ⛔ 该格有 mismatch≠0 的发（${c.mm}）⇒ 数值门禁未过，ΔT 无意义`);
}
const best = test.slice().sort((x, y) => y.dT - x.dT)[0];
if (best) {
    const v = best.dT >= 0.96 ? '✅ 过 §26.8-① 门槛（可写提交树）' :
              best.dT >= 0.6 ? '🟡 落在 0.6~0.96 ⇒ 不提交，改成 L 封顶 + blk 联合重定（§26.8-②）' :
              best.dT >= 0.15 ? '🟡 太小 ⇒ 不提交' : '⛔ ≈0 ⇒ H5 判负（§26.8-③），并给 tbest 口径那条洞一次强证据';
    console.log(`\n  ⇒ 最肥一格 ${best.tag} (${best.shp} blk=${best.blk}) ΔT = ${best.dT.toFixed(2)}µs，f = ${(best.f * 100).toFixed(0)}%  ${v}`);
    const neg = test.filter(c => c.dT < -0.1);
    if (neg.length) console.log(`  ⚠️ 有 ${neg.length} 格 on 臂反而慢 >0.1µs：${neg.map(c => `${c.tag}/b${c.blk} ${(c.dT * 1000).toFixed(0)}ns`).join('  ')}`);
}
console.log('\n  注：`预测off` 用 §26.2 的式子回代同一格，残差列看的是"式子还站得住吗"；');
console.log('      f 只对 on 臂成立（把它当"行项还剩多少"），off 臂的 f 恒 ≈1 是定义而不是发现。');
