// [题1/p20 第 1.5 阶离线判读工装，非提交面] 把 stage-1 标定好的 A/B 当已知，反解每格的
//   k_cell = (T − A − B·blk) · blk / rows      （µs / (核·行)，rows = 反向整行模式下的 task 数 = S）
// 三张表：k(m)（每行几次副本）、k(S)（"每核行数"是不是真自变量）、io 分离（同 m 同 rows 把 io 减半）。
// 用法: node fit_k.js <p20b.log> [A=1.19583] [B=0.071212] [p20_stage1.log]
//   （stage-1 那份可选，传进来是为了让 m=1 那一档不用手算就有基准）
const fs = require('fs');
const A = +(process.argv[3] || 1.19583), B = +(process.argv[4] || 0.071212);
const files = [process.argv[2], process.argv[5]].filter(Boolean);
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const rows = {};
for (const f of files) for (const l of fs.readFileSync(f, 'utf8').split(/\r?\n/)) {
    if (!l.startsWith('S ')) continue;
    const armRaw = l.match(/arm=(\w+)/)[1];
    if (armRaw === 'IDENT' || armRaw === 'SELFTEST') continue;   // IDENT 是相位对照、SELFTEST 是门禁，都不进表
    // REF 的 shape 就是 m=1 基准格（bwd,fp16,64,384,1 @b16），下面归一为 m1 一起进表
    const blk = +l.match(/ b(\d+)/)[1];
    const shp = l.match(/shape=(\S+)/)[1].split(',');
    const S = +shp[2], D = +shp[3], m = +shp[4];
    const ioK = (m + 1) * S * D * 2 / 1024;
    const arm = (m === 1 && S === 64 && D === 384) ? 'm1' : armRaw;   // 归一：m=1 基准格与 stage-1 同一标签
    const key = `${arm}|b${blk}|D${D}|S${S}`;
    (rows[key] = rows[key] || { arm, blk, S, D, m, ioK, T: [] }).T.push(+l.match(/mean=([0-9.]+)/)[1]);
}
const R = Object.values(rows).map(c => ({ ...c, n: c.T.length, T: mean(c.T), k: (mean(c.T) - A - B * c.blk) * c.blk / c.S }))
    .sort((x, y) => x.arm.localeCompare(y.arm) || x.blk - y.blk);
console.log(`已知常数：A=${A.toFixed(4)}µs  B=${(B * 1000).toFixed(1)}ns/核（stage-1 p20 的 12 格拟合）`);
console.log('REF 已并入 m=1 基准格（它的 shape 就是那一格）；IDENT 只作相位对照，不进表\n');
console.log('格            rows S   D    m   io(KiB)  blk | 实测T   k(ns/核·行)  每核行数  n');
for (const c of R) {
    console.log(`${c.arm.padEnd(8)} b${String(c.blk).padEnd(4)}  ${String(c.S).padStart(4)} ${String(c.D).padStart(5)} ${String(c.m).padStart(3)}   ${String(c.ioK).padStart(6)}        |  ${c.T.toFixed(2)}   ${(c.k * 1000).toFixed(0).padStart(6)}      ${(c.S / c.blk).toFixed(1)}     ${c.n}`);
}
const grp = f => { const g = R.filter(f); return g.length ? { n: g.length, k: mean(g.map(x => x.k)), T: mean(g.map(x => x.T)), lo: Math.min(...g.map(x => x.k)), hi: Math.max(...g.map(x => x.k)), ioK: mean(g.map(x => x.ioK)), m: g[0].m, S: g[0].S, D: g[0].D } : null; };
console.log('\n=== k(m) —— 每行副本数买多少钱（S=64 固定，blk 同档才可比）===');
for (const b of [8, 16, 32]) {
    const line = [1, 2, 4, 8].map(m => {
        const g = grp(x => x.blk === b && x.m === m && x.S === 64 && x.D === 384 && !/^S/.test(x.arm));
        return g ? `m=${String(m).padStart(2)}:${(g.k * 1000).toFixed(0)}ns` : `m=${m}:  —  `;
    });
    console.log(`  blk=${String(b).padStart(2)}  ${line.join('   ')}`);
}
{
    const pts = [1, 2, 4, 8].map(mm => { const g = grp(x => x.m === mm && x.S === 64 && x.blk === 16 && !/^S/.test(x.arm)); return g ? { m: mm, k: g.k } : null; }).filter(Boolean);
    if (pts.length >= 2) {
        const a = pts[0], b = pts[pts.length - 1];
        const slope = (b.k - a.k) / (b.m - a.m);
        console.log(`  ⇒ 每多 1 个副本每行多花 ${(slope * 1000).toFixed(0)}ns（端点法，取 ${a.m}→${b.m}，实测各 m：${pts.map(p => `m${p.m}=${(p.k * 1000).toFixed(0)}`).join(' / ')}）；若 §22 单价 54ns/次 成立且每副本 2 次往返 ⇒ 期望 ≈108ns`);
    }
}
console.log('\n=== k(S) —— "每核行数"是不是真自变量（m=1, D=384）===');
for (const b of [8, 16, 32]) {
    const a32 = grp(x => x.S === 32 && x.blk === b && x.m === 1), a64 = grp(x => x.blk === b && x.m === 1 && x.S === 64 && !/^S/.test(x.arm)), a128 = grp(x => x.S === 128 && x.blk === b && x.m === 1);
    const s = (n, g) => `${n}=${g ? (g.k * 1000).toFixed(0) + 'ns/T' + g.T.toFixed(2) : '  —  '}`;
    console.log(`  blk=${String(b).padStart(2)}  ${s('S32', a32)}   ${s('S64', a64)}   ${s('S128', a128)}`);
}
console.log('\n=== io 分离 —— 同 m 同 rows 把 io 砍一半（m=8, D=384 vs D=192），看的是 ΔT/Δio 而不是 Δk ===');
for (const b of [16, 32]) {
    const hi = grp(x => x.arm === 'm8' && x.blk === b), lo = grp(x => x.arm === 'm8D192' && x.blk === b);
    if (!hi || !lo) { console.log(`  blk=${b}: 缺一格`); continue; }
    const dK = hi.ioK - lo.ioK;
    console.log(`  blk=${String(b).padStart(2)}  io${hi.ioK.toFixed(0)}k T=${hi.T.toFixed(2)}  vs io${lo.ioK.toFixed(0)}k T=${lo.T.toFixed(2)}  ⇒ ΔT=${((hi.T - lo.T) * 1000).toFixed(0)}ns / Δio=${dK.toFixed(0)}KiB ⇒ ${(hi.T - lo.T) * 1000 / dK}ns/KiB（stage-1 拟合给的是 1.0；注意这两格 rows 同为 64，ΔT 里不含 k）`);
}
