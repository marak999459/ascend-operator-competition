// [题1/p20 第 1.5 阶离线判读工装，非提交面] 把 stage-1(12 格) + stage-1.5(18 格) 合起来做
// 四参数最小二乘：  T = A + B·blk + k0·(S/blk) + k1·(m·S/blk)
// 若 k1/k0 落在 §22 单价（52~58ns/次）的"每副本 2 次"上、且残差 RMS 不破 0.05µs，
// 就等于把 §25.3 那个"k=270 只在 m=1 标定"的口子封掉：反向 µs 档只有两个自由度
// （每行常数项 + 每副本一次），字节项可以整个扔掉。
// 用法: node fit_p20b.js <log1> [<log2> ...]
const fs = require('fs');
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const cells = {};
for (const f of process.argv.slice(2)) {
    for (const l of fs.readFileSync(f, 'utf8').split(/\r?\n/)) {
        if (!l.startsWith('S ') || /arm=IDENT|arm=SELFTEST/.test(l)) continue;
        const blk = +l.match(/ b(\d+)/)[1];
        const shp = l.match(/shape=(\S+)/)[1].split(',');
        const S = +shp[2], D = +shp[3], m = +shp[4];
        const key = `${shp.slice(0, 5).join(',')}|b${blk}`;
        (cells[key] = cells[key] || { S, D, m, blk, T: [] }).T.push(+l.match(/mean=([0-9.]+)/)[1]);
    }
}
const C = Object.entries(cells).map(([k, c]) => ({ k, S: c.S, D: c.D, m: c.m, blk: c.blk, T: mean(c.T), n: c.T.length }));
C.sort((a, b) => a.m - b.m || a.S - b.S || a.blk - b.blk);
function solve(M, v) {                        // 列主元高斯-约当
    const n = v.length;
    for (let i = 0; i < n; i++) {
        let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
        [M[i], M[p]] = [M[p], M[i]]; [v[i], v[p]] = [v[p], v[i]];
        const d = M[i][i]; for (let c = i; c < n; c++) M[i][c] /= d; v[i] /= d;
        for (let r = 0; r < n; r++) {
            if (r === i) continue;
            const f = M[r][i]; if (!f) continue;
            for (let c = i; c < n; c++) M[r][c] -= f * M[i][c];
            v[r] -= f * v[i];
        }
    }
    return v;
}
function fit(basis, label) {
    const n = basis.length, M = Array.from({ length: n }, () => new Array(n).fill(0)), v = new Array(n).fill(0);
    for (const c of C) {
        const x = basis.map(f => f(c));
        for (let i = 0; i < n; i++) { v[i] += x[i] * c.T; for (let j = 0; j < n; j++) M[i][j] += x[i] * x[j]; }
    }
    const p = solve(M, v);
    const res = C.map(c => c.T - basis.reduce((s, f) => s + p[basis.indexOf(f)] * f(c), 0));
    const rms = Math.sqrt(mean(res.map(r => r * r)));
    console.log(`${label.padEnd(34)} ${p.map(x => x.toFixed(4)).join('  ')}   RMS=${rms.toFixed(4)}µs  max|残差|=${Math.max(...res.map(Math.abs)).toFixed(3)}`);
    return { p, rms };
}
console.log(`格子数 = ${C.length}（stage-1 12 + stage-1.5 18，IDENT/SELFTEST 已剔；每格取其全部重复点的均值）`);
console.log('系数口径：µs。B=每核，k0=每行常数项，k1=每 (行·副本)\n');
fit([c => 1, c => c.blk, c => c.S / c.blk], 'A + B·blk + k·S/blk            （stage-1 形式）');
const f4 = fit([c => 1, c => c.blk, c => c.S / c.blk, c => c.m * c.S / c.blk], 'A + B·blk + k0·S/blk + k1·m·S/blk');
const { p } = f4;
console.log(`\n  ⇒ 每行 = ${(p[2] * 1000).toFixed(0)}ns + ${(p[3] * 1000).toFixed(0)}ns × m；m=1 给 ${(1e3 * (p[2] + p[3])).toFixed(0)}ns（stage-1 单点标定的 270）`);
console.log(`  ⇒ 每副本 ${p[3] * 1000}ns ÷ 2 次往返 = ${(p[3] * 500).toFixed(1)}ns/次  vs §22 独立标定的 52~58ns/次`);
const io = [c => 1, c => c.blk, c => c.S / c.blk, c => c.m * c.S / c.blk, c => (c.m + 1) * c.S * c.D * 2 / 1024];
const f5 = fit(io, '再加字节项 C·io(KiB)');
console.log(`  ⇒ 字节项 = ${(f5.p[4] * 1000).toFixed(2)}ns/KiB，RMS 由 ${f4.rms.toFixed(4)} 变到 ${f5.rms.toFixed(4)} ⇒ 字节项${f5.rms > f4.rms * 0.97 ? '基本不带信息' : '仍带信息'}`);
console.log('\n=== 四参数模型的逐格残差（|残差|>0.1µs 的格子点名）===');
for (const c of C) {
    const t = p[0] + p[1] * c.blk + p[2] * c.S / c.blk + p[3] * c.m * c.S / c.blk;
    const r = c.T - t;
    if (Math.abs(r) > 0.1) console.log(`  ${c.k.padEnd(28)} n=${String(c.n).padStart(3)} 实测 ${c.T.toFixed(2)} 模型 ${t.toFixed(2)} 残差 ${r >= 0 ? '+' : ''}${r.toFixed(2)}`);
}
console.log('\n=== 最优点：dT/dblk=0 ⇒ blk* = sqrt((k0+k1·m)·S/B)，T_min 与"合批把每核行数除以 L"的对照 ===');
for (const m of [1, 2, 4, 8]) for (const S of [32, 64, 128]) {
    const kk = p[2] + p[3] * m, bstar = Math.sqrt(kk * S / p[1]);
    const Tm = p[0] + 2 * Math.sqrt(p[1] * kk * S);
    const T16 = p[0] + p[1] * 16 + kk * S / 16;
    console.log(`  m=${String(m).padStart(2)} S=${String(S).padStart(3)}: 每行 ${(kk * 1000).toFixed(0)}ns  blk*=${bstar.toFixed(1)}  T_min=${Tm.toFixed(2)}µs  @blk16=${T16.toFixed(2)}  ⇒ 行维合批到 L=4 后 T=${(p[0] + p[1] * 16 + kk * S / 64).toFixed(2)}µs（若 k 随 L 摊薄）`);
}
