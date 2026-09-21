// [题1/R17 工装，非提交面] run_h1_sweep.sh 的 R15 行 -> "每臂两轮中位数 + 相对自然核数的位移"表。
//
// 为什么要另写一个而不是用 agg_h1.js：agg_h1.js 按**位置下标**查形状表，而 pcs[] 表是会长的
// （R17 往 tail 追加了 4 条，于是历史日志里的 c19 和今天的 c8 都可能是同一条形状）——
// 这个坑记在 code1.md §23.14。本脚本把形状表当**参数**传进来，标签用 `c<i>`/`s<k>` 两套，
// 并且只输出中位数与位移，不做 T=A+B*blk+C*io/blk 的拟合（拟合要 io，而 io 恰恰是下标最容易记错的东西）。
//
// 用法：node agg_r17.js <log> ["<label>=<S>x<D>x<m>[/b],..."]
//   第 2 个参数省略时只按标签分组，不打印 io。`/b` 后缀表示反向。
const fs = require('fs');
const lines = fs.readFileSync(process.argv[2], 'utf8').split('\n');
const names = {};
if (process.argv[3]) for (const t of process.argv[3].split(/[ ,]+/)) {
  const m = /^(c\d+|s\d+)=([0-9]+)x([0-9]+)x([0-9]+)(\/b)?$/.exec(t);
  if (!m) { console.log('### BAD TOKEN ' + t); process.exit(1); }
  const S = +m[2], D = +m[3], mm = +m[4], bwd = !!m[5];
  // io = in + out = S*D*2 + S*m*D*2，两个方向同式（run_case 里只是把 in/out 对调）
  names[m[1]] = { S, D, m: mm, bwd, io: S * D * (mm + 1) * 2 };
}
// key = label|blk|merge|row -> [mean...]
const cell = {}, order = [];
for (const L of lines) {
  const m = /^R15 (c\d+|s\d+) blk=(\S+) mg=(\S+) row=(\S+) got=(\S+) dir=(\S+) mean=(\S+) p50=(\S+) round=(\S+)$/.exec(L.trim());
  if (!m) continue;
  const k = [m[1], m[2], m[3], m[4]].join('|');
  if (!cell[k]) { cell[k] = { got: m[5], dir: m[6], v: [] }; order.push(k); }
  cell[k].v.push(parseFloat(m[7]));
}
const med = a => { const s = a.slice().sort((x, y) => x - y); const n = s.length;
  return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN; };
const nat = {};   // 自然核数臂（blk=NA）当分母
for (const k of order) { const p = k.split('|'); if (p[1] === 'NA') nat[p[0]] = med(cell[k].v); }
const rows = [];
for (const k of order) {
  const [lbl, blk, mg, row] = k.split('|');
  const t = med(cell[k].v);
  const base = blk === 'NA' ? t : nat[lbl];
  rows.push({ lbl, blk, mg, row, got: cell[k].got, dir: cell[k].dir,
    n: cell[k].v.length, spread: (Math.max(...cell[k].v) - Math.min(...cell[k].v)),
    t, d: base ? 100 * (t / base - 1) : NaN,
    io: names[lbl] ? names[lbl].io : 0 });
}
rows.sort((a, b) => (a.lbl === b.lbl ? (+a.blk === +b.blk ? 0 : (a.blk === 'NA' ? -1 : b.blk === 'NA' ? 1 : +a.blk - +b.blk)) : a.lbl.localeCompare(b.lbl, undefined, { numeric: true })));
for (const r of rows) {
  const sh = names[r.lbl];
  console.log([
    r.lbl.padEnd(4),
    sh ? (`S=${String(sh.S).padStart(5)} D=${String(sh.D).padStart(5)} m=${sh.m}${sh.bwd ? ' bwd' : ' fwd'}`).padEnd(28) : '(shape unknown)'.padEnd(28),
    sh ? ((r.io / 1024).toFixed(0) + 'KiB').padStart(8) : ''.padStart(8),
    ('blk=' + r.blk).padEnd(7), ('got=' + r.got).padEnd(8), ('n=' + r.n).padEnd(4),
    ('mg=' + r.mg).padEnd(6),
    (r.t.toFixed(2) + 'us').padStart(7),
    (r.d === 0 ? '' : (r.d > 0 ? '+' : '') + r.d.toFixed(1) + '%').padStart(8),
    ('spread=' + r.spread.toFixed(2)).padStart(11),
  ].join(' '));
}
console.log('### d% 的分母 = 同一条形状 blk=NA（自然核数）那一臂；NA 行自身的 d 恒为 0');
