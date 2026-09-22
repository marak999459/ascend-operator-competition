// [题1/R18 工装，非提交面] 把 4-run 平衡块按"每轮"摊开，用来判块内一致性。
// 用法: node agg_sym_rounds.js logs/r18_h3sym.log
const fs = require('fs');
const txt = fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/);
// rows[tag][round][slot] = {arm, mean}
const rows = {}, shapes = {};
for (const ln of txt) {
  if (!ln.startsWith('S ')) continue;
  const tag = ln.split(/\s+/)[1];
  const r = +ln.match(/ r(\d+)/)[1];
  const slot = +ln.match(/slot(\d)/)[1];
  const arm = ln.match(/arm=(\w+)/)[1];
  const mean = +ln.match(/mean=([0-9.]+)/)[1];
  shapes[tag] = ln.match(/shape=(\S+)/)[1];
  rows[tag] = rows[tag] || {};
  rows[tag][r] = rows[tag][r] || {};
  rows[tag][r][slot] = { arm, mean };
}
const tags = Object.keys(rows);
const rounds = Math.max(...tags.flatMap(t => Object.keys(rows[t]).map(Number)));
console.log('每轮块内读数（块 = slot1..4，臂序 off,on,on,off）');
console.log('tag   shape                  round  off(s1,s4)      on(s2,s3)      d_off-on');
for (const t of tags) {
  for (let r = 1; r <= rounds; r++) {
    const b = rows[t][r];
    if (!b) { console.log(`${t.padEnd(5)} ${r}  MISSING`); continue; }
    const o = [b[1], b[4]].filter(x => x && x.arm === 'off').map(x => x.mean);
    const n = [b[2], b[3]].filter(x => x && x.arm === 'on').map(x => x.mean);
    const mo = o.reduce((a, x) => a + x, 0) / o.length;
    const mn = n.reduce((a, x) => a + x, 0) / n.length;
    const fmt = a => `${a.map(x => x.toFixed(1)).join(' ')}`.padEnd(13);
    console.log(`${t.padEnd(5)} ${(shapes[t] || '').padEnd(22)} r${r}     ${fmt(o)}(${mo.toFixed(3)})  ${fmt(n)}(${mn.toFixed(3)})  ${(mn - mo >= 0 ? '+' : '') + (mn - mo).toFixed(2)}`);
  }
}
console.log('');
console.log('tag   off均值±散布         on均值±散布          Δ均值    Δ逐轮同号?  每臂n');
for (const t of tags) {
  const O = [], N = [];
  for (let r = 1; r <= rounds; r++) {
    const b = rows[t][r]; if (!b) continue;
    if (b[1] && b[4]) O.push((b[1].mean + b[4].mean) / 2);
    if (b[2] && b[3]) N.push((b[2].mean + b[3].mean) / 2);
  }
  const m = a => a.reduce((x, y) => x + y, 0) / a.length;
  const spread = a => Math.max(...a) - Math.min(...a);
  const ds = O.map((x, i) => N[i] - x);
  const same = ds.every(d => d < 0) ? '全负' : (ds.every(d => d > 0) ? '全正' : '有翻号');
  console.log(`${t.padEnd(5)} ${m(O).toFixed(3)}±${spread(O).toFixed(1)}      ${m(N).toFixed(3)}±${spread(N).toFixed(1)}      ${(m(N) - m(O)).toFixed(3)}  ${same} [${ds.map(d => (d >= 0 ? '+' : '') + d.toFixed(2)).join(' ')}]  ${O.length + N.length}`);
}
// 臂内逐 slot 原始点的散布（打印精度 0.1us ⇒ 看它是否只有一档之差）
console.log('');
console.log('逐 slot 原始点（每臂 3 轮 × 2 槽 = 6 点）');
for (const t of tags) {
  const o = [], n = [];
  for (let r = 1; r <= rounds; r++) {
    const b = rows[t][r]; if (!b) continue;
    for (const s of [1, 4]) if (b[s] && b[s].arm === 'off') o.push(b[s].mean);
    for (const s of [2, 3]) if (b[s] && b[s].arm === 'on') n.push(b[s].mean);
  }
  console.log(`${t.padEnd(5)} off ${o.map(x => x.toFixed(1)).join(' ')}   on ${n.map(x => x.toFixed(1)).join(' ')}`);
}
