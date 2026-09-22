// [题1/R18 工装，非提交面] 用"两种臂序各测若干对"反解真增益 t 与槽位相位 p。
//   模型：off 先跑的一对  (on - off) = t - p ；on 先跑的一对  (on - off) = t + p
//   ⇒ t = (D_offfirst + D_onfirst)/2 ，p = (D_onfirst - D_offfirst)/2   （p>0 = 后一发更快）
// 用法: node parity_fit.js logs/r18_h3probe.log
const fs = require('fs');
const runs = [];
for (const ln of fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/)) {
  if (!ln.startsWith('P ')) continue;
  runs.push({
    shape: ln.match(/shape=(\S+)/)[1],
    arm: ln.match(/arm=(\w+)/)[1],
    mean: +ln.match(/mean=([0-9.]+)/)[1],
  });
}
// 按日志顺序、逐形状把相邻两发配成一对（脚本本身就是成对启动的）
const byShape = {};
for (const r of runs) (byShape[r.shape] = byShape[r.shape] || []).push(r);
const out = [];
for (const shape of Object.keys(byShape)) {
  const seq = byShape[shape], pairs = { offfirst: [], onfirst: [] };
  for (let i = 0; i + 1 < seq.length; i += 2) {
    const [a, b] = [seq[i], seq[i + 1]];
    if (a.arm === b.arm) continue;                      // 破pair跳过，不污染
    const d = b.arm === 'on' ? b.mean - a.mean : a.mean - b.mean;   // 恒取 on - off
    (a.arm === 'off' ? pairs.offfirst : pairs.onfirst).push(d);
  }
  out.push({ shape, pairs });
}
const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN;
console.log('shape                   对数(of/of)  D_offfirst  D_onfirst    t=真Δ     p=相位');
for (const { shape, pairs } of out) {
  const o = pairs.offfirst, n = pairs.onfirst;
  if (!o.length || !n.length) {
    console.log(`${shape.padEnd(23)} ${String(o.length).padStart(4)}/${String(n.length).padStart(4)}      单边序，只能给上界`);
    continue;
  }
  const d1 = avg(o), d2 = avg(n);
  console.log(`${shape.padEnd(23)} ${String(o.length).padStart(4)}/${String(n.length).padStart(4)}   ${d1.toFixed(3).padStart(8)} ${d2.toFixed(3).padStart(10)}   ${((d1 + d2) / 2).toFixed(3).padStart(6)}  ${((d2 - d1) / 2).toFixed(3).padStart(7)}`);
}
