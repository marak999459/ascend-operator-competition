// [题1/R15 工装] 把 run_h1_sweep.sh 的 R15 行聚成"每臂中位数"表，并按
//   T(blk) = A + B*blk + C*io/blk
// 拟合，用来**从数据里**取 H1' 的两个常数（min_io_per_core、core_floor），不靠眼估拐点。
// 用法：node agg_h1.js <log> [io_kb_case_map]
const fs = require('fs');
const log = fs.readFileSync(process.argv[2], 'utf8').split('\n');

// 形状表（来自 npu_debug/test_mhc_expand_npu.cpp 的 prof2 用例表；io=(m+1)*S*D*2 字节，fp16）
const SHAPE = {
  0:  { S: 64,   D: 256,  m: 2, bwd: 0 },
  1:  { S: 64,   D: 256,  m: 2, bwd: 1 },
  8:  { S: 64,   D: 512,  m: 2, bwd: 0 },
  9:  { S: 128,  D: 256,  m: 2, bwd: 0 },
  10: { S: 32,   D: 1024, m: 2, bwd: 0 },
  11: { S: 200,  D: 128,  m: 2, bwd: 0 },
  12: { S: 48,   D: 256,  m: 8, bwd: 0 },
  13: { S: 8,    D: 2048, m: 2, bwd: 0 },
  14: { S: 64,   D: 512,  m: 2, bwd: 1 },
  15: { S: 128,  D: 256,  m: 2, bwd: 1 },
  // p3_band：isqrt 唯一"真的少开核"的外推带（只在 probe2 的补丁表里有这三条）
  16: { S: 64,   D: 512,  m: 16, bwd: 0 },
  17: { S: 256,  D: 1024, m: 4,  bwd: 0 },
  18: { S: 256,  D: 2048, m: 4,  bwd: 0 },
  // p4 tiny-io 阶梯（提交 8 的 case1 反弹追查）：S 固定 64，只切 io
  19: { S: 64,   D: 32,   m: 2,  bwd: 0 },
  20: { S: 64,   D: 64,   m: 2,  bwd: 0 },
  21: { S: 64,   D: 128,  m: 2,  bwd: 0 },
  // p6 反向 m 轴判律（§23.10）：S 固定 64，前两条只切 m，第三条切 D
  22: { S: 64,   D: 256,  m: 4,  bwd: 1 },
  23: { S: 64,   D: 256,  m: 8,  bwd: 1 },
  24: { S: 64,   D: 128,  m: 2,  bwd: 1 },
  // p7 反向小 S（S<num_aiv，block_dim 先被 total_tasks=S 夹住）
  25: { S: 32,   D: 256,  m: 2,  bwd: 1 },
  26: { S: 48,   D: 256,  m: 2,  bwd: 1 },
  27: { S: 32,   D: 512,  m: 2,  bwd: 1 },
  28: { S: 48,   D: 128,  m: 2,  bwd: 1 },
  29: { S: 16,   D: 256,  m: 2,  bwd: 1 },
  30: { S: 8,    D: 256,  m: 2,  bwd: 1 },
  31: { S: 24,   D: 512,  m: 2,  bwd: 1 },
};
const ioKiB = c => { const s = SHAPE[c]; return s ? ((s.m + 1) * s.S * s.D * 2) / 1024 : NaN; };

const med = a => { const s = a.slice().sort((x, y) => x - y); return s.length % 2 ? s[(s.length - 1) >> 1]
  : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };

const arm = new Map();   // key: case|blk|mg|row  -> {means:[], got:Set}
for (const L of log) {
  // row= 后面允许多个 token：AUTO 臂把整个 ROWS 变量原样打出来（如 "row=0 1"），
  // 那不是"两行数据"，只是 blk=NA 那一臂没逐个 row 跑（run_h1_sweep.sh:57）。
  const m = L.match(/^R15 c(\d+) blk=(\S+) mg=(\S+) row=(\S+).* got=(\S+) dir=(\S+) mean=(\S+) p50=(\S+) round=(\d+)/);
  if (!m) continue;
  const [, c, blk, mg, row, got, dir, mean, p50] = m;
  if (!/^[0-9.]+$/.test(mean)) continue;
  const k = `${c}|${blk}|${mg}|${row}`;
  if (!arm.has(k)) arm.set(k, { c: +c, blk, mg, row, dir, got: new Set(), means: [], p50s: [] });
  const a = arm.get(k);
  a.means.push(+mean); a.p50s.push(+p50); a.got.add(got);
}

const rows = [...arm.values()].map(a => ({
  c: a.c, blk: a.blk, mg: a.mg, row: a.row, dir: a.dir,
  got: [...a.got].join(','), n: a.means.length,
  med: +med(a.means).toFixed(2), medp50: +med(a.p50s).toFixed(2),
}));

const blkv = r => r.blk === 'NA' ? 1e9 : +r.blk;
const byCase = new Map();
for (const r of rows) { if (!byCase.has(r.c)) byCase.set(r.c, []); byCase.get(r.c).push(r); }

console.log('### 每臂中位数（剔首 mean / p50，µs；got=设备上报核数）\n');
for (const c of [...byCase.keys()].sort((x, y) => x - y)) {
  const s = SHAPE[c];
  console.log(`-- c${c}  S=${s.S} D=${s.D} m=${s.m} ${s.bwd ? 'bwd' : 'fwd'}  io=${ioKiB(c).toFixed(0)}KiB`);
  const list = byCase.get(c).sort((p, q) => (p.mg + p.row + blkv(p)) .localeCompare(q.mg + q.row + blkv(q))
    || (p.mg !== q.mg ? p.mg.localeCompare(q.mg) : p.row !== q.row ? p.row.localeCompare(q.row) : blkv(p) - blkv(q)));
  let auto = null;
  for (const r of list) {
    if (r.blk === 'NA') { auto = r; continue; }
    const base = list.find(x => x.blk === r.blk && x.mg !== r.mg && x.row === r.row);
    const d = base && base.med ? ((r.med / base.med - 1) * 100).toFixed(1) : '';
    console.log(`   blk=${String(r.blk).padStart(2)} mg=${r.mg} row=${r.row}  got=${r.got.padEnd(6)} n=${r.n}` +
      `  mean=${String(r.med).padStart(5)}  p50=${String(r.medp50).padStart(5)}  vs_opp_merge=${d ? d + '%' : ''}`);
  }
  if (auto) console.log(`   AUTO(今日面) got=${auto.got.padEnd(6)} mean=${auto.med}  p50=${auto.medp50}`);
  const best = list.filter(r => r.blk !== 'NA').sort((p, q) => p.med - q.med)[0];
  if (best && auto) {   // 没有 AUTO 臂的扫描（如 p4b 只跑 mg=off）跳过这一行，别拿 null 当基准
    const gain = ((best.med / auto.med - 1) * 100).toFixed(1);
    console.log(`   >> 最优臂 blk=${best.blk} mg=${best.mg} row=${best.row} mean=${best.med} (vs AUTO ${gain}% / 差 ${(best.med - auto.med).toFixed(2)}us)`);
  }
}

// ---- 合批/核数的双因素分解（只取两侧都有读数的 (c,blk) 组合）----
console.log('\n### 双因素分解：Δ(只开合批) vs Δ(只减核数) vs Δ(两个一起)');
for (const c of [...byCase.keys()].sort((x, y) => x - y)) {
  const list = byCase.get(c).filter(r => r.blk !== 'NA' && r.row === '0');
  const get = (b, g) => list.find(r => r.blk === b && r.mg === g);
  const ref = get('40', 'off') || get('24', 'off');
  if (!ref) continue;
  for (const b of ['2', '4', '6', '8', '12', '16', '24', '40']) {
    const on = get(b, 'on'), off = get(b, 'off');
    if (!on || !off) continue;
    console.log(`   c${c} blk=${b.padStart(2)}  merge: off=${off.med} on=${on.med} Δ=${((on.med/off.med-1)*100).toFixed(1)}%` +
      `   vs blk40+off(${ref.med}) Δ=${((on.med/ref.med-1)*100).toFixed(1)}%`);
  }
}

// ---- 拟合 T = A + B*blk + C*io/blk（对 merge=on 的臂，最小二乘走网格搜索）----
// 单位：T/blk 项用 µs；C 用 **µs / (KiB/核)**，即"单核空载搬运 1KiB 要多久"
//   （70GB/s ⇒ 1KiB/70e9 s = 0.0146µs ⇒ C≈0.015）。C 的搜索域必须覆盖这个量级，
//   否则网格会撞到下边界、把 blk=2 那个点当成带宽项去拟合（RMSE 6µs 那次就是这么错的）。
console.log('\n### 拟合 T = A + B*blk + C*io/blk  (仅 mg=on, row=0)');
const pts = rows.filter(r => r.mg === 'on' && r.row === '0' && r.blk !== 'NA' && SHAPE[r.c])
  .map(r => ({ blk: +r.blk, io: ioKiB(r.c), T: r.med }));
function fit(P) {
  let best = null;
  for (let A = 0.8; A <= 2.2; A += 0.02)
    for (let B = 0.02; B <= 0.20; B += 0.005)
      for (let C = 0.002; C <= 0.10; C += 0.001) {
        let e = 0;
        for (const p of P) { const f = A + B * p.blk + C * p.io / p.blk; e += (f - p.T) ** 2; }
        if (!best || e < best.e) best = { A, B, C, e };
      }
  return { ...best, n: P.length };
}
for (const [label, P] of [['全部 mg=on 点', pts], ['剔除 blk=2', pts.filter(p => p.blk > 2)]]) {
  const { A, B, C, e, n } = fit(P);
  console.log(`   [${label}] n=${n}  A=${A.toFixed(2)}us  B=${(B * 1000).toFixed(0)}ns/核  ` +
    `C=${(C * 1000).toFixed(1)}ns/KiB(每核)  RMSE=${Math.sqrt(e / n).toFixed(2)}us  ` +
    `=> 单核空载 ${(1.024 / C).toFixed(0)}GB/s`);
  for (const io of [64, 98, 128, 196, 294, 400, 512, 1024]) {
    const b = Math.max(1, Math.round(Math.sqrt(io * C / B)));
    console.log(`     io=${String(io).padStart(4)}KiB  blk*=${String(b).padStart(3)}  T*=${(A + B * b + C * io / b).toFixed(2)}us` +
      `  (每核 io=${(io / b).toFixed(1)}KiB -> min_io_per_core=${Math.round(io / b * 1024)}B)`);
  }
}

// ---- 实测最优点给出的"每核 io"，不经过模型 ----
console.log('\n### 直接读数：每 case 最优臂的"每核 io"（这就是 min_io_per_core 的经验值）');
for (const c of [...byCase.keys()].sort((x, y) => x - y)) {
  const list = byCase.get(c).filter(r => r.mg === 'on' && r.row === '0' && r.blk !== 'NA');
  if (!list.length) continue;
  const best = list.sort((p, q) => p.med - q.med)[0];
  const auto = byCase.get(c).find(r => r.blk === 'NA');
  console.log(`   c${String(c).padStart(2)} io=${ioKiB(c).toFixed(0).padStart(4)}KiB  blk*=${best.blk.padStart(2)}` +
    ` T*=${best.med}  每核io=${(ioKiB(c) / +best.blk).toFixed(1)}KiB` +
    (auto ? `  AUTO(got=${auto.got})=${auto.med}  Δ=${((best.med / auto.med - 1) * 100).toFixed(1)}%` : ''));
}
