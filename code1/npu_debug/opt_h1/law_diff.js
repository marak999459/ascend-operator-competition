// [题1/R17 工装，非提交面] 反向核数律的**三面 diff**：提交10 / 提交11 / v5 候选（snap 到零空核）。
// 为什么要有这个：p16 量到提交11 的上界在 `total_tasks < num_aiv` 的反向形状上是**压坏**的
// （S=32,D=2048,m=8：32→24 实测 +16.0%），机制是 ceil 分桶下 blk=24 只有 16 个核有活干、
// 8 个核白付派发（见 code1.md §23.21）。这类"核数不是可行切分"的错误**纸面就能算出来**，
// 不该等到真机。本脚本只镜像 op_host/mhc_expand.cpp 的算术，不碰设备。
//
// 用法：node law_diff.js            -> 跑内置 45 条 roster + 一张覆盖格
const UB = 196608, NUM_AIV = 40, ES = 2;            // 910B3 UB=192KiB、40 AIV、fp16/bf16=2B

function tiling(S, D, m, bwd) {
  const ub_budget = UB / 4;
  const whole = D * ES <= ub_budget;
  const dTileLen = whole ? D : (() => {
    let t = Math.min(2048, D);
    t = Math.min(t, Math.floor(ub_budget / ES));
    if (t > 16) t = Math.floor(t / 16) * 16;
    if (t < 16) t = 16;
    if (t > D) t = D;
    return Math.max(t, 1);
  })();
  // host 用的是**整数除法** (D + len - 1)/len ⇒ 这里是 floor，不是 ceil（写错会把 dtn 全表 +1）
  const dTileNum = Math.floor((D + dTileLen - 1) / dTileLen);
  const io = (m + 1) * S * D * ES;
  const tile = dTileLen * ES;
  let merge_cap = 8;
  while ((merge_cap + 1) * (merge_cap + 1) <= Math.floor(io / 4096)) ++merge_cap;
  if (Math.floor(io / 131072) > merge_cap) merge_cap = Math.floor(io / 131072);
  const merge_ok = !bwd && dTileNum === 1 && tile <= 6144 && tile % 32 === 0 && merge_cap * 2 <= S;
  let total, split;
  if (S >= NUM_AIV || merge_ok) { split = 'ROW'; total = S; }
  else if (!bwd) { const st = S * m; if (st >= NUM_AIV) { split = 'STREAM'; total = st; }
                   else { split = 'ELEM'; total = S * m * dTileNum; } }
  else { split = 'ELEM'; total = S * dTileNum; }
  return { whole, dTileLen, dTileNum, io, merge_ok, merge_cap, total, split };
}

// 三个面各自的 core_cap。face 10 = R16 律；face 11 = +12 次发起上界（整行）；face 12 = 上界再 snap。
function blk(S, D, m, bwd, face) {
  const t = tiling(S, D, m, bwd);
  let block_dim = NUM_AIV;
  if (t.total < NUM_AIV) block_dim = t.total;
  let cap;
  if (t.merge_ok) cap = t.merge_cap;
  else if (bwd) {
    const guard = Math.max(8, Math.min(16, Math.floor(S / 2)));
    cap = Math.max(Math.floor(t.io / 12288), guard);
    if (face >= 11 && t.dTileNum === 1) {
      const ceil12 = Math.max(16, Math.floor((t.total * (m + 1) + 11) / 12));
      if (cap > ceil12) {                     // 上界**确实绑住**时才动（= 提交11 的改动域内）
        let bound = ceil12;
        if (face >= 12) {                     // v5：snap 到"零空核"的可行切分
          const rpc = Math.max(1, Math.ceil(t.total / ceil12));
          bound = Math.ceil(t.total / rpc);
        }
        if (cap > bound) cap = bound;
      }
    }
  } else cap = Math.floor(t.io / 8192);
  if (cap < 8) cap = 8;
  if (cap < block_dim) block_dim = cap;
  const tpc = Math.ceil(t.total / block_dim);
  const working = Math.ceil(t.total / tpc);
  return { blk: block_dim, total: t.total, split: t.split, dtn: t.dTileNum, io: t.io,
           tpc, idle: block_dim - working, issues: tpc * (m + 1) };
}

const ROSTER = [];
const q = (nm, bwd, S, D, m) => ROSTER.push({ nm, bwd, S, D, m });
q('fwd-fp16-small',0,64,256,2); q('fwd-fp16-D100',0,7,100,2); q('fwd-fp16-D7167',0,3,7167,2);
q('fwd-fp16-S1D1',0,1,1,2); q('fwd-fp16-m1',0,5,33,1); q('fwd-fp16-m16',0,64,512,16);
q('bwd-fp16-small',1,64,256,2); q('bwd-fp16-D100',1,4,100,2); q('bwd-fp16-D7167',1,2,7167,2);
q('bwd-fp16-S1D1',1,1,1,2); q('bwd-fp16-m1',1,5,33,1); q('bwd-fp16-m16',1,64,512,16);
q('bwd-fp16-sat-m1',1,4,128,1); q('bwd-fp16-sat-m2',1,4,128,2);
q('fwd-bf16-small',0,64,256,2); q('fwd-bf16-D7167',0,3,7167,2);
q('bwd-bf16-small',1,64,256,2); q('bwd-bf16-D7167',1,2,7167,2);
q('fwd-fp16-ROW-m8',0,4,2048,8); q('bwd-fp16-m2',1,8,256,2); q('bwd-fp16-m3',1,8,256,3);
q('bwd-fp16-m4',1,8,256,4); q('bwd-fp16-m5',1,8,256,5); q('bwd-fp16-D7167-m5',1,2,7167,5);
q('bwd-bf16-m4',1,8,256,4);
q('fwd-fp16-medium',0,1024,4096,4); q('bwd-fp16-medium',1,1024,4096,4);
q('fwd-bf16-medium',0,1024,4096,4); q('bwd-bf16-medium',1,1024,4096,4);
q('fwd-fp16-large',0,8192,7168,8); q('fwd-bf16-large',0,8192,7168,8);
q('bwd-fp16-large',1,8192,7168,8); q('bwd-bf16-large',1,8192,7168,8);
[['bwd-fp16-MT-ELEM',1,4,4096,16],['bwd-fp16-MT-ODD',1,2,7167,8],['bwd-bf16-MT-ELEM',1,4,4096,16],
 ['bwd-bf16-MT-ODD',1,2,7167,8],['bwd-fp16-MTL-D70000-m8',1,2,70000,8],['bwd-fp16-MTL-D70000-m2',1,2,70000,2],
 ['bwd-fp16-MTL-D33000-m8',1,1,33000,8],['bwd-bf16-MTL-D70000-m2',1,2,70000,2],
 ['fwd-fp16-MTL-D70000-m2',0,1,70000,2],['bwd-fp16-MTL-D70001-m2',1,2,70001,2],
 ['bwd-bf16-MTL-D33001-m8',1,1,33001,8],['fwd-fp16-MTL-D70001-m2',0,1,70001,2]]
  .forEach(a => ROSTER.push({nm:a[0],bwd:a[1],S:a[2],D:a[3],m:a[4]}));

const kb = n => (n >= 1048576 ? (n/1048576).toFixed(2)+'MiB' : (n/1024).toFixed(0)+'KiB');
const dedup = new Set();
console.log('### roster 45 条（同名同形状只算一次）：三面的 block_dim / 空核数');
console.log('name                     bwd  S     D      m   io        dtn split | blk10 idle | blk11 idle | blk12 idle');
for (const c of ROSTER) {
  const key = [c.bwd,c.S,c.D,c.m].join('|');
  if (dedup.has(key)) { console.log('  (dup) ' + c.nm); continue; }
  dedup.add(key);
  const a = blk(c.S,c.D,c.m,c.bwd,10), b = blk(c.S,c.D,c.m,c.bwd,11), d = blk(c.S,c.D,c.m,c.bwd,12);
  const mark = (b.blk !== a.blk ? ' <<11' : '') + (d.blk !== b.blk ? ' <<12' : '');
  console.log(c.nm.padEnd(26) + (c.bwd?'bwd ':'fwd ') + String(c.S).padStart(4) + String(c.D).padStart(7) +
    String(c.m).padStart(4) + '  ' + kb(a.io).padStart(9) + ' ' + String(a.dtn).padStart(3) + ' ' +
    a.split.padEnd(6) + '| ' + String(a.blk).padStart(3) + String(a.idle).padStart(5) + ' | ' +
    String(b.blk).padStart(3) + String(b.idle).padStart(5) + ' | ' + String(d.blk).padStart(3) +
    String(d.idle).padStart(5) + mark);
}

console.log('\n### 覆盖格（反向、整行 D<=24576、S(m+1)<480 ∧ io>196KiB = 提交11 的改动域 + 周边）');
console.log('S    D     m   io        | blk10 idle10 tpc | blk11 idle11 | blk12 idle12 | 11vs10  12vs11');
for (const S of [24,32,40,48,64,96,128]) for (const D of [512,1024,2048,4096]) for (const m of [2,4,8,16]) {
  const t = tiling(S,D,m,1);
  if (!t.whole) continue;
  const a = blk(S,D,m,1,10), b = blk(S,D,m,1,11), d = blk(S,D,m,1,12);
  if (a.blk === b.blk && b.blk === d.blk) continue;
  console.log(String(S).padStart(3) + String(D).padStart(6) + String(m).padStart(4) + '  ' + kb(a.io).padStart(9) +
    ' | ' + String(a.blk).padStart(3) + String(a.idle).padStart(5) + String(a.tpc).padStart(5) +
    ' | ' + String(b.blk).padStart(3) + String(b.idle).padStart(5) +
    ' | ' + String(d.blk).padStart(3) + String(d.idle).padStart(5) +
    ' |  ' + ((b.blk/a.blk-1)*100).toFixed(0).padStart(4) + '%  ' + ((d.blk/b.blk-1)*100).toFixed(0).padStart(4) + '%');
}
