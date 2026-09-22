// [题1/R22 p22 离线判读工装，非提交面] H5c（反向 m>=2 gather + 宽 L·D VEC）同构建 A/B 的读表器。
// 输入 = run_p22_h5c.sh 的日志（每格 8 发：块1 off,on,on,off | 块2 on,off,off,on）。
// 按 §26.8 + §27.9-1 的判据排三件事：
//   ① 两条"由构造不可能走新分支"的对照臂（m=1 / 前向）必须先 ≈0，否则整轮读数作废（§23.27-②）；
//   ② 逐格 ΔT = mean(off) − mean(on)，按轮拆开看符号是否逐轮一致；
//   ③ 过线 = 本地 ΔT ≥ 0.96µs **且三种榜面换算的最低值 ≥ 1.6 分**（§27.7 把"两端点都过线"写成了硬要求）：
//        (a) 位移绝对搬：榜面省 = ΔT × 0.615（最不利端点）与 × 1.234（最有利端点）
//        (b) 比例传递：榜面新值 = 榜面现值 × (T_on / T_off)   ← 不看绝对水位，只看这一格自己降了几成
//      三种读法各自再走两式（线性 12.5·tbest·(1/Tn−1/To) 与幂 12.5·[(tbest/Tn)^1.5−(tbest/To)^1.5]）。
// 与 agg_p21 的差别：L 的封顶口径从 H5a 的 FWD_MERGE_BYTES(16384)/tile 换成 H5c 的 12288/tile
//   （= 每批 8 份 tile 字节 ≤ 98304），并新增"每核批数 / on 臂残差 ÷ 批数"两列 —— 那是 §27.3-②
//   那条"每批 0.3µs 固定成本"在 m>=2 上是否累加的直接判据（w7 一格专门为它铺）。
// 用法: node agg_p22.js <p22.log> [A=1.3897] [B_ns=65.0] [k0_ns=145.1] [k1_ns=107.0]
//                       [tbest=1.48] [our=4.16]
const fs = require('fs');
const A = +process.argv[3] || 1.3897, B = (+process.argv[4] || 65.0) / 1000;
const K0 = (+process.argv[5] || 145.1) / 1000, K1 = (+process.argv[6] || 107.0) / 1000;
const TB = +process.argv[7] || 1.48, OUR = +process.argv[8] || 4.16;
const RAT = [0.615, 1.234];                      // §27.7 的两个榜面/本地比例端点
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const score = (tn, to) => [12.5 * TB * (1 / tn - 1 / to),
                           12.5 * (Math.pow(TB / tn, 1.5) - Math.pow(TB / to, 1.5))];
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
const s = shp => { const q = shp.split(','); return { S: +q[2], D: +q[3], m: +q[4] }; };
// 提交态的自然核数：反向那支 host 律（op_host/mhc_expand.cpp:205-211）
//   core_cap = max(io/12288, max(8, min(16, S/2)))，再夹 8<=core_cap<=40
const natBlk = g => Math.min(40, Math.max(8, Math.max((g.m + 1) * g.S * g.D * 2 / 12288,
                                                     Math.max(8, Math.min(16, g.S / 2)))));
const R = Object.values(cells).filter(c => c.off && c.on && c.off.length && c.on.length).map(c => {
    const g = s(c.shp), tpc = g.S / c.blk;
    const rowTerm = (K0 + K1 * g.m) * tpc;
    const tile = g.D * 2;                                   // fp16
    const L = Math.min(tpc, Math.floor(12288 / tile));      // H5c 的 cap（8*L*tile<=98304）
    const nbatch = Math.ceil(tpc / L);
    const To = mean(c.off), Tn = mean(c.on);
    const resn = Tn - A - B * c.blk;                        // on 臂去掉地板+核数后剩多少
    return { ...c, ...g, tpc, L, nbatch, resn, perBatch: resn / nbatch, To, Tn, dT: To - Tn, rowTerm,
             nat: natBlk(g), rho: Tn / To, pred: A + B * c.blk + rowTerm, f: resn / rowTerm,
             rounds: [1, 2].map(r => {
                 const o = c.rounds.off[r], n = c.rounds.on[r];
                 return (o && n) ? mean(o.map(x => x.T)) - mean(n.map(x => x.T)) : NaN; }) };
}).sort((x, y) => x.tag.localeCompare(y.tag) || x.blk - y.blk);

// === 提交态报价：基准必须是"同形状自然 blk 的 off 臂"，不是同 blk 的 off 臂 ===
//   分支净值 dT   = off@X − on@X        （只回答"这一刀有没有机制"）
//   提交净值 sub  = off@自然 − on@X     （回答"交上去平台会动几 µs"，含重定核数那半）
//   三种位移刻度：0.615（§27.7 悲观端点）、ρ_anchor=OUR/off@自然（锚点比例传递）、1.234（乐观端点）
//   ⚠️ 位移后的平台时间不得越过该 case 的 tbest（榜上 1.52µs 是真读数）——越过就夹住并标注
const offNat = {};
for (const c of R) if (Math.abs(c.blk - c.nat) < 1e-9) offNat[c.shp] = c.To;
for (const c of R) {
    c.base = offNat[c.shp];
    c.sub = (c.base === undefined) ? NaN : c.base - c.Tn;
    c.rAnchor = (c.base === undefined) ? NaN : OUR / c.base;
    if (!isNaN(c.sub)) {
        c.tn = [0.615, c.rAnchor, 1.234].map(r => OUR - c.sub * r);
        c.clip = c.tn.map(t => t < TB);
        c.sc = c.tn.map((t, i) => score(Math.max(t, TB), OUR));
    }
}

const ctrl = R.filter(c => c.tag[0] === 'x'), test = R.filter(c => c.tag[0] === 'w');
console.log(`常数取自 §26.2 的 29 格拟合：A=${A.toFixed(4)}µs B=${(B * 1e3).toFixed(1)}ns/核 k0=${(K0 * 1e3).toFixed(1)}ns/行 k1=${(K1 * 1e3).toFixed(1)}ns/(行·副本)`);
console.log(`榜面锚点(举例)：tbest=${TB} 现值=${OUR}；换算读法 = 位移×{${RAT.join(', ')}} 两档 + 比例传递；无 mean 的行：${nbad}\n`);

console.log('=== ① 构造恒等对照（必须 |Δ| ≤ 0.1µs，否则整轮作废）===');
for (const c of ctrl)
    console.log(`  ${c.tag} ${c.shp.padEnd(20)} blk=${String(c.blk).padStart(2)}  off ${c.To.toFixed(2)}  on ${c.Tn.toFixed(2)}  Δ=${c.dT >= 0 ? '+' : ''}${(c.dT * 1000).toFixed(0)}ns  数值 ${/[^0-]/.test(c.mm) ? '有非零 mismatch!' : '全 mismatch=0'}`);
const ctrlBad = ctrl.filter(c => Math.abs(c.dT) > 0.1);
console.log(ctrl.length ? `  ⇒ ${ctrl.length - ctrlBad.length}/${ctrl.length} 条通过` : '  ⛔ 一条对照都没有 ⇒ 本轮没有功效，读数不得引用');

console.log('\n=== ② m>=2 各族：ΔT=分支净值 / 提交净值=自然 off − 本 on（判据 ΔT≥0.96 且最低换算≥1.6 分）===');
console.log('  格  形状                 blk/自然 tpc  L 批 | off    on    ΔT      逐轮Δ      预测off 残差 |  自然off 提交净 |  f    残差ns 每批ns | 分(保守,锚点,乐观)');
for (const c of test) {
    const rd = c.rounds.map(x => (isNaN(x) ? ' —  ' : (x >= 0 ? '+' : '') + (x * 1000).toFixed(0))).join('/');
    const sc = isNaN(c.sub) ? '—' : c.sc.map((x, i) => (c.clip[i] ? '>' : '') + Math.min(x[0], x[1]).toFixed(1)).join(',');
    console.log(`  ${c.tag.padEnd(3)} ${c.shp.padEnd(20)} ${(String(c.blk) + '/' + c.nat.toFixed(0)).padStart(6)} ${String(c.tpc).padStart(3)} ${String(c.L).padStart(3)} ${String(c.nbatch).padStart(2)} | ${c.To.toFixed(2)}  ${c.Tn.toFixed(2)}  ${(c.dT >= 0 ? '+' : '') + c.dT.toFixed(2)}   ${rd.padEnd(11)} ${c.pred.toFixed(2)} ${(c.To - c.pred >= 0 ? '+' : '') + (c.To - c.pred).toFixed(2)} |  ${c.base ? c.base.toFixed(2) : '  — '}  ${(isNaN(c.sub) ? NaN : c.sub).toFixed(2)} | ${(c.f * 100).toFixed(0).padStart(4)}%  ${(c.resn * 1000).toFixed(0).padStart(5)} ${(c.perBatch * 1000).toFixed(0).padStart(6)} | ${sc}`);
    if (/[^0-]/.test(c.mm)) console.log(`      ⛔ 该格有 mismatch≠0 的发（${c.mm}）⇒ 数值门禁未过，ΔT 无意义`);
}
const priced = test.filter(c => !isNaN(c.sub));
const ok = priced.filter(c => c.dT >= 0.96 && Math.min(...c.sc.map(x => Math.min(x[0], x[1]))) >= 1.6);
const near = test.filter(c => c.dT >= 0.6 && c.dT < 0.96);
console.log(`  注：分数列取两式(线性/幂)的较小值；带 ">" 的是"位移已越过该 case 的 tbest、被夹住"⇒ 那一格只说明它已到位，分数不再可信上探。`);
console.log(`      保守/锚点/乐观 = 位移 × {0.615, OUR/自然off, 1.234}（§27.7：平台/本地比例逐格不闭合 ⇒ 一律三读）`);
console.log(`\n  过线格（ΔT≥0.96 且三种换算最低值≥1.6 分）：${ok.length ? ok.map(c => `${c.tag}/${c.shp}/b${c.blk}`).join('  ') : '无'}`);
console.log(`  落 0.6~0.96 的格（§26.8-②：不提交，回来改 L 封顶 + blk 联合重定）：${near.length ? near.map(c => `${c.tag}/b${c.blk} ΔT=${c.dT.toFixed(2)}`).join('  ') : '无'}`);

// === ③ 只有题面 §6 声明的三档形状能给判分 case 报价，其余格是"机制标定"不是"钱" ===
const DECL = [['64,256,2', '小规模'], ['1024,4096,4', '中规模'], ['8192,7168,8', '大规模']];
console.log('\n=== ③ 题面 §6 声明的三档（official_problem_statement.md:167）命中情况 ===');
for (const c of test) {
    const hit = DECL.find(([k]) => `${c.S},${c.D},${c.m}` === k);
    if (!hit) continue;
    const tile = c.D * 2, el = tile <= 6144 && (tile & 31) === 0 && c.S >= 56 && c.m >= 2 && c.tpc >= 2;
    console.log(`  ${c.tag} = ${hit[1]}档反向 blk=${c.blk}/自然${c.nat.toFixed(0)}：ΔT=${c.dT.toFixed(2)} 提交净=${isNaN(c.sub) ? '—(缺自然 off)' : c.sub.toFixed(2)}µs ` +
                `分=${isNaN(c.sub) ? '—' : c.sc.map((x, i) => (c.clip[i] ? '>' : '') + Math.min(x[0], x[1]).toFixed(1)).join('/')}  资格(${tile}B,mode0,tpc=${c.tpc})=${el ? '✅' : '⛔'}`);
}
const best = test.slice().sort((x, y) => y.dT - x.dT)[0];
if (best) {
    const v = ok.length ? '✅ 过 §26.8-① + §27.9-1 双端点门槛' :
              best.dT >= 0.96 ? '🟡 ΔT 过线但换算未过 1.6 分 ⇒ 按 §27.7 不算过线' :
              best.dT >= 0.6 ? '🟡 只到 0.6~0.96 ⇒ 不提交（§26.8-②）' :
              best.dT >= 0.15 ? '🟡 太小 ⇒ 不提交' : '⛔ ≈0 ⇒ H5c 判负（§26.8-③ / §27.9-3）⇒ 本题转"只有地板"结案';
    console.log(`  ⇒ 最肥一格 ${best.tag} (${best.shp} blk=${best.blk}) ΔT = ${best.dT.toFixed(2)}µs、f = ${(best.f * 100).toFixed(0)}%、每批残差 ${(best.perBatch * 1000).toFixed(0)}ns  ${v}`);
    const neg = test.filter(c => c.dT < -0.1);
    if (neg.length) console.log(`  ⚠️ 有 ${neg.length} 格 on 臂反而慢 >0.1µs：${neg.map(c => `${c.tag}/b${c.blk} ${(c.dT * 1000).toFixed(0)}ns`).join('  ')}`);
}
console.log('\n  注：`每批ns` = (T_on − A − B·blk)/批数。若它仍锁在 §27.3-② 的 280~320ns ⇒ 宽 VEC 真按发起计数');
console.log('      收费（这一刀成立）；若它随 L·D 的元素数一起涨 ⇒ 钱在元素吞吐上，这一族到此为止。');
console.log('      `预测off` 用 §26.2 的式子回代同一格，看的是"式子在 m>=2 域还站得住吗"（f 只对 on 臂有意义）。');
