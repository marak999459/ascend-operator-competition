// [题1/p20 离线核对工装，非提交面] 把 fit_p20.js 拟合出的三项常数拿去对三条**已落档的旧读数**，
// 目的不是"再拟一次"，而是看这条新项 k（每核每行）能不能吃进 §23.17-4 那笔"1.8µs 无归属"。
// 用法: node check_p20.js A B k     （缺省 = fit_p20.js 本轮读数，µs）
const [A, B, K] = [process.argv[2], process.argv[3], process.argv[4]].map(x => x ? +x : null)
    .map((x, i) => x === null ? [1.19583, 0.07121, 0.27009][i] : x);
const f = (blk, rows, k = K) => A + B * blk + k * rows;
console.log(`常数：A=${A.toFixed(5)}µs  B=${(B * 1000).toFixed(1)}ns/核  k=${(K * 1000).toFixed(1)}ns/(核·行)  [来源 fit_p20.js, m=1/S=64/D∈192~640]`);

console.log('\n① 单价核对（独立标定的交叉点）');
console.log(`   m=1 反向每行 5 次发起 ⇒ k/5 = ${(K * 1000 / 5).toFixed(1)} ns/次   vs §22 由 blk=1 前向独立标定的 52~58 ns/次`);

console.log('\n② 对已落档读数（不同构建，只核量级）');
const old = [
    ['§20.5  96KB 反向 3.5~3.6', f(16, 4), '3.55'],
    ['§20.2  反向 3KB @floor=8 = 2.0（若 S=8 ⇒ 1 行/核）', f(8, 1), '2.00'],
    ['§23.17-4  m=2, blk=16, tpc=4（每核 12 次发起）= 3.70', f(16, 4), '3.70'],
];
for (const [tag, pred, meas] of old) console.log(`   ${tag.padEnd(52)} 本式 ${pred.toFixed(2)}  实测 ${meas}  差 ${(pred - +meas).toFixed(2)}`);

const rows = 4, meas = 3.70, iss = 12;
const kM = (meas - A - B * 16) / rows;
console.log('\n③ 那笔"1.8µs 无归属"在新式下还剩多少');
console.log(`   旧式（§23.5 三参数）：1.2+16×0.08+12×0.055 = ${(1.2 + 16 * 0.08 + 12 * 0.055).toFixed(2)} ⇒ 记为缺 ≈1.8µs`);
console.log(`   新式若沿用 m=1 的 k：${f(16, rows).toFixed(2)} ⇒ 缺 ${(meas - f(16, rows)).toFixed(2)}µs`);
console.log(`   反解 m=2 该行需要的 k = ${(kM * 1000).toFixed(0)}ns/行 ⇒ 每次发起 ${(kM * 1000 / (iss / rows)).toFixed(0)}ns（3 次/行）vs m=1 的 ${(K * 1000 / 5).toFixed(0)}ns`);
console.log('   ⇒ 缺口从"一整笔没有归属的固定开销"缩成"m 轴上每行发起数的计价方式"，性质从"开销"变"参数未标定"');

console.log('\n④ 本式对 S 的地板（把 io 消掉之后，µs 档只跟"行数"和"核数"走）');
console.log(`   T_min(S) = A + 2·√(B·k·S) = ${A.toFixed(3)} + ${(2 * Math.sqrt(B * K)).toFixed(4)}·√S`);
for (const S of [1, 8, 16, 64, 128, 256]) console.log(`   S=${String(S).padStart(4)} 行 ⇒ 地板 ${(A + 2 * Math.sqrt(B * K * S)).toFixed(2)}µs`);
console.log(`   反解平台 c5 的 tbest=1.48µs ⇒ 需要 S ≈ ${((1.48 - A) / (2 * Math.sqrt(B * K))) ** 2} 行`);
console.log(`   反解我方 c5 读数 4.16µs（取 blk=16）⇒ 需要 rows/核 ≈ ${((4.16 - A - B * 16) / K).toFixed(1)} ⇒ S ≈ ${(((4.16 - A - B * 16) / K) * 16).toFixed(0)} 行`);
