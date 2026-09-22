// [题1/R18 工装，非提交面] 聚合 h3_sym 的 4-run 块读数：off,on,on,off
// 每臂占 slot{1,4} / slot{2,3} 各一次；m1 这条 (m<2) 两臂是同一份机器码 ⇒ 它的
// off-on 差 = 本装置残留的"臂/位置"偏差，直接拿去折其它臂才讲得通。
const fs = require("fs");
const file = process.argv[2] || "r18_h3sym.log";
const tagOf = { m1: "bwd,fp16,64,1024,1", a2: "bwd,fp16,64,1024,2", a4: "bwd,fp16,64,1024,4",
                a8: "bwd,fp16,64,1024,8", t1a8: "bwd,fp16,40,1024,8" };
const rows = fs.readFileSync(file, "utf8").split(/\r?\n/).filter(l => /^S /.test(l)).map(l => ({
  tag: l.split(" ")[1],
  r: +l.match(/\br(\d)\b/)[1],
  slot: +l.match(/slot(\d)/)[1],
  arm: l.match(/arm=(\w+)/)[1],
  got: +l.match(/got=(\d+)/)[1],
  pass: /ALL PASS/.test(l),
  mean: +l.match(/mean=([0-9.]+)/)[1],
  shape: l.match(/shape=(\S+)/)[1],
}));
const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
const A = {};
for (const r of rows) {
  A[r.tag] = A[r.tag] || { off: [], on: [], slot: { 1: [], 2: [], 3: [], 4: [] }, shape: r.shape, nfail: 0, ngot: new Set() };
  A[r.tag][r.arm].push(r.mean);
  A[r.tag].slot[r.slot].push(r.mean);
  if (!r.pass) A[r.tag].nfail++;
  A[r.tag].ngot.add(r.got);
}
console.log("tag   shape                 off      on      d_us    d_rel  slot1/2/3/4        got  pass");
for (const k of Object.keys(tagOf)) {
  const v = A[k];
  if (!v) { console.log(k, "MISSING"); continue; }
  const o = avg(v.off), n = avg(v.on), d = n - o;
  console.log(k.padEnd(5), v.shape.padEnd(21),
    o.toFixed(3).padStart(6), n.toFixed(3).padStart(7),
    d.toFixed(3).padStart(7), (100 * d / o).toFixed(1).padStart(6) + "%",
    " [" + [1, 2, 3, 4].map(s => avg(v.slot[s]).toFixed(2)).join(" ") + "]",
    [...v.ngot].join("/").padStart(4), (v.off.length + v.on.length - v.nfail) + "/" + (v.off.length + v.on.length));
}
const res = avg(A.m1.on) - avg(A.m1.off);
console.log("\nm1 对照（两臂同一份机器码）残差 = " + res.toFixed(3) + " us  <- 其它臂的 d_us 要先折掉它");
for (const k of Object.keys(tagOf)) {
  if (k === "m1" || !A[k]) continue;
  const o = avg(A[k].off), n = avg(A[k].on);
  const d = (n - o) - res;
  console.log("  " + k.padEnd(5) + " 折残差后 d=" + d.toFixed(3) + " us (" + (100 * d / o).toFixed(1) + "% of off)");
}
const pooled = [1, 2, 3, 4].map(s => { const a = []; for (const k in A) a.push(...A[k].slot[s]); return avg(a); });
console.log("位置均值（全形状合并） slot1..4 = " + pooled.map(x => x.toFixed(2)).join(" ") +
            "  极差 " + (Math.max(...pooled) - Math.min(...pooled)).toFixed(2) + " us");
console.log("n = " + rows.length + " 行 / rounds = " + Math.max(...rows.map(r => r.r)));
