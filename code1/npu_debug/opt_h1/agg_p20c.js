// [题1/p20 第 1.5b 阶离线判读工装，非提交面] 读 run_p20c_fwdfam.sh 的 8-run 平衡块日志，
// 逐格给 Δ = mean(off) − mean(on)（合批摊薄值多少钱），并按 §25.3 的 k=270ns/(核·行) 反解
// "off 臂相当于 tpc 次行任务"时的隐含 k。两块对照：IDENT 与反向负对照必须 ≈0，否则整组判脏。
// 用法: node agg_p20c.js <p20c.log>
const fs = require('fs');
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const g = {};
for (const l of fs.readFileSync(process.argv[2], 'utf8').split(/\r?\n/)) {
    if (!l.startsWith('S ')) continue;
    const arm = l.match(/arm=(\w+)/)[1];
    const blk = +l.match(/ b(\d+)/)[1];
    const shp = l.match(/shape=(\S+)/)[1].split(',');
    const S = +shp[2], D = +shp[3], m = +shp[4];
    const r = +(l.match(/ r(\d+) /) || [, '0'])[1], slot = +l.match(/slot(\d+)/)[1];
    // ⚠️ runner v1 的 nomenclature 缺陷：one() 把"额外 env"当第 4 参，于是 off/on 臂的日志里
    //   r 字段印成了 rMHC_NO_MERGE=1 / r-（轮号丢失）。不影响判读：两臂各占 slot{1,2,3,4} 一次
    //   （on: r1s1,r1s4,r2s2,r2s3；off: r1s2,r1s3,r2s1,r2s4），平衡性由 slot 覆盖即可证。
    const key = `${arm}|b${blk}|${shp.slice(0, 5).join(',')}`;
    (g[key] = g[key] || { arm, blk, S, D, m, slots: {} }).slots[slot] = (g[key].slots[slot] || 0) + 1;
    (g[key].T = g[key].T || []).push(+l.match(/mean=([0-9.]+)/)[1]);
    if (!/ALL PASS/.test(l)) g[key].bad = (g[key].bad || 0) + 1;
}
const cells = Object.values(g).map(c => ({ ...c, T: mean(c.T), n: c.T.length, spread: Math.max(...c.T) - Math.min(...c.T) }));
const find = t => cells.find(c => c.arm === t);
const out = [];
for (const off of cells.filter(c => /off$/.test(c.arm) && !/^(IDENT|bwd)/.test(c.arm))) {
    const base = off.arm.replace(/off$/, 'on');
    const on = find(base);
    if (!on) { out.push(`${base}: 缺 on 臂`); continue; }
    const tpc = off.S / off.blk, L = Math.min(tpc, Math.floor(16384 / (off.D * 2)));
    const d = off.T - on.T;
    const kimp = d / (tpc * (1 - 1 / L)) * 1000;
    out.push(`${base.padEnd(9)} b${String(off.blk).padEnd(3)} S${off.S} m${off.m} | on=${on.T.toFixed(2)}(±${(on.spread / 2).toFixed(2)}) off=${off.T.toFixed(2)}(±${(off.spread / 2).toFixed(2)}) n=${on.n}/${off.n} Δ=${(d >= 0 ? '+' : '')}${(d * 1000).toFixed(0)}ns | tpc=${tpc} L=${L} ⇒ 隐含 k=${kimp.toFixed(0)}ns/(核·行)（stage-1 反向 m=1 = 270）`);
    if (off.bad || on.bad) out[out.length - 1] += `  ⚠️ 非 ALL PASS off=${off.bad || 0}/on=${on.bad || 0}`;
}
const id = [find('IDENTon'), find('IDENToff')].filter(Boolean);
const bw = [find('bwdon'), find('bwdoff')].filter(Boolean);
console.log('=== 对照先判（不为 0 ⇒ 整组作废）===');
if (id.length === 2) console.log(`  IDENT  on=${id[0].T.toFixed(2)} off=${id[1].T.toFixed(2)} Δ=${((id[1].T - id[0].T) * 1000).toFixed(0)}ns  ← 装置本身`);
if (bw.length === 2) console.log(`  反向   on=${bw[0].T.toFixed(2)} off=${bw[1].T.toFixed(2)} Δ=${((bw[1].T - bw[0].T) * 1000).toFixed(0)}ns  ← NO_MERGE 不该动反向（merge_ok 含 !backward）`);
console.log('\n=== 逐格 Δ（前向合批 on → off）===');
console.log(out.join('\n'));
const chg = cells.filter(c => /off$/.test(c.arm) && !c.arm.startsWith('IDENT'));
console.log(`\n格子数 = ${chg.length}（必须 >0 才有读数）；槽位占用：`);
for (const c of chg) console.log(`  ${c.arm} b${c.blk} slots=${JSON.stringify(c.slots)}`);
