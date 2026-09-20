// msprof op_summary.csv 摘要器（非提交工具）。用法：node prof_sum.js <op_summary.csv> [more.csv...]
// 方向按 Input Shapes 维度数判：2 维 = 前向（x=[S,D]），3 维 = 反向（grad=[S,m,D]）
const fs = require('fs');

function parseCsv(text) {
    const rows = [];
    let row = [], field = '', inQ = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (inQ) {
            if (c === '"') { if (text[i + 1] === '"') { field += '"'; i++; } else inQ = false; }
            else field += c;
        } else if (c === '"') inQ = true;
        else if (c === ',') { row.push(field); field = ''; }
        else if (c === '\n') { row.push(field); rows.push(row); row = []; field = ''; }
        else if (c !== '\r') field += c;
    }
    if (field.length) row.push(field);
    return rows.filter(r => r.length > 1);
}
const num = s => parseFloat(String(s).replace(/[^\d.eE+-]/g, ''));
function stat(a) {
    const s = [...a].sort((x, y) => x - y);
    const mean = s.reduce((x, y) => x + y, 0) / s.length;
    return { n: s.length, mean, min: s[0], max: s[s.length - 1], p50: s[(s.length - 1) >> 1] };
}
for (const path of process.argv.slice(2)) {
    const rows = parseCsv(fs.readFileSync(path, 'utf8'));
    const h = rows[0], ix = k => h.findIndex(c => c.trim().replace(/\(us\)$/, '') === k);
    const iIn = ix('Input Shapes'), iDur = ix('Task Duration'), iAiv = ix('aiv_time');
    const iVec = ix('aiv_vec_time'), iM2 = ix('aiv_mte2_time'), iM3 = ix('aiv_mte3_time');
    const iSc = ix('aiv_scalar_time');
    const iBlk = ix('Block Num'), iTid = ix('Task ID');
    const g = { fwd: [], bwd: [] };
    const sub = { fwd: [], bwd: [] };
    const v = { fwd: [], m2: [], m3: [], bwd: [], bvec: [], bm2: [] };
    for (const r of rows.slice(1)) {
        const dims = r[iIn].split(',').length;
        const k = dims === 2 ? 'fwd' : 'bwd';
        g[k].push(num(r[iDur]));
        if (k === 'fwd') { v.m2.push(num(r[iM2])); v.m3.push(num(r[iM3])); v.vec = v.vec || []; v.vec.push(num(r[iVec])); v.faiv = (v.faiv || []).concat(num(r[iAiv])); v.fsc = (v.fsc || []).concat(num(r[iSc])); }
        else { v.bvec.push(num(r[iVec])); v.bm2.push(num(r[iM2])); v.baiv = (v.baiv || []).concat(num(r[iAiv])); v.bsc = (v.bsc || []).concat(num(r[iSc])); }
        sub[k].push(num(r[iTid]));
    }
    const drop = a => a.slice(1);   // 首任务是 warm-up（含冷启动），A/B 一律剔除
    console.log(`# ${(path.split(/[\\/]/).find(s => /^v\d/.test(s)) || path)}  ${path.split(/[\\/]/).pop()}`);
    for (const k of ['fwd', 'bwd']) {
        if (!g[k].length) continue;
        const all = stat(g[k]), st = stat(drop(g[k]));
        console.log(`  ${k}: n=${all.n} 含首任务 mean=${all.mean.toFixed(1)}us | 剔首 mean=${st.mean.toFixed(1)} min=${st.min.toFixed(1)} p50=${st.p50.toFixed(1)} max=${st.max.toFixed(1)}us blk=${num(rows[1][iBlk])}`);
    }
    const m = a => (a && a.length ? (a.reduce((x, y) => x + y, 0) / a.length).toFixed(1) : '—');
    console.log(`  fwd 流水线均值(us) aiv=${m(drop(v.faiv || []))} mte2=${m(drop(v.m2))} mte3=${m(drop(v.m3))} vec=${m(drop(v.vec || []))} scalar=${m(drop(v.fsc || []))}`);
    console.log(`  bwd 流水线均值(us) aiv=${m(drop(v.baiv || []))} mte2=${m(drop(v.bm2))} vec=${m(drop(v.bvec))} scalar=${m(drop(v.bsc || []))}`);
}
