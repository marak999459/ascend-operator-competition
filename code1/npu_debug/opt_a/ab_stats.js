// msprof A/B 配对统计器（非提交工具）。用法：
//   node ab_stats.js <case>:<label>:<op_summary.csv> [...more...]
// 输出每个 (case,label) 的剔首 mean / sd / sem / 截尾 mean，并对第一个 label 做差值与 z。
// 为什么要 sem：prof_sum.js 把 mean 打成 0.1us 一位小数，而 A1 这类改动预期只有 1~2%，
// 一位小数的分辨率（0.1us = 3.3%）根本判不动，必须带上不确定度。
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
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const sd = a => Math.sqrt(a.reduce((x, y) => x + (y - mean(a)) ** 2, 0) / (a.length - 1));
function pctl(sorted, q) { return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]; }

const groups = new Map();
for (const arg of process.argv.slice(2)) {
    const [cs, label, ...rest] = arg.split(':');
    const path = rest.join(':');
    const rows = parseCsv(fs.readFileSync(path, 'utf8'));
    const h = rows[0], ix = k => h.findIndex(c => c.trim().replace(/\(us\)$/, '') === k);
    const iDur = ix('Task Duration');
    const all = rows.slice(1).map(r => num(r[iDur])).filter(v => !Number.isNaN(v));
    const d = all.slice(1);                       // 剔首（warm-up）
    const sorted = [...d].sort((x, y) => x - y);
    const trimmed = sorted.slice(0, Math.ceil(sorted.length * 0.8));  // 去掉最慢 20% 的长尾
    if (!groups.has(cs)) groups.set(cs, []);
    groups.get(cs).push({
        label, n: d.length,
        mean: mean(d), sem: sd(d) / Math.sqrt(d.length),
        p50: pctl(sorted, 0.5), tmean: mean(trimmed),
        tsem: sd(trimmed) / Math.sqrt(trimmed.length),
    });
}
for (const [cs, arr] of groups) {
    const base = arr[0];
    console.log(`[${cs}]`);
    for (const s of arr) {
        const dm = s.mean - base.mean, ds = Math.sqrt(s.sem ** 2 + base.sem ** 2) || 1e-9;
        const dt = s.tmean - base.tmean, dts = Math.sqrt(s.tsem ** 2 + base.tsem ** 2) || 1e-9;
        console.log(`  ${s.label.padEnd(8)} n=${s.n} mean=${s.mean.toFixed(3)}±${s.sem.toFixed(3)}` +
            ` Δ=${(dm >= 0 ? '+' : '')}${dm.toFixed(3)}us(${(100 * dm / base.mean).toFixed(2)}%,z=${(dm / ds).toFixed(1)})` +
            ` | 截尾 mean=${s.tmean.toFixed(3)}±${s.tsem.toFixed(3)}` +
            ` Δ=${(dt >= 0 ? '+' : '')}${dt.toFixed(3)}us(${(100 * dt / base.tmean).toFixed(2)}%,z=${(dt / dts).toFixed(1)})` +
            ` p50=${s.p50.toFixed(2)}`);
    }
}
