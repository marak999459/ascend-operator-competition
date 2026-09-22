// [题1 只读分析工装，非提交面] 从排名页 JSON 反解"总分是怎么由 8 条 case 算出来的"。
// 用法: node score_model_fit.js <dir_with_rank_p*.json> [我方队名正则，默认见 OUR_TEAM_RE]
// 输出: (1) 全场面值 CSV；(2) 候选打分函数的拟合残差；(3) 给定 c5/c1 变动时的分数敏感度；
//       (4) 逐 case 全场分布 + 我方名次；(5) "把我方第 i 条挤到某档面值 = 总分多几分"的两式精确差分。
const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || '/tmp';
const OUR_TEAM_RE = 'fszqsn';      // 与 连接信息.md 里那个账号名同源，不写全称
const ours = process.argv[3] || OUR_TEAM_RE;
const files = fs.readdirSync(dir).filter(f => /^rank_p\d+\.json$/.test(f)).sort();
if (!files.length) { console.error('no rank_p*.json in ' + dir); process.exit(1); }

let tids = null, tbest = null;
const rows = new Map();
for (const f of files) {
    const j = JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8'));
    if (!j.rows) continue;
    if (!tids) { tids = j.testcases.map(t => t._id); tbest = j.testcases.map(t => t.tbest); }
    for (const r of j.rows) {
        if (r.status !== 'Pass' || !Array.isArray(r.result) || r.result.length !== tids.length) continue;
        const byId = {};
        let allPos = true, caseScoreSum = 0;
        for (const c of r.result) {
            byId[c.testcase_id] = c;
            if (!(c.time > 0)) allPos = false;
            caseScoreSum += (c.score || 0);
        }
        const times = tids.map(id => (byId[id] ? byId[id].time : 0));
        if (!allPos) continue;
        rows.set(r.submission_id, {
            rank: r.rank, score: r.score,
            who: [r.team && r.team.team_name, r.user && r.user.nickname].filter(Boolean).join(' '),
            time: r.submission?.create_time || r.create_time, times, caseScoreSum,
        });
    }
}
const R = [...rows.values()].sort((a, b) => a.score - b.score);
console.log(`有效行 ${R.length} / 去重后；tbest = ${tbest.join(' ')}`);

// 我方所有在榜行（按队名匹配）+ 榜首/前 10 的逐 case 面值
const mine = R.filter(r => new RegExp(ours, 'i').test(r.who));
console.log('\n=== 我方在榜行 ===');
for (const m of mine) {
    console.log(`  rank=${m.rank} score=${m.score} ${m.time}  ${m.times.join(' ')}`);
}
console.log('\n=== 榜首 6 行 + 前 10 的 c1/c5 ===');
for (const m of R.slice(-6).reverse()) {
    console.log(`  rank=${String(m.rank).padStart(3)} score=${m.score.toFixed(2)}  c1=${String(m.times[0]).padStart(5)} c5=${String(m.times[4]).padStart(5)}  ${m.times.join(' ')}`);
}

// 逐 case score 字段是不是全 0（若非 0，说明总分另有构成）
const nz = R.filter(r => r.caseScoreSum !== 0).length;
console.log(`per-case score 求和非 0 的行数 = ${nz}  ⇒ ${nz ? '总分含逐 case score 项，需另解' : '逐 case score 恒为 0 ⇒ 总分只由 time 组合而来'}`);

const ratio = (r) => r.times.map((t, i) => tbest[i] / t);

// 候选：score = 100 * mean( min(ratio_i, cap)^p )，另测 "均值比"(先平均 time 再比) 与 "比值均值"
function resid(f) {
    let mx = 0, sum = 0, n = 0;
    for (const r of R) {
        const e = f(r), d = e - r.score;
        if (Math.abs(d) > mx) mx = Math.abs(d);
        sum += d * d; n++;
    }
    return { max: mx, rms: Math.sqrt(sum / n) };
}
const mean = a => a.reduce((x, y) => x + y, 0) / a.length;
const cands = [];
for (const p of [0.5, 0.75, 1, 1.25, 1.5, 2]) {
    cands.push([`mean(ratio^${p})*100 (无上限)`, r => 100 * mean(ratio(r).map(x => Math.pow(x, p)))]);
    cands.push([`mean(min(ratio,1)^${p})*100`, r => 100 * mean(ratio(r).map(x => Math.pow(Math.min(x, 1), p)))]);
}
cands.push(['100*mean(ratio) clip1 (线性均值)', r => 100 * mean(ratio(r).map(x => Math.min(x, 1)))]);
cands.push(['100 * tbest_mean/our_mean', r => 100 * mean(tbest) / mean(r.times)]);
cands.push(['几何均值 100*exp(mean ln ratio)', r => 100 * Math.exp(mean(ratio(r).map(Math.log)))]);
cands.push(['调和均值 100/mean(1/ratio)', r => 100 / mean(ratio(r).map(x => 1 / x))]);
console.log('\n=== 候选打分函数拟合（对全场面值行）===');
console.log('式子                                 RMS    最大绝对残差');
for (const [name, f] of cands) {
    const s = resid(f);
    console.log(`${name.padEnd(34)} ${s.rms.toFixed(2).padStart(6)} ${s.max.toFixed(2).padStart(12)}`);
}

// ---- 线性回归：score = sum_i w_i * ratio_i （8 个未知数、79 个方程）----
// 若残差能压到 ~0.01 级，就说明"总分 = 逐 case 比值的加权和"，而 w 直接告诉我们**每条 case 值多少分**。
function solve(A, b) {
    const n = b.length, M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
        let piv = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
        [M[c], M[piv]] = [M[piv], M[c]];
        const d = M[c][c] || 1e-12;
        for (let r = 0; r < n; r++) {
            if (r === c) continue;
            const f = M[r][c] / d;
            for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
        }
    }
    return M.map((row, i) => row[n] / row[i]);
}
for (const [label, tf] of [['ratio', x => x], ['ratio^2', x => x * x], ['ln(1/ratio) 取负', x => -Math.log(x)]]) {
    const AtA = Array.from({ length: 8 }, () => Array(8).fill(0)), Atb = Array(8).fill(0);
    for (const r of R) {
        const v = ratio(r).map(tf);
        for (let i = 0; i < 8; i++) {
            Atb[i] += v[i] * r.score;
            for (let j = 0; j < 8; j++) AtA[i][j] += v[i] * v[j];
        }
    }
    const w = solve(AtA, Atb);
    let mx = 0, sum = 0;
    for (const r of R) {
        const v = ratio(r).map(tf);
        const e = v.reduce((a, x, i) => a + x * w[i], 0) - r.score;
        if (Math.abs(e) > mx) mx = Math.abs(e);
        sum += e * e;
    }
    console.log(`\n=== 回归 score = sum w_i * ${label} ===`);
    console.log('  w = ' + w.map(x => x.toFixed(2).padStart(7)).join(' '));
    console.log(`  RMS=${Math.sqrt(sum / R.length).toFixed(3)}  max|res|=${mx.toFixed(3)}`);
}

// ---- 最简可用工：score ≈ 100*mean(ratio) + b（单偏置）。先看 b 是不是常数，再决定能不能拿它当决策尺 ----
const lin = R.map(r => ({ p: 100 * mean(r.times.map((t, i) => tbest[i] / t)), s: r.score, r }));
const d = lin.map(x => x.p - x.s);
const sd = Math.sqrt(mean(d.map(x => (x - mean(d)) ** 2)));
console.log(`\n\n=== 线性比值模型 score ≈ 100*mean(tbest/time) + b ===`);
console.log(`  b = ${(-mean(d)).toFixed(2)}  残差标准差 ${sd.toFixed(2)}  全距 ${Math.min(...d).toFixed(2)}~${Math.max(...d).toFixed(2)}`);
const sorted = [...lin].sort((a, b) => b.s - a.s);
for (let k = 0; k < 4; k++) {
    const g = sorted.slice(k * 20, (k + 1) * 20);
    if (!g.length) break;
    const gd = g.map(x => x.p - x.s);
    console.log(`  分数桶 ${k}（score ${g[g.length - 1].s.toFixed(1)}~${g[0].s.toFixed(1)}）偏置均值 ${mean(gd).toFixed(2)}  散布 ${Math.min(...gd).toFixed(2)}~${Math.max(...gd).toFixed(2)}`);
}
const show = (r, tag) => console.log(`  ${tag} ratio: ` + r.times.map((t, i) => (tbest[i] / t).toFixed(3)).join(' ') + `   score=${r.score}`);
show(R.find(r => r.rank === 1), '榜首  ');
for (const m of mine) show(m, '我方  ');

// ---- 决策尺：每条 case 上"省 1µs 值几分"= (100/8) * tbest / t^2 ----
console.log(`\n=== 敏感度（线性比值模型下）分/µs = 12.5 * tbest / t^2 ===`);
console.log('  case   tbest      榜首面值→分/µs        我方面值→分/µs');
for (let i = 0; i < 8; i++) {
    const tb = tbest[i], lead = R.find(r => r.rank === 1).times[i];
    const m = mine.length ? mine[0].times[i] : null;
    const f = t => (12.5 * tb / (t * t)).toFixed(2);
    console.log(`   c${i + 1}  ${String(tb).padStart(8)}   ${String(lead).padStart(8)} → ${f(lead).padStart(6)}      ` +
        (m ? `${String(m).padStart(8)} → ${f(m).padStart(6)}` : ''));
}

// ---- 每条 case 的全场分布 + 我方名次：用来"反推判分档的形状上界"（别人做到过 X ⇒ 该档 io 必须容得下 X）----
console.log('\n\n=== 逐 case 全场分布（79 条有效行，µs 档看左尾）===');
const myRow = mine[0];
for (let i = 0; i < 8; i++) {
    const v = R.map(r => r.times[i]).sort((a, b) => a - b);
    const q = p => v[Math.min(v.length - 1, Math.floor(p * v.length))];
    const myT = myRow ? myRow.times[i] : NaN;
    const better = v.filter(x => x < myT).length;
    console.log(`  c${i + 1} tbest=${String(tbest[i]).padStart(8)}  min=${String(v[0]).padStart(8)} p10=${String(q(0.1)).padStart(8)}` +
        ` p25=${String(q(0.25)).padStart(8)} 中位=${String(q(0.5)).padStart(8)} p75=${String(q(0.75)).padStart(8)}` +
        `  我方=${String(myT).padStart(8)}（我方名次 ${better + 1}/${v.length}）`);
}

// ---- "追平到某一档值几分"：两式**精确差分**（不用导数近似，避免低位外推失真）----
// 线性式 12.5*Σ(tbest/t) 与幂式 12.5*Σ(tbest/t)^1.5，只动第 i 条、其余七条不变。
console.log('\n\n=== 把第 i 条从我方面值挤到"某档面值"= 总分增量（两式精确差分）===');
if (myRow) {
    const cols = [[0, 'min'], [0.1, 'p10'], [0.25, 'p25'], [0.5, '中位']];
    console.log('  case  我方  ' + cols.map(c => `${c[1]}值(分)`.padStart(13)).join(''));
    for (let i = 0; i < 8; i++) {
        const v = R.map(r => r.times[i]).sort((a, b) => a - b);
        const q = p => v[Math.min(v.length - 1, Math.floor(p * v.length))];
        const term = (t, p) => Math.pow(tbest[i] / t, p);
        const cells = cols.map(([pp, nm]) => {
            const tgt = q(pp);
            if (tgt >= myRow.times[i]) return '   —(已优于)'.padStart(13);
            const dl = 12.5 * (term(tgt, 1) - term(myRow.times[i], 1));
            const dp = 12.5 * (term(tgt, 1.5) - term(myRow.times[i], 1.5));
            return `${dl.toFixed(2)}|${dp.toFixed(2)}`.padStart(13);
        });
        console.log(`   c${i + 1} ${String(myRow.times[i]).padStart(7)}  ` + cells.join(''));
    }
    console.log('  （每格 = 线性式 | 幂式；"—" 表示我方已在该档之前）');
}

// ---- 配对法（独立于任何公式）：找"只有一条 case 明显不同、其余七条几乎相同"的两行 ⇒ 直接读 分/µs ----

console.log('\n=== 配对法：单 case 位移 → 总分位移（其余七条 |Δ|≤tol 才算有效对）===');
for (const tol of [0.015, 0.03]) {
    console.log(`\n-- 其余 case 相对差 ≤ ${(tol * 100).toFixed(1)}% --`);
    const acc = Array.from({ length: 8 }, () => []);
    const arr = [...rows.values()];
    for (let a = 0; a < arr.length; a++) {
        for (let b = a + 1; b < arr.length; b++) {
            const ra = arr[a], rb = arr[b];
            let cand = -1, ok = true, nbig = 0;
            for (let i = 0; i < 8; i++) {
                const d = Math.abs(ra.times[i] - rb.times[i]) / Math.max(ra.times[i], rb.times[i]);
                const absd = Math.abs(ra.times[i] - rb.times[i]);
                if (absd < 0.05) continue;              // 差太小，反推不出斜率
                if (d <= tol) continue;
                nbig++; cand = i;
            }
            if (nbig !== 1 || cand < 0) continue;
            const dt = ra.times[cand] - rb.times[cand];
            const ds = ra.score - rb.score;
            if (Math.abs(dt) < 0.05) continue;
            acc[cand].push({ ds, dt, ratio: ds / dt, i: cand });
        }
    }
    for (let i = 0; i < 8; i++) {
        const g = acc[i];
        if (!g.length) { console.log(`  c${i + 1} (tbest ${tbest[i]}): 0 对`); continue; }
        const rs = g.map(x => x.ratio).sort((x, y) => x - y);
        const med = rs[Math.floor(rs.length / 2)];
        const ex = g.sort((x, y) => Math.abs(x.dt) - Math.abs(y.dt));
        console.log(`  c${i + 1} (tbest ${String(tbest[i]).padStart(8)}): ${String(g.length).padStart(3)} 对  中位斜率 ${med.toFixed(3)} 分/µs` +
            `   例: ${ex.slice(0, 2).map(x => `Δt=${x.dt.toFixed(2)}µs→Δscore=${x.ds.toFixed(2)}`).join(' ; ')}`);
    }
}

