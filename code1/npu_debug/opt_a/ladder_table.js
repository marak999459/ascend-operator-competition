// [题1/R11~R12 工装，非提交面] 把 run_r11b.sh / run_r12.sh 的日志转成 (模态或变体 × 核数) 表。
//   node npu_debug/opt_a/ladder_table.js <log> [...]
// 为什么要有这个：一轮联合扫描是 54~78 行"@@@@ + c0/c1 …"交错文本，人眼对齐三次就会看错列
// （本项目已因"读错行"废过判定）。这里固定输出：每格 = 各轮剔首 mean 的中位数，
// 括号里给全部读数，最后一列给"该模态跨核数的最小值"，避免只盯单格噪声。
// 格尾的 @NN 是**设备上报**的 block 数（数据行 blk=），不是 host 预测值：
// R11-B 的教训是"强制核数"与"实际上报核数"必须同时留痕，否则一次 tiling 静默变化就能伪造整条曲线。
const fs = require('fs');
const files = process.argv.slice(2);
let mode = '?', blk = '?', round = '?';
const data = {};   // case -> mode -> blk -> [values]
const devs = {};   // case -> mode -> blk -> Set(设备上报 blk)
for (const f of files) {
    for (const line of fs.readFileSync(f, 'utf8').split('\n')) {
        let m = line.match(/^@@@@ round=(\d+) (?:mode|variant)=(\S+)(?: blk=(\S+))?/);
        if (m) { round = m[1]; mode = m[2]; blk = m[3] ? m[3].replace(/^b/, '') : '-'; continue; }
        m = line.match(/^(c\d+)\s+(fwd|bwd): .*剔首 mean=([\d.]+)(?:.*\bblk=(\d+))?/);
        if (!m) continue;
        const [, cs, dir, v, dev] = m;
        const key = `${cs}/${dir}`;
        data[key] ??= {}; data[key][mode] ??= {}; data[key][mode][blk] ??= [];
        data[key][mode][blk].push(parseFloat(v));
        (devs[key] ??= {})[mode] ??= {};
        (devs[key][mode][blk] ??= new Set()).add(dev === undefined ? '?' : dev);
    }
}
const med = a => { const s = [...a].sort((x, y) => x - y); const n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; };
for (const [key, modes] of Object.entries(data)) {
    const blks = [...new Set(Object.values(modes).flatMap(o => Object.keys(o)))]
        .sort((a, b) => (a === 'nat' ? 1e9 : +a) - (b === 'nat' ? 1e9 : +b));   // nat 单列放最后
    const ml = Math.max(...Object.keys(modes).map(s => s.length));
    console.log(`\n### ${key}  剔首 mean 中位数 us（@NN=设备上报核数，括号=全部读数）`);
    console.log('mode'.padEnd(ml) + ' ' + blks.map(b => `blk${b}`.padStart(18)).join('') + '   best');
    for (const md of Object.keys(modes)) {
        let best = Infinity, bestb = '';
        const cells = blks.map(b => {
            const v = modes[md][b];
            if (!v) return ' '.repeat(18);
            const me = med(v);
            if (me < best) { best = me; bestb = b; }
            const dv = [...new Set(devs[key][md][b])].join('+');
            return `${me.toFixed(1)}@${dv}(${v.map(x => x.toFixed(1)).join('/')})`.padStart(18);
        });
        console.log(md.padEnd(ml) + ' ' + cells.join('') + `   ${best.toFixed(1)} @${bestb}`);
    }
}
