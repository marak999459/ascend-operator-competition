#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P35：用**榜单全量行**标定"平台 score 到底是时间的什么函数"，从而回答一个卡了本项目的
可判读性问题：**两发提交之间 score 差多少才算真收益**（§15.42 只给了"单点时间噪声 ≥10 %"，
§15.52(a)/§15.54(3) 却靠"同码两发都是 20.64"反推 score 免疫噪声 —— 只有 3 个点，撑不住）。

做法：拉排行榜若干页 ⇒ 每行取 (score, 六点时间) ⇒ 在全量行上试几种候选聚合
（Σ1/t、Σ tbest/t、Σlog t、几何均值…）与 score 的秩相关 / 对数-对数斜率；
再用"每点独立 ±11 % 均匀抖动"蒙特卡洛，算出**score 自身的噪声带**（这才是 P32 的
22.03 vs P25 的 22.17 该按几分之几来读的尺子）。
只读接口，不提交、不写任何凭据。
"""
import json
import math
import os
import random
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
ME = os.environ.get('ME', '')


def fetch():
    c = CANNJudgeClient()
    if not c.load_session():
        print('NO_SESSION'); return []
    rows = []
    for page in range(1, 8):
        try:
            d = c._get_json('/api/problems/%s/ranking' % PID,
                            params={'page': page, 'size': 80})
        except Exception as e:            # noqa: BLE001
            print('PAGE_ERR', page, repr(e)[:120]); break
        rr = (d or {}).get('rows') or (d or {}).get('data', {}).get('rows') or []
        if not rr:
            break
        rows.extend(rr)
        if len(rr) < 80:
            break
    print('PAGES_ROWS %d' % len(rows))
    return rows


def parse(rows):
    out = []
    for r in rows:
        sc = r.get('score')
        res = r.get('result')
        if not isinstance(res, list) or not res:
            continue
        ts, prs = [], []
        for it in res:
            if not isinstance(it, dict):
                continue
            t = it.get('time')
            if t is None:
                t = (it.get('metrics') or {}).get('time')
            ts.append(float(t) if t else 0.0)
            prs.append(it.get('precision_ratio'))
        if len(ts) < 6 or min(ts) <= 0:
            continue
        try:
            sc = float(sc)
        except (TypeError, ValueError):
            continue
        if sc <= 0:
            continue
        out.append((sc, ts[:6], str(r.get('user_name') or r.get('user') or '')[:18]))
    return out


def spearman(x, y):
    def rk(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for kk in range(i, j + 1):
                r[idx[kk]] = avg
            i = j + 1
        return r
    a, b = rk(x), rk(y)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = math.sqrt(sum((a[i] - ma) ** 2 for i in range(n)) *
                    sum((b[i] - mb) ** 2 for i in range(n)))
    return num / den if den else float('nan')


def main():
    data = parse(fetch())
    if len(data) < 10:
        print('TOO_FEW_ROWS', len(data)); return
    T = [d[1] for d in data]
    S = [d[0] for d in data]
    tbest = [min(t[i] for t in T) for i in range(6)]
    print('N=%d tbest=%s' % (len(data), ' '.join('%.2f' % x for x in tbest)))
    cands = {
        'sum 1/t': [sum(1.0 / x for x in t) for t in T],
        'sum tbest/t': [sum(tb / x for tb, x in zip(tbest, t)) for t in T],
        'sum log t': [sum(math.log(x) for x in t) for t in T],
        'geo mean t': [math.exp(sum(math.log(x) for x in t) / 6.0) for t in T],
        'sum t': [sum(t) for t in T],
        'prod(t)^-1/6': [1.0 / math.exp(sum(math.log(x) for x in t) / 6.0) for t in T],
    }
    print('--- 秩相关 |Spearman(score, agg)| ---')
    best = None
    for name, v in cands.items():
        rho = spearman(S, v) if name not in ('sum log t', 'geo mean t', 'sum t') else spearman(S, [-x for x in v])
        print('%-14s rho=%+.4f' % (name, rho))
        if best is None or abs(rho) > abs(best[1]):
            best = (name, rho, v)
    name, rho, v = best
    print('BEST %s rho=%+.4f' % (name, rho))
    # 对数-对数斜率：log(score) = a + b*log(agg)
    xs = [math.log(x) for x in v]
    ys = [math.log(s) for s in S]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / \
        sum((xs[i] - mx) ** 2 for i in range(n))
    a = my - b * mx
    resid = [ys[i] - (a + b * xs[i]) for i in range(n)]
    rms = math.sqrt(sum(r * r for r in resid) / n)
    print('fit: score = %.4f * (%s)^%.4f    残差 rms(log)=%.4f (=%.2f%%)' %
          (math.exp(a), name, b, rms, 100 * (math.exp(rms) - 1)))
    # 我们自己那几行
    for sc, t, who in data:
        if ME and ME.lower() in who.lower():
            agg = sum(tb / x for tb, x in zip(tbest, t))
            print('ME row: score=%.2f agg=%.4f pred=%.2f times=%s' %
                  (sc, agg, math.exp(a + b * math.log(agg)),
                   ' '.join('%.2f' % x for x in t)))
    # 蒙特卡洛：每点独立 ±11% 均匀抖动（§15.42 量到的同码两发散布）⇒ score 自身的噪声带
    def pred(t):
        agg = sum(tb / x for tb, x in zip(tbest, t))
        return math.exp(a + b * math.log(agg))

    random.seed(7)
    P = {'P25': [7.68, 7.56, 11.34, 12.84, 11.00, 15.68],
         'P32': [7.82, 8.38, 12.08, 12.00, 11.04, 14.72]}
    band = {}
    for tag, base in P.items():
        preds = sorted(pred([x * random.uniform(0.89, 1.11) for x in base])
                       for _ in range(4000))
        lo, hi = preds[int(0.025 * len(preds))], preds[int(0.975 * len(preds))]
        band[tag] = (pred(base), lo, hi)
        print('%s: 无抖动预测 %.2f   实测 %.2f   95%% 抖动带 [%.2f, %.2f] ⇒ 单发半宽 ±%.2f' %
              (tag, pred(base), {'P25': 22.17, 'P32': 22.03}[tag], lo, hi, (hi - lo) / 2))
    # 两发之差：两次独立测量各自抖动 ⇒ 差的 95 % 带
    diffs = []
    random.seed(11)
    for _ in range(4000):
        t25 = [x * random.uniform(0.89, 1.11) for x in P['P25']]
        t32 = [x * random.uniform(0.89, 1.11) for x in P['P32']]
        diffs.append(pred(t32) - pred(t25))
    diffs.sort()
    q1, q99 = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
    print('两发之差 P32−P25：无抖动 %.3f   95%% 带 [%.3f, %.3f]   实测 −0.14 ⇒ %s' %
          (pred(t32) - pred(t25), q1, q99,
           '在带内（不可判）' if q1 < -0.14 < q99 else '出带（可判）'))


if __name__ == '__main__':
    main()
