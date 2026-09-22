#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P51：把"我们和榜前的差距"按用例拆开 —— 每点差多少倍、各点贡献多少分差。

口径（§15.54(4) 已标定）：平台 score 是 `Σ_i (tbest_i / t_i)` 的单调函数（37 行上 Spearman
+1.0000），所以"分差"可以逐点拆：某点我们对 tbest 的比值越小，那点贡献的分数越少。
tbest_i = 该用例上全服最快（只取 Pass 且有分数的行）。

用法（真机）：
  ME=<user_id>        取榜上我们自己那一行（默认题 3 账号的 id）
  OURS=7.68,8.34,…    改用这六点算（分析"恢复 P38 之后"的差距时给 P38 的时间）
"""
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
ME = os.environ.get('ME', '6aa96550b0477ec41e91eefc')
OURS = [float(x) for x in os.environ.get('OURS', '').split(',') if x.strip()]

c = CANNJudgeClient()
if not c.load_session():
    print('NO_SESSION'); sys.exit(1)

rows = []
for page in range(1, 9):
    try:
        d = c._get_json('/api/problems/%s/ranking' % PID, params={'page': page, 'size': 80})
    except Exception as e:  # noqa: BLE001
        print('PAGE_ERR', repr(e)[:100]); break
    rr = (d or {}).get('rows') or (d or {}).get('data', {}).get('rows') or []
    if not rr:
        break
    rows.extend(rr)
    if len(rr) < 80:
        break


def times(r):
    return [t.get('time') for t in (r.get('result') or [])]


def ident(r):
    u = r.get('user')
    return {str(r.get('user_id') or ''), str(r.get('id') or ''),
            (str(u.get('id') or '') if isinstance(u, dict) else str(u or ''))}


ok = []
for r in rows:
    t = times(r)
    if r.get('status') in ('Pass', 'Accepted') and r.get('score') and len(t) == 6 \
            and all(isinstance(x, (int, float)) and x > 0 for x in t):
        ok.append((float(r['score']), t, ident(r)))
ok.sort(key=lambda x: -x[0])
print('ROWS %d  USABLE %d' % (len(rows), len(ok)))
if len(ok) < 3:
    for r in rows[:4]:
        print('RAWKEYS', sorted(r.keys()))
        print('RAWID', {k: v for k, v in r.items() if 'user' in k.lower() or k == 'id'})
    sys.exit(0)

NC = 6
tbest = [min(t[i] for _, t, _ in ok) for i in range(NC)]
mine = (None, OURS) if len(OURS) == NC else None
if mine is None:
    for sc, t, ids in ok:
        if ME in ids:
            mine = (sc, t); break
if mine is None:
    print('MINE_NOT_FOUND  ids样本=', sorted(ok[0][2])[:3]); sys.exit(1)
lead = ok[0]


def contrib(t):
    return [tbest[i] / t[i] for i in range(NC)]


cb_l, cb_m = contrib(lead[1]), contrib(mine[1])
sl, sm = sum(cb_l), sum(cb_m)
print('TBEST   ', ['%.2f' % x for x in tbest])
print('LEAD#1  %.2f' % lead[0], lead[1])
print('OURS    %s %s' % (('%.2f' % mine[0]) if mine[0] else '(给定)', mine[1]))
print('SUM(tbest/t)  lead=%.4f  ours=%.4f  差 %.2f 分当量（=%.2fx）' % (sl, sm, sl - sm, sl / sm))
print('')
print('case   tbest   lead#1   ours    lead比值  ours比值   占总分差   我们/榜首')
for i in range(NC):
    share = (cb_l[i] - cb_m[i]) / (sl - sm) if sl > sm else 0
    print('C%d   %6.2f  %6.2f  %6.2f   %6.3f   %6.3f    %6.1f%%     %5.2fx' %
          (i + 1, tbest[i], lead[1][i], mine[1][i], cb_l[i], cb_m[i], 100 * share,
           mine[1][i] / lead[1][i]))
print('')
print('散布(最大/最小)  lead=%.2fx  ours=%.2fx  tbest=%.2fx' %
      (max(lead[1]) / min(lead[1]), max(mine[1]) / min(mine[1]), max(tbest) / min(tbest)))
for i in range(NC):
    v = sorted(t[i] for _, t, _ in ok)
    print('C%d 全服: min=%.2f p25=%.2f 中位=%.2f p75=%.2f 我们=%.2f 慢于我们 %d/%d 行' %
          (i + 1, v[0], v[len(v) // 4], v[len(v) // 2], v[3 * len(v) // 4], mine[1][i],
           sum(1 for x in v if x > mine[1][i]), len(v)))

# ---- 用全量行拟合 score–Σ(tbest/t) 的 log-log 斜率，再拿它给"整体快 f 倍"定价 ----
import math  # noqa: E402

pts = []
for sc, t, _ in ok:
    s = sum(tbest[i] / t[i] for i in range(NC))
    if s > 0:
        pts.append((math.log(s), math.log(sc)))
n = len(pts)
mx = sum(p[0] for p in pts) / n
my = sum(p[1] for p in pts) / n
sxx = sum((x - mx) ** 2 for x, _ in pts)
sxy = sum((x - mx) * (y - my) for x, y in pts)
b = sxy / sxx
a = my - b * mx
sse = sum((y - (a + b * x)) ** 2 for x, y in pts)
sst = sum((y - my) ** 2 for x, y in pts)
print('')
print('拟合 log(score) = %.3f + %.3f·log(Σ tbest/t)   R²=%.4f   n=%d' %
      (a, b, 1 - sse / sst, n))
base = sum(tbest[i] / mine[1][i] for i in range(NC))
print('我们若六点同时快 f 倍 ⇒ 拟合分（⚠️ f 大了会先把 tbest 自己刷掉，是上界）：')
for f in (1.0, 1.13, 1.5, 2.0, 3.0, 3.9, 5.3, 8.0):
    print('   f=%4.2fx  Σ=%.3f  拟合 score≈%.2f' % (f, base * f, math.exp(a + b * math.log(base * f))))

# 斜率 0.42 与"我们这三发的实际 score/Σ 比值"差得远 ⇒ 把候选模型一次摊开比 R²
S = []
for sc, t, _ in ok:
    s = sum(tbest[i] / t[i] for i in range(NC))
    S.append((s, sc))
sb = sum(x for x, _ in S) / n
so = sum(y for _, y in S) / n
sxx2 = sum((x - sb) ** 2 for x, _ in S)
k1 = sum((x - sb) * (y - so) for x, y in S) / sxx2
c1 = so - k1 * sb
r2lin = 1 - sum((y - (k1 * x + c1)) ** 2 for x, y in S) / sum((y - so) ** 2 for x, y in S)
k0 = sum(x * y for x, y in S) / sum(x * x for x, _ in S)      # 过原点 score = k·Σ
r2org = 1 - sum((y - k0 * x) ** 2 for x, y in S) / sum((y - so) ** 2 for x, y in S)
L = [(sum(math.log(t[i] / tbest[i]) for i in range(NC)), sc) for sc, t, _ in ok]
mxl = sum(p[0] for p in L) / n
sxx3 = sum((p[0] - mxl) ** 2 for p in L)
kb = sum((p[0] - mxl) * (p[1] - so) for p in L) / sxx3
cb = so - kb * mxl
r2log = 1 - sum((p[1] - (kb * p[0] + cb)) ** 2 for p in L) / sum((p[1] - so) ** 2 for p in L)
print('')
print('候选模型（n=%d，同一批行）：' % n)
print('  score = %.2f + %.2f·Σ(tbest/t)        R²=%.4f' % (c1, k1, r2lin))
print('  score = %.2f·Σ(tbest/t)  (过原点)      R²=%.4f   我们的 Σ=%.3f → 预分 %.1f' %
      (k0, r2org, base, k0 * base))
print('  score = %.2f %+.2f·Σ ln(t/tbest)      R²=%.4f' % (cb, kb, r2log))
print('  log-log 斜率 %.3f                    R²=%.4f' % (b, 1 - sse / sst))



