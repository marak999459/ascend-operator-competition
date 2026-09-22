#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读榜单当前态：自己那一行（score / 六点时间 / 名次）+ 榜首 + 前 5。只读接口。

口径来自 §15.42(b)：唯一可用的端点是 `/api/problems/{pid}/ranking`，
CLI 自带的 `rank` 子命令那条路径服务端 404，别用。
"""
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
c = CANNJudgeClient()
if not c.load_session():
    print('NO_SESSION'); sys.exit(1)

rows = []
for page in range(1, 8):
    try:
        d = c._get_json('/api/problems/%s/ranking' % PID, params={'page': page, 'size': 80})
    except Exception as e:  # noqa: BLE001
        print('PAGE_ERR', repr(e)[:120]); break
    rr = (d or {}).get('rows') or (d or {}).get('data', {}).get('rows') or []
    if not rr:
        break
    rows.extend(rr)
    if len(rr) < 80:
        break

try:                        # /api/users/me 服务端 404，身份一律靠 WANT 时间反查
    me = (c._get_json('/api/users/me') or {})
except Exception:  # noqa: BLE001
    me = {}
mud = me.get('user') or me
mid = str(mud.get('id') or mud.get('_id') or '')
mail = str(mud.get('email') or mud.get('username') or '')
# 刚才那发 P38 的六点时间（submission 6ab251170304f72a566142ef）⇒ 用它反查我们那一行，
# 比按 user_id 匹配稳（榜单行的身份字段口径没查过）。
WANT = [float(x) for x in os.environ.get(
    'WANT', '7.68,8.34,11.48,13.52,9.98,14.42').split(',')]


def times(r):
    return [t.get('time') for t in (r.get('result') or [])]


def show(tag, r, pos):
    print('%-8s #%s %-24s score=%-7s status=%-6s t=%s' %
          (tag, pos, str(r.get('user_id') or r.get('user') or '')[:24],
           r.get('score'), r.get('status'), times(r)))


scored = [r for r in rows if r.get('score') not in (None, 0, '0')]
print('ROWS %d  SCORED %d  me=%s %s' % (len(rows), len(scored), mid[:8], mail[:6]))
for i, r in enumerate(scored[:5]):
    show('TOP', r, i + 1)


def near(a, b, tol=0.02):
    return len(a) == len(b) and all(abs(x - y) <= tol * max(1.0, abs(y))
                                    for x, y in zip(a, b))


hit = False
for i, r in enumerate(scored):
    if near(times(r), WANT) or (mid and str(r.get('user_id')) == mid):
        show('MINE', r, i + 1); hit = True
if not hit:
    print('MINE  (六点时间与 WANT 都不吻合 ⇒ 把这份输出的 TOP 行原样回来看)')
