#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""题 3 榜上当前态：我们自己那一行 + 榜首 + 前 5，另附全量分数列便于对账。只读接口。

唯一可用端点 = /api/problems/{pid}/ranking（CLI 自带的 latest 子命令服务端 404）。
身份匹配用 session.json 里的 user_id，不靠时间反查。
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
c = CANNJudgeClient()
if not c.load_session():
    print('NO_SESSION')
    sys.exit(1)
sess = json.load(open(os.path.expanduser('~/.cannjudge/session.json')))
me = str(sess.get('user_id') or '')

rows = []
for page in range(1, 9):
    try:
        d = c._get_json('/api/problems/%s/ranking' % PID, params={'page': page, 'size': 100})
    except Exception as e:  # noqa: BLE001
        print('PAGE_ERR', repr(e)[:120])
        break
    rr = (d or {}).get('rows') or (d or {}).get('data', {}).get('rows') or []
    if not rr:
        break
    rows.extend(rr)
    if len(rr) < 100:
        break


def times(r):
    return [t.get('time') for t in (r.get('result') or [])]


print('ROWS %d  me=%s' % (len(rows), me[:10]))
for i, r in enumerate(rows):
    uid = str(r.get('user_id') or r.get('user') or r.get('_id') or '')
    tag = 'ME ' if (me and me in uid or uid and me[:8] == uid[:8]) else '   '
    if tag == 'ME ' or i < 5:
        print('%s#%-3s %-22s score=%-7s status=%-6s t=%s' %
              (tag, i + 1, uid[:22], r.get('score'), r.get('status'), times(r)))
best = [t for t in (times(rows[0]) if rows else [])]
print('TOP6 scores:', [r.get('score') for r in rows[:6]])
print('row keys sample:', sorted(rows[0].keys()) if rows else [])
