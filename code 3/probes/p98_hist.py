#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""我们自己的提交史：每发六点时间 + 服务端 best_time + 反推分数。只读。

端点 /api/submissions/user/{uid} 可用（榜单只给 latest，这个给全history）。
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
N = int(os.environ.get('N', '18'))
c = CANNJudgeClient()
assert c.load_session()
sess = json.load(open(os.path.expanduser('~/.cannjudge/session.json')))
ME = str(sess.get('user_id'))
rows = c._get_json('/api/submissions/user/%s' % ME, params={'size': 60}) or []
mine = [r for r in rows if r.get('problem_id') == PID]
mine = mine[:N]
print('共 %d 发（取最近 %d），本題 %d 发' % (len(rows), N, len(mine)))
best = None
for r in mine:
    res = r.get('result') or []
    ts = [x.get('time') for x in res]
    bt = [x.get('best_time') for x in res]
    if bt and all(bt):
        best = bt
    sc = r.get('score')
    print('%s %-9s id=%s score=%-6s t=%s' % (
        str(r.get('create_time'))[5:19].replace('T', ' '), r.get('status'),
        str(r.get('_id'))[-6:], sc, ts))
if best:
    print('服务端 best_time = %s' % best)
    s = sum(b / t for b, t in zip(best, best)) if False else None
for r in mine[:1]:
    print('最新一发完整 result：')
    print(json.dumps(r.get('result'), ensure_ascii=False))
    print('其它字段：', {k: v for k, v in r.items() if k not in ('result', 'problem', 'user_id')})
