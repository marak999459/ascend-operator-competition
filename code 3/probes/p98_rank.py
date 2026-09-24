#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""榜单里找我们自己那一行：按 user_id 前缀全量扫，报名次/分数/六点时间。只读。"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
c = CANNJudgeClient()
assert c.load_session()
ME = str(json.load(open(os.path.expanduser('~/.cannjudge/session.json'))).get('user_id'))
rows = []
for page in range(1, 12):
    d = c._get_json('/api/problems/%s/ranking' % PID, params={'page': page, 'size': 100}) or {}
    rr = d.get('rows') or d.get('data', {}).get('rows') or []
    if not rr:
        break
    rows.extend(rr)
    if len(rr) < 100:
        break
print('ME=%s rows=%d' % (ME, len(rows)))
hit = 0
for i, r in enumerate(rows):
    blob = json.dumps(r, ensure_ascii=False)
    if ME in blob:
        hit += 1
        print('RANK %-3s score=%-7s status=%-6s sub=%s t=%s' % (
            r.get('rank'), r.get('score'), r.get('status'),
            str(r.get('submission_id') or r.get('submission'))[-8:],
            [x.get('time') for x in (r.get('result') or [])]))
if not hit:
    print('榜单里没有我们 ⇒ 打印前两行的 user 字段口径')
    for r in rows[:2]:
        print(json.dumps({k: r[k] for k in r if 'user' in k or k == 'rank'}, ensure_ascii=False)[:300])
