#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读：把榜上我们自己那一行的**完整** submission_id 打出来（p98_rank.py 只截后 8 位）。
用法：ssh H 'python3 -' < probes/p128_fullsub.py   （⛔ 不落盘到提交源，纯读数）"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
c = CANNJudgeClient()
assert c.load_session()
ME = str(json.load(open(os.path.expanduser('~/.cannjudge/session.json'))).get('user_id'))
for page in range(1, 12):
    d = c._get_json('/api/problems/%s/ranking' % PID, params={'page': page, 'size': 100}) or {}
    rows = d.get('rows') or d.get('data', {}).get('rows') or []
    if not rows:
        break
    for r in rows:
        if ME in json.dumps(r, ensure_ascii=False):
            print('FULL rank=%s sub=%s score=%s' % (
                r.get('rank'), r.get('submission_id') or r.get('submission'), r.get('score')))
