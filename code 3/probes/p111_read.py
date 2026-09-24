#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最近 N 发的六点时间 + 分数，只打印紧凑一行一发（不 dump 源码，那是 p98_detail 的副作用）。
用法： python3 p111_read.py [发数]
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = '6a7c22d6a52e0f540a8a098d'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
c = CANNJudgeClient()
assert c.load_session()
sess = json.load(open(os.path.expanduser('~/.cannjudge/session.json')))
ME = str(sess.get('user_id'))
lst = c._get_json('/api/submissions/user/%s' % ME, params={'size': 60}) or []
mine = [r for r in lst if r.get('problem_id') == PID][:N]
KEYS = ('score', 'total_score', 'final_score', 'aggregate_score', 'rank', 'theory_score')
for r in mine:
    d = c._get_json('/api/submissions/%s' % r['_id']) or {}
    res = d.get('result') or []
    ts = [x.get('time') for x in res]
    bt = [x.get('best_time') for x in res]
    info = {k: d.get(k) for k in KEYS if d.get(k) is not None}
    if not info:
        info = {k: v for k, v in d.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}
    print('%s id=%s t=%s best=%s info=%s' % (
        r.get('created_at') or r.get('create_time'), str(r['_id'])[-7:], ts, bt, info))
