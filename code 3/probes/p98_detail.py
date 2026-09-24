#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐发提交详情：六点时间 + 服务端 best_time。只读。
用法： python3 p98_detail.py [发数]
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
c = CANNJudgeClient()
assert c.load_session()
sess = json.load(open(os.path.expanduser('~/.cannjudge/session.json')))
ME = str(sess.get('user_id'))
lst = c._get_json('/api/submissions/user/%s' % ME, params={'size': 60}) or []
mine = [r for r in lst if r.get('problem_id') == PID][:N]
print('提交史（新→旧），共 %d 发' % len(mine))
hdr = None
for r in mine:
    d = c._get_json('/api/submissions/%s' % r['_id']) or {}
    res = d.get('result') or []
    ts = [x.get('time') for x in res]
    bt = [x.get('best_time') for x in res]
    tcids = [str(x.get('testcase_id'))[-2:] for x in res]
    if hdr is None and tcids:
        hdr = tcids
        print('tc 尾号: %s' % hdr)
    print('%s %-11s id=%-7s sc=%-6s t=%-46s best=%s  extra=%s' % (
        str(d.get('create_time'))[5:19].replace('T', ' '), d.get('status'),
        str(r.get('_id'))[-7:], d.get('score'), ts, bt,
        {k: v for k, v in d.items() if k not in
         ('result', 'problem', 'user_id', 'contest', '_id', 'ID', 'problem_id',
          'create_time', 'status', 'score', 'valid')}))
cont = c._get_json('/api/contests/%s' % '6a7bf087a52e0f540a88e167') or {}
print('CONTEST keys:', {k: cont.get(k) for k in ('name', 'start_time', 'end_time', 'ongoing',
                                                 'ranking_submission_mode', 'submission_limit',
                                                 'daily_submission_limit', 'total_submission_limit')})
