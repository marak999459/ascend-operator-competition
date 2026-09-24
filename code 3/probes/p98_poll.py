#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单发详情轮询：状态 + 六点时间 + 服务端 best_time。只读，不落任何凭据。
用法： python3 p98_poll.py <submission_id>
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

sid = sys.argv[1]
c = CANNJudgeClient()
assert c.load_session()
d = c._get_json('/api/submissions/%s' % sid) or {}
res = d.get('result') or []
print('id=%s status=%s score=%s create=%s' % (
    sid, d.get('status'), d.get('score'), d.get('create_time')))
for x in res:
    print('  tc=...%s time=%s best=%s msg=%s' % (
        str(x.get('testcase_id'))[-6:], x.get('time'), x.get('best_time'),
        str(x.get('msg'))[:40]))
if not res:
    print('  (result 为空)')
print('  其余键: %s' % sorted(k for k in d.keys()))
