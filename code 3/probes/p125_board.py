#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""只读：把榜上我们自己那一行的**完整 JSON**打出来。
存在的理由：① p98_rank.py 只截后 8 位、也没有时刻字段，而 429 的真实窗口只能从"上一次成功提交
是几点"反推（⛔ 不能再靠猜 sleep 多少分钟空转）；② 榜单名次要从行里现读，不能沿用文档里的旧值。
用法：ssh H 'python3 -' < probes/p125_board.py
"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = os.environ.get('PID', '6a7c22d6a52e0f540a8a098d')
c = CANNJudgeClient()
assert c.load_session()
ME = str(json.load(open(os.path.expanduser('~/.cannjudge/session.json'))).get('user_id'))
print("可用方法：%s" % [m for m in dir(c) if not m.startswith('_')])
for page in range(1, 12):
    d = c._get_json('/api/problems/%s/ranking' % PID, params={'page': page, 'size': 100}) or {}
    rr = d.get('rows') or d.get('data', {}).get('rows') or []
    if not rr:
        break
    for r in rr:
        if ME in json.dumps(r, ensure_ascii=False):
            print("=== 我们的整行 ===")
            print(json.dumps(r, ensure_ascii=False, indent=1))
    if len(rr) < 100:
        break
