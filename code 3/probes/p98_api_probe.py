#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""探平台只读接口：能不能拿到六点用例的形状 / 我们自己的提交历史。全部 GET，不写任何东西。"""
import json
import os
import sys

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = '6a7c22d6a52e0f540a8a098d'
TC = ['6a8d6a7082cffa8f16133d1e', '6a8d6a7082cffa8f16133d22', '6a8d6a7082cffa8f16133d26',
      '6a8d6a7082cffa8f16133d2a', '6a8d6a7082cffa8f16133d2e', '6a8d6a7082cffa8f16133d32']
SUB = '6ab35f420304f72a56dc215f'
c = CANNJudgeClient()
assert c.load_session()
sess = json.load(open(os.path.expanduser('~/.cannjudge/session.json')))
ME = str(sess.get('user_id'))


def try_get(tag, path, params=None):
    try:
        d = c._get_json(path, params=params)
        s = json.dumps(d, ensure_ascii=False)
        print('[OK] %-34s %s' % (tag, s[:420]))
        return d
    except Exception as e:  # noqa: BLE001
        print('[--] %-34s %s' % (tag, repr(e)[:90]))
        return None


print('=== 1) 单用例详情候选端点 ===')
try_get('testcase/%s' % TC[0], '/api/testcases/%s' % TC[0])
try_get('testcases/%s' % TC[0], '/api/testcases/%s' % TC[0])
try_get('problems/pid/testcases', '/api/problems/%s/testcases' % PID)
try_get('problem/testcases?problem_id', '/api/testcases', params={'problem_id': PID, 'size': 50})
try_get('problems/pid', '/api/problems/%s' % PID)

print('=== 2) 提交详情 / 我的提交列表 ===')
try_get('submissions/%s' % SUB, '/api/submissions/%s' % SUB)
try_get('submissions/user/me', '/api/submissions/user/%s' % ME, params={'size': 30})
try_get('submissions?problem_id', '/api/submissions', params={'problem_id': PID, 'size': 20})
try_get('problems/pid/submissions', '/api/problems/%s/submissions' % PID, params={'size': 20})
