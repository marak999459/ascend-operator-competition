#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P125 发次（带 429 真值捕获）—— 存在的理由：文档里"429 退避 ≥12 min"这条**已经被今天打脸**
（17:05 成功 ⇒ 18:04、18:23 两发仍 429），而 CLI 把响应体吞了只剩一句"HTTP 429" ⇒ 只能自己 POST
一次把 status/headers/body 打出来，才知道等的到底是"分钟级速率窗"还是"小时/天级配额"。
⛔ 不打印 payload（里面是提交源全文），只打印响应侧。
用法（远端）：python3 p125_submit_diag.py <project-dir> <tag>
"""
import json
import os
import sys
import time

sys.path.insert(0, '/mnt/workspace/gitCode/cann/cann-learning-hub/skills/cannjudge-submit')
from cannjudge_cli import CANNJudgeClient  # noqa: E402

PID = '6a7c22d6a52e0f540a8a098d'
DIR = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/sfa_real/p125q_code')
TAG = sys.argv[2] if len(sys.argv) > 2 else 'x'

c = CANNJudgeClient()
assert c.load_session(), "会话不可用"
pl = c.prepare_submission(PID, DIR)
sizes = {k: (len(v) if isinstance(v, str) else sum(len(str(x)) for x in v)) for k, v in pl.items()}
print("TAG=%s DIR=%s payload 字段=%s" % (TAG, DIR, sizes))
resp = c.session.post('%s/api/submissions/submit' % c.BASE_URL, json=pl, timeout=60)
print("HTTP %s" % resp.status_code)
if resp.status_code != 200:
    for k, v in resp.headers.items():
        if any(s in k.lower() for s in ('retry', 'rate', 'limit', 'date', 'server')):
            print("H %-16s %s" % (k, v))
    print("BODY %s" % resp.text[:600])
    sys.exit(2)
print("BODY %s" % resp.text[:300])
sid = (resp.json().get('data') or {}).get('submissionId')
print("SUBMISSION_ID=%s" % sid)
open(os.path.expanduser('~/sfa_real/p125_sub_%s.id' % TAG), 'w').write(str(sid))
for i in range(60):
    time.sleep(20)
    d = c.get_submission(sid)
    row = d.get('data') or d
    st = row.get('status')
    res = row.get('result') or row.get('testcases') or []
    ts = [x.get('time') for x in res if isinstance(x, dict)]
    print("[%2d] status=%-10s t=%s" % (i, st, ts))
    if st and st.lower() not in ('pending', 'running', 'queuing', 'waiting', 'processing'):
        print("FINAL %s" % json.dumps({k: v for k, v in row.items()
                                       if k in ('status', 'score', 'result', 'error', 'message')},
                                      ensure_ascii=False)[:2000])
        break
