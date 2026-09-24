#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P123 收尾第三批 = 付费：§5 四行与 §1.1/§6 三处重复，压回一份（读数与锚点都留，细节交 LOG）。"""
import io
import re
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
DOC = REPO + "/code3.md"
rows = [x for x in io.open(REPO + "/code 3/probes/p123_rows2.txt", encoding="utf-8").read().split("\n") if x.strip()]
assert len(rows) == 4, len(rows)
s = io.open(DOC, encoding="utf-8").read()
n0 = len(s.encode("utf-8"))
done = []

for row in rows:
    num = re.match(r"\| \*\*(\d+)\*\*", row).group(1)
    pat = r"^\| \*\*" + num + r"\*\* \|.*$"
    m = re.search(pat, s, re.M)
    if not m:
        sys.exit(f"§5 #{num} 找不到 ⇒ 全文件不改")
    s = s[:m.start()] + row + s[m.end():]
    done.append(f"#{num}")

# 那句"9/24~9/25 那六发…"是 P121 时代留下的引导语，现在它引导的清单已经长过它了 ⇒ 删（它下面直接是 bullets）
frag = re.search(r"（后三发是否证）：\n", s)
if frag:
    s = s[:frag.start()] + "：\n" + s[frag.end():]
    done.append("删旧引导语")

io.open(DOC, "w", encoding="utf-8").write(s)
print("已改:", " ".join(done))
print(f"本批净变化 {len(s.encode('utf-8')) - n0} B（{len(s.encode('utf-8'))} B，行数 {s.count(chr(10)) + 1}）")
