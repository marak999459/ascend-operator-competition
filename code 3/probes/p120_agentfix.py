#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P120：AGENT.MD 第三题那一行整行换掉（L0 只放当前口径、一行）。
   新行文本放 `p120_agent_line.txt`（heredoc 写入，避免中文引号与 Python 字符串定界符打架）。
   锚点 = 行首，必须命中 1 行；`--check` 只数不改。"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "..", "..", "AGENT.MD")
PAYLOAD = os.path.join(HERE, "p120_agent_line.txt")
ANCHOR = "| 第三题 `sparse_flash_attention` |"

lines = io.open(DOC, encoding="utf-8").read().split("\n")
new = io.open(PAYLOAD, encoding="utf-8").read().rstrip("\n")
assert new.startswith(ANCHOR), "载荷行首与锚点不符"
hits = [i for i, l in enumerate(lines) if l.startswith(ANCHOR)]
if len(hits) != 1:
    print(f"锚点失败：命中 {len(hits)} 行")
    sys.exit(1)
i = hits[0]
if lines[i] == new:
    print("已是新行 ⇒ 不动")
    sys.exit(0)
if "--check" in sys.argv:
    print(f"锚点命中 1 行（第 {i + 1} 行），未改文件")
    sys.exit(0)
old = lines[i]
lines[i] = new
io.open(DOC, "w", encoding="utf-8").write("\n".join(lines))
print(f"第 {i + 1} 行整行替换  净增长（字符数，UTF-8 下中文 3 B）={len(new) - len(old)}")
