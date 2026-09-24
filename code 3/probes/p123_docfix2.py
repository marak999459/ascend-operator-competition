#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P123 收尾第二批（行文本在 p123_rows.txt，这里只做锚点替换，避免引号地狱）。"""
import io
import re
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
DOC = REPO + "/code3.md"
ROWS = [x for x in io.open(REPO + "/code 3/probes/p123_rows.txt", encoding="utf-8").read().split("\n") if x.strip()]
assert len(ROWS) == 6, len(ROWS)
row28, row119, row120, row121, row123, row124 = ROWS

s = io.open(DOC, encoding="utf-8").read()
n0 = len(s.encode("utf-8"))
done = []


def swap(pat, new, name):
    global s
    m = re.search(pat, s, re.M)
    if not m:
        sys.exit(f"{name}: 锚点 {pat[:40]} 找不到 ⇒ 全文件不改")
    if len(re.findall(pat, s, re.M)) != 1:
        sys.exit(f"{name}: 锚点命中多次 ⇒ 全文件不改")
    s = s[:m.start()] + new + s[m.end():]
    done.append(name)


# §5：在 #27 行后插一行（该行以 `#15.101` | 结尾，唯一）
m = re.search(r"^\| \*\*27\*\* \|.*$", s, re.M)
assert m, "§5 #27 找不到"
s = s[:m.end()] + "\n" + row28 + s[m.end():]
done.append("§5 #28 插入")

swap(r"^\| — \| ~~\*\*P119 = PV 段 ×2.*$", row119, "§6 P119 压回")
swap(r"^\| — \| ~~\*\*P120 = 给.+$", row120, "§6 P120 压回")
swap(r"^\| — \| ~~\*\*P121 = 标量下标扫描计量器\*\*.*$", row121, "§6 P121 压回")
swap(r"^\| \*\*P0\*\* \| \*\*P123 = 依赖链计量器\*\*.*$", row123 + "\n" + row124, "§6 P123 结掉 + P124 挂上")

io.open(DOC, "w", encoding="utf-8").write(s)
print("已改:", "  ".join(done))
print(f"本批净变化 {len(s.encode('utf-8')) - n0} B（{n0} → {len(s.encode('utf-8'))}，行数 {s.count(chr(10)) + 1}）")
