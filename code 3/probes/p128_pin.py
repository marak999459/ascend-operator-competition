#!/usr/bin/env python3
"""P128 判别器：**只作用在本地提交源** code 3/code/op_host/…（这一发要发货的字节）。
口径与预登记见 probes/p128_design.txt。

三行改动（都在 `CalcBlocking` 的选档循环）：
  1. L240 旁加 `uint64_t bestUnits = 0;`            —— 当前 incumbent 的单元数
  2. L304 那条迟滞门套上护栏                          —— incumbent 单元数 >= 2*coreNum 就不许降档
  3. 替换块里补 `bestUnits = units;`                  —— 与 bestCost/bestNb 同步更新

⇒ incumbent 单元数 < 80 的形状**逐位等于今天的 auto**，>= 80 的形状**逐位等于 P127**，
   中间那 6 格（qN>=16 ∧ rows<=64）是"停在中途档"的第三态。kernel 三枚不动。

模式：apply / revert / check。每步都带命中数断言（§7 第 2 条）。
"""
import hashlib
import os
import shutil
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
HOST = os.path.join(REPO, "code 3/code/op_host/sparse_flash_attention.cpp")
PRISTINE = os.path.join(REPO, "code 3/probes/backup/p127_pre_probe/sparse_flash_attention.cpp")
SHA_PRISTINE = "3a8f53050c8f15ae"

A1 = "    uint32_t bestNb = 0, bestBlk = 0, bestKs = 1U;"
B1 = A1 + "\n    uint64_t bestUnits = 0;   // P128：当前 incumbent（大 nb 侧）的单元数"
A2 = ("                if (bestCost == 0 || cost * 20ULL < bestCost * 19ULL) {")
B2 = ("                // P128：incumbent 单元数已到 2 波 ⇒ 保住大档，不再为 5 % 的模型差价降档。\n"
      "                if (bestCost == 0 || (bestUnits >= 2ULL * coreNum ? false : cost * 20ULL < bestCost * 19ULL)) {")
A3 = "                    bestKs = ks;\n"
B3 = "                    bestKs = ks;\n                    bestUnits = units;\n"

ANCHOR = "cost * 20ULL"
TOKENS = ("printf", "fflush", "fprintf", "std::cout", "cerr", "TODO", "FIXME", "#if 0", "调试")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def lines_with(src, tok):
    """与 `grep -c` 同口径：数**命中的行**，不是出现次数。"""
    return sum(1 for line in src.split("\n") if tok in line)


def report(tag, src):
    p128 = lines_with(src, "P128")
    bu = lines_with(src, "bestUnits")
    old = lines_with(src, A2)
    anchor = lines_with(src, ANCHOR)
    bc0 = lines_with(src, "bestCost == 0")
    oldtab = lines_with(src, "{32, 16, 8, 4, 2, 1}")
    forbid = sum(1 for line in src.split("\n") if any(tok in line for tok in TOKENS))
    print("[%s] P128=%d bestUnits=%d 旧门=%d %s=%d bestCost==0=%d OLDTAB=%d 禁用词=%d sha=%s"
          % (tag, p128, bu, old, ANCHOR, anchor, bc0, oldtab, forbid, sha(HOST)[:16]))
    assert bc0 == 2, "bestCost==0 应为 2 行（304/316），实得 %d" % bc0
    assert oldtab == 1, "NB_CAND 候选表被动了（%d）" % oldtab
    assert forbid == 0, "禁用词命中 %d 行" % forbid
    return p128, bu, old


mode = sys.argv[1] if len(sys.argv) > 1 else "check"

if mode == "apply":
    src = open(HOST, encoding="utf-8").read()
    assert sha(HOST)[:16] == SHA_PRISTINE or "P128" in src, \
        "当前 host 既不是 pristine 也不是已钉档 ⇒ 中止"
    if "P128" in src:
        report("apply: 已钉档", src)
        sys.exit(0)
    for a in (A1, A2, A3):
        assert src.count(a) == 1, "锚点命中 %d 次（需 1）：%s" % (src.count(a), a.strip()[:40])
    out = src.replace(A1, B1, 1).replace(A2, B2, 1).replace(A3, B3, 1)
    with open(HOST, "w", encoding="utf-8") as fh:
        fh.write(out)
    p128, bu, old = report("apply", out)
    assert p128 == 2 and bu == 3 and old == 0, "钉档后计数不对（需 P128=2 bestUnits=3 旧门=0）"
    print("PINNED sha=%s" % sha(HOST))

elif mode == "revert":
    assert sha(PRISTINE)[:16] == SHA_PRISTINE, "备份本身不是 pristine，拒绝覆盖"
    shutil.copyfile(PRISTINE, HOST)
    p128, bu, old = report("revert", open(HOST, encoding="utf-8").read())
    assert p128 == 0 and bu == 0 and old == 1, "还原后锚点没回来"
    assert sha(HOST)[:16] == SHA_PRISTINE, "还原后 sha 不等于 pristine"
    print("REVERTED sha=%s" % sha(HOST))

elif mode == "check":
    src = open(HOST, encoding="utf-8").read()
    p128, bu, old = report("check: %s" % ("PINNED" if "P128" in src else "PRISTINE"), src)
    assert (p128, bu, old) in ((0, 0, 1), (2, 3, 0)), \
        "既不是 pristine 也不是钉档（P128=%d bestUnits=%d 旧门=%d）" % (p128, bu, old)

else:
    sys.exit("未知模式：%s（apply|revert|check）" % mode)
