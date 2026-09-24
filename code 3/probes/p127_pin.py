#!/usr/bin/env python3
"""P127 判别器：**只作用在本地提交源** code 3/code/op_host/…（这一发要发货的字节）。
口径与预登记见 probes/p127_design.txt。

改的那一行（host 选档循环里的"改档"门，L304）：
    - if (bestCost == 0 || cost * 20ULL < bestCost * 19ULL) {   // 0.95 迟滞：小一档够便宜才换
    + if (bestCost == 0) {                                      // 单元数下界档：不换
`NB_CAND` 由大到小遍历 ⇒ 去掉改档支 = 首位可行候选胜出 = 永远取**最大可行 nb**。

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
OLD = "if (bestCost == 0 || cost * 20ULL < bestCost * 19ULL) {"
NEW = ("if (bestCost == 0) {   "
       "// P127 单元数下界档：候选由大到小，首位可行者直接胜出（口径见 probes/p127_design.txt）")
ANCHOR = "cost * 20ULL"
TOKENS = ("printf", "fflush", "fprintf", "std::cout", "cerr", "TODO", "FIXME", "#if 0", "调试")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def lines_with(src, tok):
    """与 `grep -c` 同口径：数**命中的行**，不是出现次数。"""
    return sum(1 for line in src.split("\n") if tok in line)


def report(tag, src):
    hit_new = lines_with(src, "if (bestCost == 0) {   // P127")
    hit_old = lines_with(src, OLD)
    hit_anchor = lines_with(src, ANCHOR)
    hit_bc0 = lines_with(src, "bestCost == 0")
    forbid = sum(1 for line in src.split("\n") if any(tok in line for tok in TOKENS))
    print("[%s] P127钉=%d  旧门=%d  %s=%d  bestCost==0=%d  禁用词=%d  sha=%s"
          % (tag, hit_new, hit_old, ANCHOR, hit_anchor, hit_bc0, forbid, sha(HOST)[:16]))
    # bestCost == 0 恒为 2 行：钉档后的 L304 + L316 的"无可行的候选"
    assert hit_bc0 == 2, "bestCost==0 行数应为 2（304/316），实得 %d" % hit_bc0
    assert forbid == 0, "禁用词命中 %d 行" % forbid
    return hit_new, hit_old, hit_anchor


mode = sys.argv[1] if len(sys.argv) > 1 else "check"

if mode == "apply":
    src = open(HOST, encoding="utf-8").read()
    assert sha(HOST)[:16] == SHA_PRISTINE or "P127 单元数下界档" in src, \
        "当前 host 既不是 pristine 也不是已钉档 ⇒ 中止"
    if "P127 单元数下界档" in src:
        report("apply: 已钉档", src)
        sys.exit(0)
    assert src.count(OLD) == 1, "锚点命中 %d 次（需 1）" % src.count(OLD)
    out = src.replace(OLD, NEW, 1)
    with open(HOST, "w", encoding="utf-8") as fh:
        fh.write(out)
    hit_new, hit_old, hit_anchor = report("apply", out)
    assert hit_new == 1 and hit_old == 0 and hit_anchor == 0, "钉档后仍残留 0.95 迟滞门"
    print("PINNED sha=%s" % sha(HOST))

elif mode == "revert":
    assert sha(PRISTINE)[:16] == SHA_PRISTINE, "备份本身不是 pristine，拒绝覆盖"
    shutil.copyfile(PRISTINE, HOST)
    hit_new, hit_old, hit_anchor = report("revert", open(HOST, encoding="utf-8").read())
    assert hit_new == 0 and hit_old == 1 and hit_anchor == 1, "还原后锚点没回来"
    assert sha(HOST)[:16] == SHA_PRISTINE, "还原后 sha 不等于 pristine"
    print("REVERTED sha=%s" % sha(HOST))

elif mode == "check":
    src = open(HOST, encoding="utf-8").read()
    pinned = "P127 单元数下界档" in src
    hit_new, hit_old, _ = report("check: %s" % ("PINNED" if pinned else "PRISTINE"), src)
    assert hit_new + hit_old == 1, "既不是钉档也不是 pristine（命中 %d/%d）" % (hit_new, hit_old)

else:
    sys.exit("未知模式：%s（apply|revert|check）" % mode)
