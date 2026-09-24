#!/usr/bin/env python3
# P126 判别器的补丁生成器：**只作用在本地提交源** code 3/code/op_host/…（这一发是要发货的字节，
# 不是探针 only 的远端旋钮）。口径见 probes/p126_design.txt。
#   apply   —— pristine -> NB_CAND 钉成 {1}
#   revert  —— 从 probes/backup/p126_pre_probe/ 拉回 pristine
#   check   —— 只报命中数与 sha，不动文件
# 每一步都带命中数断言（§7 第 2 条：远端/本地旋钮必须有命中数当门）。
import hashlib
import os
import shutil
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition"
HOST = os.path.join(REPO, "code 3/code/op_host/sparse_flash_attention.cpp")
PRISTINE = os.path.join(REPO, "code 3/probes/backup/p126_pre_probe/sparse_flash_attention.cpp")
SHA_PRISTINE = "3a8f53050c8f15ae"
OLD = "constexpr uint32_t NB_CAND[] = {32, 16, 8, 4, 2, 1};"
TABLE_BODY = "{32, 16, 8, 4, 2, 1}"
NEW = ("constexpr uint32_t NB_CAND[] = {1};   "
       "// P126 判别器：头分块候选钉成单档，其余口径见 probes/p126_design.txt")
TOKENS = ("printf", "fflush", "fprintf", "std::cout", "cerr", "TODO", "FIXME", "#if 0", "调试")


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def lines_with(src, tok):
    """与 `grep -c` 同口径：数**命中的行**，不是出现次数（L229 一行里有两次 NB_CAND）。"""
    return sum(1 for line in src.split("\n") if tok in line)


def counts(src):
    return (lines_with(src, "NB_CAND[] = {1};"), lines_with(src, "NB_CAND"),
            lines_with(src, TABLE_BODY))


def report(tag, src):
    hit_pin, hit_nb, hit_full = counts(src)
    forbid = sum(1 for line in src.split("\n") if any(tok in line for tok in TOKENS))
    print("[%s] NB_CAND{1}=%d  NB_CAND=%d  旧表体=%d  禁用词=%d  sha=%s"
          % (tag, hit_pin, hit_nb, hit_full, forbid, sha(HOST)[:16]))
    assert hit_nb == 4, "NB_CAND 行数应为 4（36/229/241/244），实得 %d" % hit_nb
    assert forbid == 0, "禁用词命中 %d 行" % forbid
    return hit_pin, hit_full


mode = sys.argv[1] if len(sys.argv) > 1 else "check"

if mode == "apply":
    src = open(HOST, encoding="utf-8").read()
    assert sha(HOST)[:16] == SHA_PRISTINE or "NB_CAND[] = {1};" in src, \
        "当前 host 既不是 pristine 也不是已钉档 ⇒ 中止"
    if "NB_CAND[] = {1};" in src:
        report("apply: 已钉档", src)
        sys.exit(0)
    assert src.count(OLD) == 1, "锚点命中 %d 次（需 1）" % src.count(OLD)
    out = src.replace(OLD, NEW, 1)
    with open(HOST, "w", encoding="utf-8") as fh:
        fh.write(out)
    hit_pin, hit_full = report("apply", out)
    assert hit_pin == 1 and hit_full == 0, "钉档后仍残留旧表"
    print("PINNED sha=%s" % sha(HOST))

elif mode == "revert":
    assert sha(PRISTINE)[:16] == SHA_PRISTINE, "备份本身不是 pristine，拒绝覆盖"
    shutil.copyfile(PRISTINE, HOST)
    report("revert", open(HOST, encoding="utf-8").read())
    assert sha(HOST)[:16] == SHA_PRISTINE, "还原后 sha 不等于 pristine"
    print("REVERTED sha=%s" % sha(HOST))

elif mode == "check":
    src = open(HOST, encoding="utf-8").read()
    pinned = "NB_CAND[] = {1};" in src
    report("check: %s" % ("PINNED" if pinned else "PRISTINE"), src)

else:
    sys.exit("未知模式：%s（apply|revert|check）" % mode)
