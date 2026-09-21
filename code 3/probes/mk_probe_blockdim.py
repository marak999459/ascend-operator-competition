#!/usr/bin/env python3
"""**只给远端副本用**的 blockDim 探针：回答"小用例 kernel 那 ≈9 µs 固定开销是不是
`SetBlockDim(40)` 带来的"。

模式：
  clean    不打补丁（自报 md5，必须等于本地干净源）
  bd1      SetBlockDim(1)                    —— 固定开销的下界（只有 1 个块启动）
  bdauto   SetBlockDim(min(coreNum, units))  —— 候选改法：按单元数收缩启动块数
           units = rows * ceil(Q_N / nb)，rows = B*Q_S，与 CalcBlocking 同一口径

⛔ 本地提交源永不落探针（§15.14(e)）；跑完由 run_blockdim_probe.sh 还原干净构建。
"""
import hashlib
import sys

SRC = ("/home/fszqsn/ops_comp/ascend-operator-competition/code 3/code/"
       "op_host/sparse_flash_attention.cpp")
CLEAN_MD5 = "fe679da0b24342fd809912ae18fc8882"

ANCHOR = "    context->SetBlockDim(static_cast<uint32_t>(num_cores_aiv));"

BD1 = """    context->SetBlockDim(1u);   // SFA_PROBE_BLOCKDIM bd1"""

BDAUTO = """    {
        // SFA_PROBE_BLOCKDIM bdauto：单元数 = rows * ceil(Q_N/nb)，与 CalcBlocking 同源
        const uint64_t probeRows = static_cast<uint64_t>(B) * static_cast<uint64_t>(Q_S);
        const uint64_t probeUnits = probeRows * ((Q_N + nb - 1U) / nb);
        const uint32_t probeDim = (probeUnits < static_cast<uint64_t>(num_cores_aiv))
                                      ? static_cast<uint32_t>(probeUnits)
                                      : static_cast<uint32_t>(num_cores_aiv);
        context->SetBlockDim(probeDim == 0u ? 1u : probeDim);
    }"""


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "clean"
    src = open(SRC, encoding="utf-8").read()
    got = hashlib.md5(src.encode()).hexdigest()
    if got != CLEAN_MD5:
        print(f"!! 本地源 md5 已变（当前 {got}），CLEAN_MD5 要更新", file=sys.stderr)
    if mode == "clean":
        sys.stdout.write(src)
        return 0
    if src.count(ANCHOR) != 1:
        print(f"!! SetBlockDim 锚点命中 {src.count(ANCHOR)} 次，不是 1 次", file=sys.stderr)
        return 2
    rep = {"bd1": BD1, "bdauto": BDAUTO}.get(mode)
    if rep is None:
        print("用法: mk_probe_blockdim.py [clean|bd1|bdauto]", file=sys.stderr)
        return 2
    sys.stdout.write(src.replace(ANCHOR, rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
