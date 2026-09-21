#!/usr/bin/env python3
"""**只给远端副本用**的"固定开销解剖"探针（code3.md §15.17）。

背景：批量口径（连发 20 次只 sync 一次）下，r1_min 这种只有 512 个输出元素的用例
仍要 **7.6 µs**，而 `SetBlockDim(40→1)` 只省 1.1 µs ⇒ 剩下的 6~7 µs 是每次 launch
的固定开销。比赛平台榜单是 µs 量级 ⇒ 这块可能就是小用例分数的大头，先量清构成。

模式（都叠加在 bdauto 之上，bdauto 已实测"处处不劣"，是准备保留的改动）：
  base     只有 bdauto                    —— 基准（应复现 7.6 µs）
  bare     入口立刻 return                 —— 下界：host 调用 + 硬件启动
  tilread  读 tiling + 碰 6 个字段后 return —— 减 bare = tiling 的 GM 读代价
  nopro    读 tiling + 完整 Init，循环空转  —— 减 tilread = 14 次 InitBuffer + SetGm

⚠️ 提前 return 的用例输出是脏的，diff 必然失败 —— 这一轮只看时间，不看对拍。
⚠️ 三个模式的"一定走到/一定不走"都靠数据依赖条件（`query != nullptr` 恒真、
   `probeAcc != 0xFFFFFFFF` 恒真、`u < probeCap` 恒假）：编译器无法折叠，
   既保证运行时行为确定，又保证想量的那次读不会被 DCE 掉。
⛔ 探针只写远端副本；本地提交源永不落探针。
"""
import hashlib
import sys

REPO = "/home/fszqsn/ops_comp/ascend-operator-competition/code 3/code"
HOST = REPO + "/op_host/sparse_flash_attention.cpp"
KER = REPO + "/op_kernel/sparse_flash_attention.cpp"
CLEAN = {"host": "fe679da0b24342fd809912ae18fc8882", "kernel": "3d366c529adf6d005bd73fab023df95d"}

BD_ANCHOR = "    context->SetBlockDim(static_cast<uint32_t>(num_cores_aiv));"
BD_PATCH = """    {
        // SFA_PROBE_BLOCKDIM bdauto：块数 = 单元数（与 CalcBlocking 同一口径），封顶 coreNum
        const uint64_t probeRows = static_cast<uint64_t>(B) * static_cast<uint64_t>(Q_S);
        const uint64_t probeUnits = probeRows * ((Q_N + nb - 1U) / nb);
        const uint32_t probeDim = (probeUnits < static_cast<uint64_t>(num_cores_aiv))
                                      ? static_cast<uint32_t>(probeUnits)
                                      : static_cast<uint32_t>(num_cores_aiv);
        context->SetBlockDim(probeDim == 0u ? 1u : probeDim);
    }"""

ENTRY = """    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);
    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingData, tiling_data, tiling);"""

PROBE = {
    "bare": """    if (query != nullptr) { return; }   // SFA_PROBE_FLOOR bare：入口即退
""" + ENTRY,
    "tilread": ENTRY + """
    const uint32_t probeAcc = tiling_data.Q_D + tiling_data.Q_N + tiling_data.B +
                              tiling_data.nb + tiling_data.n_blk + tiling_data.sparse_count;
    if (probeAcc != 0xFFFFFFFFu) { return; }   // SFA_PROBE_FLOOR tilread：读完 tiling 就退
""",
    "nopro": ENTRY,
}

PROC_ANCHOR = """        for (uint32_t u = unitBegin_; u < unitEnd_; u += unitStep_) {"""
PROC_PATCH = """        const uint32_t probeCap = (unitEnd_ == 0xFFFFFFFFu) ? unitEnd_ : unitBegin_;
        for (uint32_t u = unitBegin_; u < probeCap; u += unitStep_) {   // SFA_PROBE_FLOOR nopro"""


def patch_host(src: str) -> str:
    if BD_ANCHOR not in src:
        return src          # bdauto 已落进提交源（P14），无需再打补丁
    return src.replace(BD_ANCHOR, BD_PATCH)


def patch_kernel(src: str, mode: str) -> str:
    if mode == "base":
        return src
    if src.count(ENTRY) != 1:
        raise RuntimeError("kernel 入口锚点命中异常")
    out = src.replace(ENTRY, PROBE[mode])
    if mode == "nopro":
        if out.count(PROC_ANCHOR) != 1:
            raise RuntimeError("kernel Process 锚点命中异常")
        out = out.replace(PROC_ANCHOR, PROC_PATCH)
    return out


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode not in ("base", "bare", "tilread", "nopro"):
        print("用法: mk_probe_floor.py [base|bare|tilread|nopro]", file=sys.stderr)
        return 2
    hs, ks = open(HOST, encoding="utf-8").read(), open(KER, encoding="utf-8").read()
    for name, txt in (("host", hs), ("kernel", ks)):
        if hashlib.md5(txt.encode()).hexdigest() != CLEAN[name]:
            print(f"!! {name} 本地源 md5 已变（当前 "
                  f"{hashlib.md5(txt.encode()).hexdigest()}），CLEAN 要更新", file=sys.stderr)
    blob = ["===HOST===", patch_host(hs), "===KERNEL==="]
    blob.append(patch_kernel(ks, mode))
    sys.stdout.write("\n".join(blob))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
