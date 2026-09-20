#!/usr/bin/env python3
# [PROBE P1] 把探针注入隔离树（~/ops_comp/probe1）。每条 patch 断言"锚点恰好出现 1 次"，
# 锚点不唯一/找不到 => 直接退出，绝不产生半拉子源码。提交树 ~/ops_comp/code1 永不动。
import os, sys, shutil, hashlib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))

KJ = os.path.join(ROOT, "op_kernel", "mhc_expand.cpp")
HS = os.path.join(ROOT, "op_host", "mhc_expand.cpp")
TN = os.path.join(ROOT, "npu_debug", "test_mhc_expand_npu.cpp")

with open(os.path.join(HERE, "fwd_probe.cpp"), encoding="utf-8") as f:
    PROBE = f.read().rstrip("\n")

OLD_LOOP = """        for (uint32_t k = k_begin; k < k_end; ++k) {
            const int64_t dst_off = static_cast<int64_t>(i) * tiling_.m * tiling_.D +
                                    static_cast<int64_t>(k) * tiling_.D + jt * tiling_.dTileLen;
            DataCopyPad(o_gm_[dst_off], x_local, cp);
        }
"""

PATCHES = [
    # 1) 前向输出 GM 声明长度放大：越界写只落在自己的分配里，且不超过声明（防污染整卡）
    (KJ, """            o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o),
                                  static_cast<int64_t>(s) * m * d);
""",
     """            o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X *>(o),
                                  static_cast<int64_t>(s) * m * d * 8 + 262144);  // [PROBE]
"""),
    # 2) 逐副本循环 -> 单发 blockCount=m（口径裁定本体）
    (KJ, OLD_LOOP, PROBE + "\n"),
    # 3) 强制单核：把"多核/派发"从受控实验里剔除，同时保证毒化范围只有 1 个核
    (HS, """        tiling->splitMode = split_mode;
        tiling->blockDim = block_dim;
""",
     """        block_dim = 1;   // [PROBE] 单核受控
        tiling->splitMode = split_mode;
        tiling->blockDim = block_dim;
"""),
    # 4) harness 输出缓冲同步放大（与 1) 的声明一致）
    (TN, "        if (aclrtMalloc(&dout, out_bytes, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) break;\n",
     "        if (aclrtMalloc(&dout, out_bytes * 8 + (1u << 20), ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS) break;  // [PROBE]\n"),
    # 5) 新 mode：probe，形状固定 S=64/D=256（dTileNum=1、ROW 模式），m 由 MHC_PROBE 选变体
    (TN, '    bool prof   = !strcmp(mode, "prof");\n',
     '    bool prof   = !strcmp(mode, "prof");\n    bool probe  = !strcmp(mode, "probe");\n'),
    (TN, """    if (quick) {
        RUN("fwd-fp16-small",  false, DT_FP16, 0, 64, 256, 2);
""",
     """    if (probe) {
        int pv = getenv("MHC_PROBE") ? atoi(getenv("MHC_PROBE")) : 8;
        RUN("probe-fp16", false, DT_FP16, 0, 64, 256, pv);
    }
    if (quick) {
        RUN("fwd-fp16-small",  false, DT_FP16, 0, 64, 256, 2);
"""),
]

def md5(b):
    return hashlib.md5(b).hexdigest()

TEXTS = {}
for p in (KJ, HS, TN):
    with open(p, encoding="utf-8") as f:
        TEXTS[p] = f.read()
ORIG = {p: md5(TEXTS[p].encode()) for p in TEXTS}

# 锚点唯一性先全部验完再落笔：任何一条不唯一 => 一个字节都不改
for p, old, new in PATCHES:
    n = TEXTS[p].count(old)
    if n != 1:
        print("ANCHOR FAIL (%d hits) in %s :: %r" % (n, os.path.basename(p), old[:70]))
        sys.exit(3)
print("### all %d anchors unique" % len(PATCHES))

for p, old, new in PATCHES:
    TEXTS[p] = TEXTS[p].replace(old, new, 1)

for p in (KJ, HS, TN):
    bak = p + ".bak_pre_probe"
    if not os.path.exists(bak):                      # 只备一次，别把已注入版当基线
        shutil.copy2(p, bak)
    with open(p, "w", encoding="utf-8") as f:
        f.write(TEXTS[p])

print("### patched (md5 pristine -> probe)")
for p in (KJ, HS, TN):
    print("  %-38s %s -> %s" % (os.path.relpath(p, ROOT), ORIG[p],
                                md5(TEXTS[p].encode())))
print("### [PROBE] markers: kernel=%d host=%d harness=%d" % (
    TEXTS[KJ].count("[PROBE]"), TEXTS[HS].count("[PROBE]"), TEXTS[TN].count("[PROBE]")))
print("### per-copy loop left in kernel (m>=5 / control path) = %d" %
      TEXTS[KJ].count("DataCopyPad(o_gm_[dst_off], x_local, cp);"))
