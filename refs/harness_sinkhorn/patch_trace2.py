#!/usr/bin/env python3
# 在 kernel 的每个 SyncAll 前插入 workspace 打点；host 侧读回并打印。
# workspace 布局: [0, 40960) 用于调试，每核 64 个 int32 槽
#   slot[0] = 该核到达的最后一个打点编号
#   slot[1] = 该核 SyncAll 累计次数
import re

KP = "/home/developer/mhc_build/op_kernel/mhc_sinkhorn.cpp"
s = open(KP).read()

# ---- 1) Init 接收 workspace，设置 dbgGm_ ----
s = s.replace(
    "    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,\n"
    "                                const MhcSinkhornTilingData &tiling) {",
    "    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights, GM_ADDR workspace,\n"
    "                                const MhcSinkhornTilingData &tiling) {")

s = s.replace(
    "        pipe_.InitBuffer(mat_buf_, N_MAX * N_MAX * sizeof(DT_LOGITS));",
    "        dbgGm_ = reinterpret_cast<__gm__ int32_t *>(workspace);\n"
    "        pipe_.InitBuffer(mat_buf_, N_MAX * N_MAX * sizeof(DT_LOGITS));")

# ---- 2) 加 Mark() 与成员 ----
s = s.replace("private:\n",
"""private:
    // ===== 调试打点（写入 workspace 前 40960 字节）=====
    __aicore__ inline void Mark(int32_t site) {
        const uint32_t blk = GetBlockIdx();
        const int32_t base = (int32_t)(blk * 64u);
        dbgGm_[base + 0] = site;                 // 最后一个打点
        dbgGm_[base + 1] = (int32_t)(dbgCounter_ + 1);  // SyncAll 累计次数
        dbgCounter_++;
    }
    __gm__ int32_t *dbgGm_ = nullptr;
    uint32_t dbgCounter_ = 0;
""", 1)

# ---- 3) 每个 SyncAll 前打点 ----
site = [0]
def repl(m):
    site[0] += 1
    return "Mark(%d); AscendC::SyncAll();" % site[0]
s = re.sub(r"AscendC::SyncAll\(\);", repl, s)

# ---- 4) 核函数传 workspace ----
s = s.replace("op.Init(logits, weights, tiling_data);",
              "op.Init(logits, weights, workspace, tiling_data);")

open(KP, "w").write(s)
print("kernel patched: SyncAll sites =", site[0])

# ---- 5) harness: workspace 至少 40960 字节，并读回打印 ----
HP = "/home/developer/mhc_test/harness/test_mhc_sinkhorn.cpp"
h = open(HP).read()
h = h.replace(
    "    void *ws = nullptr;\n"
    "    if (wsSize > 0) {\n"
    "        CHK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), \"aclrtMalloc ws\");\n"
    "    }",
    "    if (wsSize < 40960) wsSize = 40960;   // 调试需要\n"
    "    void *ws = nullptr;\n"
    "    CHK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), \"aclrtMalloc ws\");")

h = h.replace(
    '    printf("[OK] 同步成功，kernel 执行完成\\n");',
    '''    printf("[OK] 同步成功，kernel 执行完成\\n");
    {
        int32_t dbg[40960/4];
        aclrtMemcpy(dbg, sizeof(dbg), ws, sizeof(dbg), ACL_MEMCPY_DEVICE_TO_HOST);
        printf("[TRACE] 每核 (最后打点, SyncAll次数):\\n");
        for (int b = 0; b < 40; ++b) {
            printf("  blk=%-3d last=%-4d syncs=%-3d\\n", b, dbg[b*64+0], dbg[b*64+1]);
        }
    }''')
open(HP, "w").write(h)
print("harness patched")
