#!/usr/bin/env python3
# 在 kernel 的每个 SyncAll 前插入 GM 打点，定位死锁位置
import re, sys

p = "/home/developer/mhc_build/op_kernel/mhc_sinkhorn.cpp"
s = open(p).read()

# 1) 类里加 debug 相关成员与辅助函数
helper = '''
    // ===== 调试打点 =====
    __aicore__ inline void Mark(int32_t site) {
        if (dbgGm_ == nullptr) return;
        const uint32_t blk = GetBlockIdx();
        // 每个核独立一段：blk*64 + 序号，记录 (site, counter)
        const int32_t idx = (int32_t)(blk * 64u + dbgCounter_);
        if (dbgCounter_ < 64u) {
            dbgGm_[idx] = site;
        }
        dbgCounter_++;
    }
    __global__ int32_t *dbgGm_ = nullptr;
    uint32_t dbgCounter_ = 0;
'''

# 2) Init 里接受 workspace 并设置 dbgGm_
s = s.replace(
    "    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,\n"
    "                                const MhcSinkhornTilingData &tiling) {",
    "    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights, GM_ADDR workspace,\n"
    "                                const MhcSinkhornTilingData &tiling) {\n"
    "        dbgGm_ = (workspace != nullptr) ? reinterpret_cast<__global__ int32_t *>(workspace) : nullptr;")

# 3) 插入 helper（放在 private: 之后）
s = s.replace("private:\n", "private:\n" + helper, 1)

# 4) 给每个 SyncAll 打点：编号递增
site = [0]
def repl(m):
    site[0] += 1
    return "Mark(%d); AscendC::SyncAll();" % site[0]
s = re.sub(r"AscendC::SyncAll\(\);", repl, s)

# 5) 核函数里把 workspace 传进 Init
s = s.replace("op.Init(logits, weights, tiling_data);",
              "op.Init(logits, weights, workspace, tiling_data);")

# 6) 结束时每个核输出自己的轨迹（只有 blk 0 输出全部）
s = s.replace(
"""        // 兜底栅栏：确保所有核都走到同一点再退出
        AscendC::SyncAll();""",
"""        AscendC::SyncAll();
        if (GetBlockIdx() == 0 && dbgGm_ != nullptr) {
            for (uint32_t b = 0; b < 40u; ++b) {
                PRINTF("[TRACE] blk=%d: %d %d %d %d %d %d %d %d | %d %d %d %d %d %d %d %d\\n",
                    (int)b,
                    (int)dbgGm_[b*64+0], (int)dbgGm_[b*64+1], (int)dbgGm_[b*64+2], (int)dbgGm_[b*64+3],
                    (int)dbgGm_[b*64+4], (int)dbgGm_[b*64+5], (int)dbgGm_[b*64+6], (int)dbgGm_[b*64+7],
                    (int)dbgGm_[b*64+8], (int)dbgGm_[b*64+9], (int)dbgGm_[b*64+10], (int)dbgGm_[b*64+11],
                    (int)dbgGm_[b*64+12], (int)dbgGm_[b*64+13], (int)dbgGm_[b*64+14], (int)dbgGm_[b*64+15]);
            }
        }""")

open(p, "w").write(s)
print("patched, SyncAll sites =", site[0])
