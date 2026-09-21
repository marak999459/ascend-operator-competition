#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务 #22：arch22 上 aicore 收到的 `workspace` 形参到底指向哪块内存、可写窗口多大。

用法: mk_probe_wsaddr.py <mode>   —— 把打完探针补丁的 kernel 打到 stdout（只推远端副本）

背景（源码 + 真机双向对上）：
  构建侧生成的入口桩在调用户函数前先做 `GM_ADDR usr = AscendC::GetUserWorkspace(workspace);`，
  而 arch22 的实现是
      return __get_kfc_workspace_addr() + RESERVED_WORKSPACE;   // 2201: RESERVED = 16 MB
  ⇒ **用户函数里的 workspace 形参 = 框架 KFC 基址 + 16 MB，和调用方 aclrtMalloc 那块的关系
     只有"差一个固定 16 MB"**（§15.18 当时只证到"不是同一块"，没量出偏移到 16 MB）。
  如果可写窗口正好就是调用方声明的那段，那么"往 `workspace - 16MB` 写"就是**零额外成本**的
  跨核暂存通道（只需把 op 声明的 workspace 从 0 提到"够用即可"），P11 立刻重开。

模式（host 侧一律 sed 成声明 128 KB，见 run_wsaddr.sh）：
  addr        只回报指针差，不碰 workspace ⇒ 先确认"报告通道 + 指针算术"本身清白。
  wsbase0     写 `workspace - 16MB + 0`        ⇒ 期望 OK ⇒ 通道成立（P11 的修法）。
  wsbase120k  写 `workspace - 16MB + 120 KB`   ⇒ 期望 OK ⇒ 窗口 ≈ 声明尺寸（128 KB）。
  wsbase1m    写 `workspace - 16MB + 1 MB`     ⇒ 期望挂 ⇒ 窗口不超过声明尺寸（反向对照）。
  wsself      写 `workspace + 0`               ⇒ 期望挂 ⇒ 复现 §15.18 的原始故障（机理闭环）。

报告通道 = softmax_sum_out 的 **int32 视图**（aicore 里 float↔整数转换一律禁止，
所以指针差用 int64 拆高低 32 位写，harness 侧按 int32 打印）：
  int32[4*bi + 0..1] = (workspace - attention_out) 的 lo/hi（元素数，/4 才是字节）
  int32[4*bi + 2..3] = (workspace - key) 的 lo/hi
  float [500]        = 9.0 完成标记（写在测试写之后 ⇒ 标记在 = 测试写没挂）
  测试写本身落在调用方那块 buffer 里（`5.0f`，每块偏 64 个 float），由 harness 直接 memcpy
  `ws` 回来验：既证明"写得进去"，也证明"这块 buffer 就是调用方 aclrtMalloc 的那块"。
"""
import io
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'addr'
OFF_ELEMS = {                      # 测试写点相对 `workspace - 16MB` 的元素偏移（float 个数）
    'addr': None,
    'wsbase0': 0,
    'wsbase120k': 120 * 1024 // 4,
    'wsbase1m': 1024 * 1024,
    'wsself': -4 * 1024 * 1024,    # 相对 `workspace - 16MB` 偏 -16MB == 相对 workspace 偏 0
}
assert MODE in OFF_ELEMS, MODE
RESERVED_F = 4 * 1024 * 1024       # 16 MB / 4 = 4194304 个 float

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
SRC = os.path.join(BASE, 'op_kernel', 'sparse_flash_attention.cpp')
s = io.open(SRC, encoding='utf-8').read()

ANCHOR = '    GET_TILING_DATA_WITH_STRUCT(SparseFlashAttentionTilingData, tiling_data, tiling);\n'
assert s.count(ANCHOR) == 1, 'entry anchor miss'

TEST_WRITE = ('' if OFF_ELEMS[MODE] is None else (
    '        // 每块各写自己那 256 B（64 个 float），顺带证明"多块共用同一基址 + 互不覆盖"\n'
    '        GlobalTensor<float> wg;\n'
    '        wg.SetGlobalBuffer(base + %d + bi * 64u);\n'
    '        wg.SetValue(0, 5.0f);\n' % OFF_ELEMS[MODE]))

PROBE = (
    '\n    // ==================== [WSADDRPROBE] 远端探针专用（绝不进提交源） ====================\n'
    '    {\n'
    '        const uint32_t bi = GetBlockIdx();\n'
    '        __gm__ float *pws = reinterpret_cast<__gm__ float *>(workspace);\n'
    '        __gm__ float *pout = reinterpret_cast<__gm__ float *>(attentionOut);\n'
    '        __gm__ float *pkey = reinterpret_cast<__gm__ float *>(key);\n'
    '        __gm__ float *base = pws - %d;           // 假设：调用方那块 buffer 的头\n' % RESERVED_F +
    '        GlobalTensor<int32_t> rg;\n'
    '        rg.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(softmaxSumOut));\n'
    '        rg.SetValue(200 + bi, 7);           // 到达标记：钉死"有块没启动"还是"槽位/竞争问题"（§15.21 末）\n'
    '        const int64_t dOut = pws - pout;\n'
    '        const int64_t dKey = pws - pkey;\n'
    '        rg.SetValue(4 * bi + 0, static_cast<int32_t>(dOut & 0xFFFFFFFF));\n'
    '        rg.SetValue(4 * bi + 1, static_cast<int32_t>(dOut >> 32));\n'
    '        rg.SetValue(4 * bi + 2, static_cast<int32_t>(dKey & 0xFFFFFFFF));\n'
    '        rg.SetValue(4 * bi + 3, static_cast<int32_t>(dKey >> 32));\n'
    + TEST_WRITE +
    '        GlobalTensor<float> mg;\n'
    '        mg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(softmaxSumOut));\n'
    '        mg.SetValue(500, 9.0f);\n'
    '        return;\n'
    '    }\n'
)

s = s.replace(ANCHOR, ANCHOR + PROBE, 1)
sys.stdout.write(s)
