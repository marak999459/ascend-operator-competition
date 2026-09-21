#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务 #22 探针的 harness 侧读数补丁（**只在远端跑**，改 ~/sfa_real/test_sfa_dev.cpp）。

配合 mk_probe_wsaddr.py：打印
  1) 调用方自己拿到的 wsSize / ws / out / key 地址；
  2) kernel 回报的 `workspace - attention_out`、`workspace - key` 指针差（int32 高低位拼回），
     以及 40 个块的 workspace 是否同址；
  3) kernel 的完成标记（float[500]==9）+ 调用方 ws buffer 里 5.0f 的落点分布。
跑完直接 return 6（探针 kernel 本来就不算数，别让后面的正确性判定搅浑输出）。
"""
import io
import os
import re
import sys

P = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/sfa_real/test_sfa_dev.cpp')
s = io.open(P, encoding='utf-8').read()
if '[WSADDRPROBE]' in s:
    print('harness 已打过补丁，跳过')
    sys.exit(0)

A1 = ('    if (wsSize > 0 && aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)'
      ' { std::printf("[FAIL] ws\\n"); return 2; }\n')
assert s.count(A1) == 1, 'A1 miss'
s = s.replace(A1, A1 + '''    // ---- [WSADDRPROBE] 读数 1/3：调用方侧地址 + 出口缓冲清零 ----
    std::printf("[WSADDR] wsSize=%llu ws=%p out=%p key=%p dsum=%p\\n",
                (unsigned long long)wsSize, ws, d_out, d_k, d_sum);
    if (wsSize > 0) { aclrtMemset(ws, wsSize, 0, wsSize); }
    aclrtMemset(d_sum, nlse*4, 0, nlse*4);
    aclrtMemset(d_max, nlse*4, 0, nlse*4);
''', 1)

A2 = '    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) { std::printf("[FAIL] sync（kernel 挂了？）\\n"); return 2; }\n'
assert s.count(A2) == 1, 'A2 miss'
s = s.replace(A2, A2 + '''    {   // ---- [WSADDRPROBE] 读数 2/3：kernel 侧地址 + 多次 launch 稳定性 ----
        for (int extra = 0; extra < 3; ++extra) {
            if (!RunOnce()) { std::printf("[WSADDR] [FAIL] launch #%d\\n", extra + 2); return 6; }
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                std::printf("[WSADDR] [FAIL] sync #%d  ⇒ 第 %d 次 launch 挂\\n", extra + 2, extra + 2); return 6; }
        }
        constexpr int NBI = 40;
        std::vector<int32_t> ri(NBI * 4);
        std::vector<float> rf(nlse);
        std::vector<float> wsv(wsSize / 4u);
        if (aclrtMemcpy(ri.data(), NBI * 4 * 4, d_sum, NBI * 4 * 4, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            aclrtMemcpy(rf.data(), nlse * 4, d_sum, nlse * 4, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
            (wsSize > 0 && aclrtMemcpy(wsv.data(), wsSize, ws, wsSize, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS)) {
            std::printf("[WSADDR] [FAIL] D2H\\n"); return 6;
        }
        auto ld = [&](int b) {
            return static_cast<int64_t>(static_cast<uint32_t>(ri[b * 4])) |
                   (static_cast<int64_t>(ri[b * 4 + 1]) << 32);
        };
        auto lk = [&](int b) {
            return static_cast<int64_t>(static_cast<uint32_t>(ri[b * 4 + 2])) |
                   (static_cast<int64_t>(ri[b * 4 + 3]) << 32);
        };
        int ndiff = 0;
        for (int b = 1; b < NBI; ++b) { if (ld(b) != ld(0) || lk(b) != lk(0)) { ++ndiff; } }
        std::printf("[WSADDR] kernel: workspace-out = %lld 元素 = %lld B   workspace-key = %lld B\\n",
                    (long long)ld(0), (long long)ld(0) * 4, (long long)lk(0) * 4);
        std::printf("[WSADDR] block1..39 与 block0 不同址的块数 = %d   b1=%lld B b2=%lld B\\n",
                    ndiff, (long long)ld(1) * 4, (long long)ld(2) * 4);
        {   // 到达标记 int32[200+bi]==7 ⇒ 这个块真的跑到了探针里；==0 ⇒ 压根没启动
            const int32_t *rs = reinterpret_cast<const int32_t *>(rf.data());
            int miss = 0;
            std::printf("[WSADDR] 未到达块 = ");
            for (int b = 0; b < NBI; ++b) {
                if (nlse > 200 + b && rs[200 + b] == 0) { std::printf("%d ", b); ++miss; }
            }
            std::printf("共 %d/%d\\n", miss, NBI);
        }
        std::printf("[WSADDR] (out + delta) = %p  vs  调用方 ws = %p   (差 %lld B)\\n",
                    reinterpret_cast<void *>(reinterpret_cast<char *>(d_out) + ld(0) * 4), ws,
                    ws ? (long long)(reinterpret_cast<char *>(d_out) + ld(0) * 4 -
                                     reinterpret_cast<char *>(ws)) : 0LL);
        std::printf("[WSADDR] 完成标记 float[500] = %.1f (9=kernel 走完)\\n",
                    nlse > 500 ? rf[500] : -1.0f);
        size_t hits = 0, first = 0, last = 0;
        for (size_t i = 0; i < wsv.size(); ++i) {
            if (wsv[i] == 5.0f) { if (!hits) { first = i; } last = i; ++hits; }
        }
        std::printf("[WSADDR] 调用方 ws buffer 里 5.0f 命中 = %zu  首元素 = %zu  末元素 = %zu\\n",
                    hits, first, last);
        std::printf("[WSADDR] => 探针结束，退出（不算正确性）\\n");
        return 6;
    }
''', 1)

io.open(P, 'w', encoding='utf-8').write(s)
print('harness 已打 WSADDR 补丁')
