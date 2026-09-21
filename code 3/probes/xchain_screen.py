#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""**第二段** harness 补丁（只改远端副本）：在 `cube_harness_patch.py` v8 的读数区里加一屏 CHAIN。

必须**先**打 v8（它负责 D2H `d_max`/`d_sum`、出口缓冲清零、以及 `SFA_CUBE_EXIT` 那句 return），
本脚本只在那句 return 之前插一段只读判定 ⇒ 探针档（`mk_probe_xchain.py`）的三条结论一屏看完：
  pubN       = 40 个发布槽里真正 >999.5 的个数      （=40 ⇒ 40 块的标量写都落了地）
  sWrote     = 40 个读回槽里非 0 的个数             （=40 ⇒ 每块都执行到了读回那句）
  sOk        = 读回值恰为偏置 1.0 的个数            （=40 ⇒ V 流水标量写跨核可见）
  sBad       = 读回值既非 0 也非 1 的个数           （读错 / 读到别人的图案）
  pubBad     = 批量发布区里对不上自己图案的元素数  （=0 ⇒ MTE3 DataCopy 写得出去）
  relayBad   = 中继区里对不上邻居图案的元素数      （=0 ⇒ 别人写的字节我 DataCopy 读得回来）
阴性对照 `nosync` 档：去掉 SyncAll 后 relayBad 应大面积非 0；**若它还是 0，说明这一屏根本不
依赖屏障，本次探针不能用来裁定通道**（§15.30 系列的教训：先证"读数有意义"，再证结论）。
"""
import io
import os

p = os.path.join(os.path.expanduser('~'), 'sfa_real', 'test_sfa_dev.cpp')
t = io.open(p, encoding='utf-8').read()
BR0 = t.count('{') - t.count('}')

KEY = '[CUBE] CHAIN'
if KEY in t:
    print('harness already patched (chain v1)')
    raise SystemExit(0)

A = 'if (std::getenv("SFA_CUBE_EXIT") != nullptr) { return 7; }'
if t.count(A) != 1:
    raise SystemExit('[FAIL] 找不到 v8 的 SFA_CUBE_EXIT 锚点（count=%d）⇒ 先跑 cube_harness_patch.py'
                     % t.count(A))
i = t.index(A)
j = t.rindex('\n', 0, i) + 1

INJ = '''        if (std::getenv("SFA_CHAIN") != nullptr) {   /* [CUBEPROBE] XCHAIN：AIV->AIV 输出张量通道 */
            const int N = 40;
            const float *gmx = pm.data();
            int pbad = 0, rbad = 0, sok = 0, swrite = 0, sbad = 0, pubn = 0;
            double smax = 0.0, rmax = 0.0;
            for (int x = 0; x < N; ++x) { if (gsum[x] > 999.5) { ++pubn; } }   // 真实发板块数
            /* 读回槽三态：1=通道通、0=这块根本没写、其余=读错了（出口缓冲 launch 前被 memset，
               不偏置的话"通"和"没写"都是 0，分不开）。 */
            for (int x = 0; x < N; ++x) {
                const double v = std::fabs((double)gsum[N + x]);
                if (v < 1e-6) { continue; }
                ++swrite;
                const double e = std::fabs(v - 1.0);
                if (e < 1e-4) { ++sok; } else { ++sbad; if (e > smax) { smax = e; } }
            }
            for (int x = 0; x < N; ++x) {
                const int q = (x + 1) % N;
                for (int g = 0; g < 16; ++g) {
                    const double pub = (double)x + (double)g * 0.001;
                    const double rel = (double)q + (double)g * 0.001;
                    const double dp = std::fabs((double)gmx[x * 16 + g] - pub);
                    const double dr = std::fabs((double)gsum[80 + x * 16 + g] - rel);
                    if (dp > 2e-3) { ++pbad; }
                    if (dr > 2e-3) { ++rbad; if (dr > rmax) { rmax = dr; } }
                }
            }
            std::printf("[CUBE] CHAIN N=%d pubN=%d sWrote=%d sOk=%d sBad=%d sMax=%.6g pubBad=%d relayBad=%d "
                        "relayMax=%.6g | s[0]=%.4f sN=%d relay[0]=%.4f expect=%.4f pub[0]=%.4f\\n",
                        N, pubn, swrite, sok, sbad, smax, pbad, rbad, rmax, (double)gsum[0],
                        (int)gsum[N], (double)gsum[80], (double)gmx[16], (double)gmx[0]);
        }
'''

t = t[:j] + INJ + t[j:]
BR = t.count('{') - t.count('}')
if BR != BR0:
    raise SystemExit('[FAIL] 注入后花括号平衡从 %d 变成 %d ⇒ 拒绝落盘' % (BR0, BR))
io.open(p, 'w', encoding='utf-8').write(t)
print('harness patched: [CUBEPROBE] chain v1 (braces %+d -> %+d)' % (BR0, BR))
