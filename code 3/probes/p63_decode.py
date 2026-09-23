#!/usr/bin/env python3
# P63 解码：从 write 模式落盘的 .bin 里把 16 槽 fp16 记录挑出来。
#   marker 77 = AIV（槽 = 行尾 -256 + sub*16），78 = AIC（槽 = 行尾 -256 + 64）
#   记录里的值都 < 4096 ⇒ fp16 精确可表示，读回来不需要容差。
import os
import struct
import sys

d = sys.argv[1]
for fn in sorted(os.listdir(d)):
    p = os.path.join(d, fn)
    if not os.path.isfile(p):
        continue
    b = open(p, 'rb').read()
    if len(b) % 2:
        continue
    f = list(struct.unpack('%de' % (len(b) // 2), b))
    hits = 0
    for i, x in enumerate(f):
        if x != 77.0 and x != 78.0:
            continue
        if i + 16 > len(f):
            continue
        r = f[i:i + 16]
        if not (0.0 < r[2] < 64.0 and 0.0 <= r[1] < 128.0 and 0.0 < r[6] < 4096.0):
            continue
        tag = 'AIV' if r[0] == 77.0 else 'AIC'
        # AIV: blk nblk sub aChunk sets units beg end step headBase nb N1 nBlk cnt D
        # AIC: blk nblk sub cstep cp   --  units cu nHeadBlk N1 nBlk cnt
        print('%-22s @%-9d %-3s %-14s %s'
              % (fn, i, tag, 'blk%d/sub%d' % (r[1], r[4]), [int(x) for x in r]))
        hits += 1
    print('-- %s: %d 条记录 (%d 元素)' % (fn, hits, len(f)))
