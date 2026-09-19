#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 batch 分解 c6_multiB 的对拍差异，定位结构性错误"""
import glob
import struct

CASE = '/home/fszqsn/sfa_work/c6_multiB.bin'

d = open(CASE, 'rb').read()
pos = 0
h = {}
for _ in range(11):
    e = d.index(b'\n', pos)
    line = d[pos:e].decode().strip()
    pos = e + 1
    if line.startswith('SFA_CASE'):
        continue
    k, v = line.split()
    h[k] = float(v)

B, S1, S2, N1, D, CN = (int(h['B']), int(h['S1']), int(h['S2']),
                        int(h['N1']), int(h['D']), int(h['COUNT']))
print('B=%d S1=%d S2=%d N1=%d D=%d COUNT=%d SBS=%d MODE=%d'
      % (B, S1, S2, N1, D, CN, h['SBS'], h['MODE']))

nQ = B * S1 * N1 * D
per = S1 * N1 * D
nK = B * S2 * D
nR = B * S1 * N1 * 64
nKR = B * S2 * 64
nI = B * S1 * CN

# 索引段偏移
idx_off = pos + nQ * 2 + nK * 2 + nK * 2 + nR * 2 + nKR * 2
exp_off = idx_off + nI * 4
exp = struct.unpack('<%de' % nQ, d[exp_off:exp_off + nQ * 2])

# 打印每 batch 每 s 的稀疏索引
idx = struct.unpack('<%di' % nI, d[idx_off:idx_off + nI * 4])
print('\n稀疏索引 (每行前几个有效块号):')
for b in range(B):
    for s in range(S1):
        base = (b * S1 + s) * CN
        vals = [idx[base + i] for i in range(CN)]
        valid = [v for v in vals if v >= 0]
        print('  b=%d s=%d valid=%s' % (b, s, valid))

best = None
bn = -1
for f in glob.glob('/tmp/sfa_pid_*.bin'):
    dd = open(f, 'rb').read()
    if len(dd) != nQ * 2:
        continue
    v = struct.unpack('<%de' % nQ, dd)
    nz = sum(1 for x in v if x != 0.0)
    if nz > bn:
        bn = nz
        best = v

print('\nbest dump: nz=%d/%d' % (bn, nQ))
print('\n每 batch 分解:')
for b in range(B):
    lo, hi = b * per, (b + 1) * per
    bad = 0
    for i in range(lo, hi):
        if abs(best[i] - exp[i]) > 1e-3 and abs(best[i] - exp[i]) / (abs(exp[i]) + 1e-3) > 1e-2:
            bad += 1
    nz = sum(1 for i in range(lo, hi) if best[i] != 0.0)
    enz = sum(1 for i in range(lo, hi) if exp[i] != 0.0)
    print('  batch %d: bad=%4d/%d  got_nz=%4d exp_nz=%4d' % (b, bad, per, nz, enz))
    for s in range(S1):
        slo = lo + s * N1 * D
        shi = slo + N1 * D
        sbad = 0
        for i in range(slo, shi):
            if abs(best[i] - exp[i]) > 1e-3 and abs(best[i] - exp[i]) / (abs(exp[i]) + 1e-3) > 1e-2:
                sbad += 1
        if sbad:
            print('      s=%d bad=%d/%d  got[0..3]=%s exp[0..3]=%s'
                  % (s, sbad, N1 * D,
                     ['%.5f' % x for x in best[slo:slo + 4]],
                     ['%.5f' % x for x in exp[slo:slo + 4]]))
