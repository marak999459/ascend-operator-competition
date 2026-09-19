#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比 /tmp/sfa_pid_*.bin 与用例文件里的参考输出"""
import glob
import struct
import sys
import os


def read_header(path):
    d = open(path, 'rb').read()
    pos = 0
    hdr = {}
    for _ in range(11):
        e = d.index(b'\n', pos)
        line = d[pos:e].decode('ascii').strip()
        pos = e + 1
        if line.startswith('SFA_CASE'):
            continue
        k, v = line.split()
        hdr[k] = float(v)
    return d, pos, hdr


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else '/home/fszqsn/sfa_work/mini.bin'
    d, pos, h = read_header(case)
    B, S1, S2, N1, D = int(h['B']), int(h['S1']), int(h['S2']), int(h['N1']), int(h['D'])
    COUNT = int(h['COUNT'])
    nQ = B * S1 * N1 * D
    nK = B * S2 * D
    nR = B * S1 * N1 * 64
    nKR = B * S2 * 64
    nI = B * S1 * COUNT
    pos += nQ * 2 + nK * 2 + nK * 2 + nR * 2 + nKR * 2 + nI * 4
    exp = struct.unpack('<%de' % nQ, d[pos:pos + nQ * 2])

    print('case=%s  B=%d S1=%d S2=%d N1=%d D=%d COUNT=%d' % (os.path.basename(case), B, S1, S2, N1, D, COUNT))
    print('EXPECT[:8] =', ['%.5f' % x for x in exp[:8]])

    for f in sorted(glob.glob('/tmp/sfa_pid_*.bin')):
        dd = open(f, 'rb').read()
        if len(dd) != nQ * 2:
            print('  %s  尺寸不符 %d != %d' % (f, len(dd), nQ * 2))
            continue
        v = struct.unpack('<%de' % nQ, dd)
        nz = sum(1 for x in v if x != 0.0)
        maxd = max(abs(a - b) for a, b in zip(v, exp))
        print('  %-28s nz=%4d/%d  maxdiff=%.3e  [:6]=%s'
              % (os.path.basename(f), nz, nQ, maxd, ['%.5f' % x for x in v[:6]]))


if __name__ == '__main__':
    main()
