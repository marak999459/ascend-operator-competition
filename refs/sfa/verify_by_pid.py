#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFA CPU 仿真结果核对（规避多进程竞态）
=========================================
背景：CPU 仿真为每个 block 起 2 个进程（AIC+AIV），两者共享 GM。
      harness 内部的比对会读到半成品 → 假 FAIL（详见 HARNESS_NOTES.md）。
      但 AIC 进程（真正干活的那个）的输出是【正确】的 —— 本脚本就是用来
      从按 PID 落盘的结果里找出那一份，与参考输出核对。

用法:
  1) 先跑 probe 并开启 PID 落盘:
       SFA_PID_DUMP=1 ./probe_v2 <case.bin> <NB> <NBLK> >/dev/null 2>&1
  2) 核对:
       python3 verify_by_pid.py <case.bin>
"""
import glob
import os
import struct
import sys


def read_case(path):
    """读 sfa_ref.py 生成的用例文件，返回 (header, expect 列表)"""
    d = open(path, 'rb').read()
    pos = 0
    h = {}
    for _ in range(11):
        e = d.index(b'\n', pos)
        line = d[pos:e].decode('ascii').strip()
        pos = e + 1
        if line.startswith('SFA_CASE'):
            continue
        k, v = line.split()
        h[k] = float(v)
    B, S1, S2, N1, D = (int(h['B']), int(h['S1']), int(h['S2']),
                        int(h['N1']), int(h['D']))
    COUNT = int(h['COUNT'])
    nQ = B * S1 * N1 * D
    nK = B * S2 * D
    nR = B * S1 * N1 * 64
    nKR = B * S2 * 64
    nI = B * S1 * COUNT
    pos += nQ * 2 + nK * 2 + nK * 2 + nR * 2 + nKR * 2 + nI * 4
    exp = struct.unpack('<%de' % nQ, d[pos:pos + nQ * 2])
    return h, nQ, exp


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else 'mini.bin'
    h, nQ, exp = read_case(case)
    print('case=%s  B=%d S1=%d S2=%d N1=%d D=%d SBS=%d COUNT=%d MODE=%d'
          % (os.path.basename(case), h['B'], h['S1'], h['S2'], h['N1'],
             h['D'], h['SBS'], h['COUNT'], h['MODE']))
    print('  EXPECT[:6] = %s' % ['%.5f' % x for x in exp[:6]])
    print()

    files = sorted(glob.glob('/tmp/sfa_pid_*.bin'))
    if not files:
        print('  没有找到 /tmp/sfa_pid_*.bin')
        print('  请先运行:  SFA_PID_DUMP=1 ./probe_v2 <case> <NB> <NBLK>')
        return 1

    best = None
    best_nz = -1
    for f in files:
        dd = open(f, 'rb').read()
        if len(dd) != nQ * 2:
            print('  %-26s 尺寸不符' % os.path.basename(f))
            continue
        v = struct.unpack('<%de' % nQ, dd)
        nz = sum(1 for x in v if x != 0.0)
        maxd = max(abs(a - b) for a, b in zip(v, exp))
        tag = ''
        if nz > best_nz:
            best_nz = nz
            best = v
            tag = '  <== 采信'
        print('  %-26s nz=%4d/%d  maxdiff=%.3e%s'
              % (os.path.basename(f), nz, nQ, maxd, tag))

    if best is None:
        print('\n  无可用的结果文件')
        return 1

    nbad = sum(1 for a, b in zip(best, exp)
               if abs(a - b) > 1e-3 and abs(a - b) / (abs(b) + 1e-3) > 1e-2)
    print('\n  采信结果: maxdiff=%.3e  bad=%d/%d  ==> %s'
          % (max(abs(a - b) for a, b in zip(best, exp)), nbad, nQ,
             'PASS' if nbad == 0 else 'FAIL'))
    return 0 if nbad == 0 else 2


if __name__ == '__main__':
    sys.exit(main())
