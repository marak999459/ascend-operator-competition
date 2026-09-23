#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P71 判读（Gram + 环行号 ×4）：dump 的 attention_out 前 16 格 = AIV 从环里读回的 **Gram 行**（B 侧喂 A tile）。

期望（全部只用 p1.bin 的 Q/Qrope 本地算，不涉及 K/idx）：
  AIC 的 A tile = ND2NZ(qGm_[ab], nValue=16, dValue=D) ⇒ lane n ↔ GM 平铺行 (ab/D + n)；
  Mmad C[m][n] = Σ_k A[m][k]·B[n][k]，B=A ⇒ C = Gram；Fixpipe 把 m lane r 写成环的第 r 行；
  AIV 读环行 (headBase_+i) ⇒ 落进 attention_out 的【全局头 h=headBase_+i】那一行，
  所以 dumped[h][j] = SCALE · Gram_flat[f0+h][f0+j]，f0 = ab/D（p1/p2 都被夹到 0 ⇒ 与 s 无关）。
三条先验不变量（不看期望也能判）：Gram 对称、对角 = SCALE·‖q‖²>0、与 s 无关。
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_case import load  # noqa: E402


def rd_f16(path):
    d = open(path, 'rb').read()
    u = struct.unpack('<%dH' % (len(d) // 2), d)
    return [struct.unpack('e', struct.pack('H', x))[0] for x in u]


def rd_f32(path):
    d = open(path, 'rb').read()
    return list(struct.unpack('<%df' % (len(d) // 4), d))


def gram(C, f0, n=16):
    """Gram_flat[i][j] = ⟨q 平铺行 f0+i, q 平铺行 f0+j⟩ +（rope 同理），未乘 scale"""
    D, Dr = C['D'], C['Dr']
    rows = []
    for k in range(n):
        a, b = (f0 + k) * D, (f0 + k + 1) * D
        ra, rb = (f0 + k) * Dr, (f0 + k + 1) * Dr
        if b > len(C['q']) or rb > len(C['qr']):
            rows.append(None)
            continue
        rows.append(C['q'][a:b] + C['qr'][ra:rb])
    g = []
    for i in range(n):
        row = []
        for j in range(n):
            if rows[i] is None or rows[j] is None:
                row.append(float('nan'))
                continue
            row.append(sum(x * y for x, y in zip(rows[i], rows[j])))
        g.append(row)
    return g


LANE = lambda h: 4 * h          # [p71] AIV 读的是环行 4*(headBase_+i)
lane = LANE


def main(casename, casepath, dumpdir, nrow=99):
    C = load(casepath)
    B, S1, N1, sc = C['B'], C['S1'], C['N1'], C['SCALE']
    ou = rd_f16('%s/%s.out' % (dumpdir, casename))
    mx = rd_f32('%s/%s.max' % (dumpdir, casename))
    G = gram(C, 0, 16)
    print('== %s  B=%d S1=%d N1=%d D=%d  qTot行=%d  scale=%.6g' %
          (casename, B, S1, N1, C['D'], B * S1 * N1, sc))
    for b in range(B):
        for s in range(min(S1, nrow)):
            base = (b * S1 + s) * N1 * 512
            for h in range(N1):
                got = ou[base + h * 512: base + h * 512 + 16]
                exp = [sc * G[lane(h)][j] for j in range(16)]
                bad, nan = [], []
                for j in range(16):
                    if got[j] != got[j]:
                        nan.append(j)
                    elif abs(got[j] - exp[j]) > 2e-2 * max(1.0, abs(exp[j])):
                        bad.append(j)
                print(' (%d,%d) h=%d %s' % (b, s, h, ' '.join('%8.4f' % x for x in got)))
                print('          期望 %s' % ' '.join('%8.4f' % x for x in exp))
                print('          nan=%s 错格=%s' % (nan if nan else '-', bad if bad else '-'))
            if s == 0:
                print('  对称性（读回）: G[0][1]=%8.4f G[1][0]=%8.4f | G[0][2]=%8.4f G[2][0]=%8.4f'
                      % (ou[base + 0 * 512 + 1], ou[base + 1 * 512 + 0],
                         ou[base + 0 * 512 + 2], ou[base + 2 * 512 + 0]))
                print('  对角（读回）: %s   LSE max 列: %s' %
                      (' '.join('%8.4f' % ou[base + h * 512 + LANE(h)] for h in range(N1)),
                       ' '.join('%7.3f' % mx[(b * S1 + s) * N1 + h] for h in range(N1))))
    # 与 s 无关这条不变量：把每一行的前 16 格当成 16-bit 指纹看有几种
    fps = {}
    for b in range(B):
        for s in range(S1):
            base = (b * S1 + s) * N1 * 512
            for h in range(N1):
                f = tuple(round(x, 3) for x in ou[base + h * 512: base + h * 512 + 16])
                fps.setdefault(f, []).append('%d/%d' % (b * S1 + s, h))
    print('指纹种数=%d（期望 1 种：A tile 被夹到同一基址 ⇒ 与 s 无关）' % len(fps))
    for f, who in sorted(fps.items(), key=lambda kv: -len(kv[1]))[:4]:
        print('  x%-3d %s' % (len(who), ' '.join('%8.4f' % x for x in f)))
        print('        属于 %s' % ', '.join(who[:12]))


if __name__ == '__main__':
    for cs in (sys.argv[1:] or ['p1', 'p2']):
        main(cs, '/tmp/p68/%s.bin' % cs if os.path.exists('/tmp/p68/%s.bin' % cs)
             else '/tmp/p70/%s.bin' % cs, '/tmp/p71/dump')
