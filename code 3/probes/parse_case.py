#!/usr/bin/env python3
"""P68：把 case .bin 解开 -> 逐 (row, head) 的真 score 矩阵 + 在线 softmax 的期望 m/l。
纯标准库（无 numpy）；只用于本地对拍，不进提交。"""
import struct, sys, math, json

def load(path):
    raw = open(path, 'rb').read()
    assert raw[:11] == b'SFA_CASE 2\n', raw[:11]
    i = 11
    hdr = {}
    while True:
        j = raw.index(b'\n', i)
        line = raw[i:j].decode('ascii')
        i = j + 1
        k, v = line.split(' ', 1)
        hdr[k] = v
        if k == 'LSE':
            break
    H = {k: int(v) for k, v in hdr.items() if k not in ('SCALE',)}
    H['SCALE'] = float(hdr['SCALE'])
    B, S1, S2, N1, D, SBS, COUNT = (H['B'], H['S1'], H['S2'], H['N1'], H['D'], H['SBS'], H['COUNT'])
    Dr = 64

    def f16(n):
        vals = struct.unpack('<%de' % n, raw[i:i + 2 * n]); return list(vals), i_up(n, 2)
    def take(n, fmt, sz):
        nonlocal i
        v = list(struct.unpack('<%d%s' % (n, fmt), raw[i:i + sz * n])); i += sz * n; return v

    q  = take(B * S1 * N1 * D, 'e', 2)
    k  = take(B * S2 * D, 'e', 2)
    v  = take(B * S2 * D, 'e', 2)
    qr = take(B * S1 * N1 * Dr, 'e', 2)
    kr = take(B * S2 * Dr, 'e', 2)
    idx = take(B * S1 * COUNT, 'i', 4)
    asq = take(B, 'i', 4)
    ask = take(B, 'i', 4)
    exp_out = take(B * S1 * N1 * D, 'e', 2)
    exp_max = take(B * S1 * N1, 'f', 4)
    exp_sum = take(B * S1 * N1, 'f', 4)
    H.update(B=B, S1=S1, S2=S2, N1=N1, D=D, SBS=SBS, COUNT=COUNT, Dr=Dr,
             q=q, k=k, v=v, qr=qr, kr=kr, idx=idx, asq=asq, ask=ask,
             exp_out=exp_out, exp_max=exp_max, exp_sum=exp_sum)
    return H


def tokens_of(C, b, s):
    """MODE=3 / SBS 展开后的有序 token 列表（遇 -1 即停 + 阈值截断）"""
    B, S1, S2, SBS, COUNT, MODE = C['B'], C['S1'], C['S2'], C['SBS'], C['COUNT'], C['MODE']
    act_s1 = C['asq'][b] if b < len(C['asq']) else S1
    act_s2 = C['ask'][b] if b < len(C['ask']) else S2
    if s >= act_s1:
        return None
    thr = (act_s2 - act_s1) + s + 1 if MODE == 3 else act_s2
    out = []
    base = (b * S1 + s) * COUNT
    for j in range(COUNT):
        blk = C['idx'][base + j]
        if blk < 0:
            break
        for o in range(SBS):
            t = blk * SBS + o
            if t < thr and t < act_s2:
                out.append(t)
    return out


def scores(C, b, s, toks):
    """score[h][i] = (q_h · k_i + qr_h · kr_i) —— 【不乘 scale】，留给调用方"""
    N1, D, Dr = C['N1'], C['D'], C['Dr']
    row = []
    for h in range(N1):
        qo = ((b * C['S1']) + s) * N1 * D + h * D
        qro = ((b * C['S1']) + s) * N1 * Dr + h * Dr
        arr = []
        for t in toks:
            ko = (b * C['S2'] + t) * D
            koro = (b * C['S2'] + t) * Dr
            acc = 0.0
            q = C['q']; kk = C['k']
            for d in range(D):
                acc += q[qo + d] * kk[ko + d]
            for d in range(Dr):
                acc += C['qr'][qro + d] * C['kr'][koro + d]
            arr.append(acc)
        row.append(arr)
    return row


def online(S, scale, beg=0, end=None):
    """按 chunk 顺序在线 softmax（口径与 kernel 一致：m 初值 -2e38，P=exp(clamp(s-m,-88))）"""
    m = -2e38
    l = 0.0
    o = [0.0] * 16   # 只要前 16 维看形状
    for i, x in enumerate(S):
        if i < beg or (end is not None and i >= end):
            continue
        xs = x * scale
        mn = max(m, xs)
        alpha = math.exp(max(m - mn, -88.0))
        l = l * alpha + math.exp(max(xs - mn, -88.0))
        m = mn
    return m, l


if __name__ == '__main__':
    path = sys.argv[1]
    C = load(path)
    print('hdr:', {k: C[k] for k in ('B', 'S1', 'S2', 'N1', 'D', 'SBS', 'COUNT', 'MODE', 'SCALE')})
    b, s = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 0)
    toks = tokens_of(C, b, s)
    print('row(%d,%d) tokens=%d 前 8: %s 末 8: %s' % (b, s, len(toks), toks[:8], toks[-8:]))
    S = scores(C, b, s, toks)
    sc = C['SCALE']
    for h in range(C['N1']):
        row = S[h]
        raw = max(row); sml = min(row)
        m, l = online(row, sc)
        print(' h=%d  max_raw=%9.4f  max_scaled=%8.5f  gold max/sum=%8.4f/%9.3f  '
              'local m=%8.4f l=%9.3f   mean=%7.4f' %
              (h, raw, raw * sc, C['exp_max'][b * C['S1'] * C['N1'] + s * C['N1'] + h],
               C['exp_sum'][b * C['S1'] * C['N1'] + s * C['N1'] + h], m, l,
               sum(row) / len(row)))
