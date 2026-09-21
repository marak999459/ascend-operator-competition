#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""差分探针：给"每 token 一次 idx 标量 GM 读"和"每 token 三条 MTE2 搬运"标价。

纪律同 §15.11 的 bench 探针 —— **只写远端副本**，本地提交源一个字节不动，跑完立刻
`dev.sh build` 覆盖回干净版。探针会让结果数值错，只读时间。

    SFA_PROBE_NOIDX : NextTokenBlock 不读 GM，blk 用 (tokIdx*97)&2047 这个置换代替
                      —— 同一批 token、同样分散 ⇒ K/V 访存局部性与真版本同级
    SFA_PROBE_NOMTE : chunk 填充的三条 CopyGm2Ub 去掉，filled 照常推进
                      —— V 侧调用条数/rep 完全不变，读的是残留数据

用法（把打好的文件写到 stdout，再管道进 ssh）：
    python3 mk_probe_idxscan.py <noidx|nomte|both|clean>
"""
import sys

SRC = "/home/fszqsn/ops_comp/ascend-operator-competition/code 3/code/op_kernel/sparse_flash_attention.cpp"

SCAN_REAL = """            const int32_t blk = idxGm_.GetValue(idxBase + tokIdx);
            ++tokIdx;
            if (blk < 0) { return false; }                 // 官方：遇 -1 即停
"""
COPY_REAL2 = """            CopyGm2Ub(kb[filled * D_],     kGm_[kOff],  run * D_);
            CopyGm2Ub(vb[filled * D_],     vGm_[kOff],  run * D_);
            CopyGm2Ub(kr[filled * Dr_],    krGm_[rOff], run * Dr_);
"""

mode = sys.argv[1] if len(sys.argv) > 1 else 'clean'
text = open(SRC, encoding='utf-8').read()

DEF = {'noidx': 'SFA_PROBE_NOIDX', 'nomte': 'SFA_PROBE_NOMTE',
       'nosoftmax': 'SFA_PROBE_NOSOFTMAX', 'notail': 'SFA_PROBE_NOTAIL'}
defines = ['SFA_PROBE_NOIDX', 'SFA_PROBE_NOMTE'] if mode == 'both' else \
    ([] if mode == 'clean' else [DEF[mode]])

anchor = "\n__aicore__ inline uint32_t UbAlignBuf(uint32_t len)"
assert text.count(anchor) == 1, 'anchor UbAlignBuf'
if defines:
    text = text.replace(anchor, '\n' + '\n'.join('#define ' + d for d in defines) + '\n' + anchor)

if 'SFA_PROBE_NOIDX' in defines:
    assert text.count(SCAN_REAL) == 1, 'scan anchor'
    probe = ('#ifdef SFA_PROBE_NOIDX\n'
             '            const int32_t blk = static_cast<int32_t>((tokIdx * 97u) & 2047u);\n'
             '            ++tokIdx;\n'
             '#else\n' + SCAN_REAL + '#endif\n')
    text = text.replace(SCAN_REAL, probe)

if 'SFA_PROBE_NOMTE' in defines:
    assert text.count(COPY_REAL2) == 1, 'copy anchor'
    probe = ('#ifdef SFA_PROBE_NOMTE\n'
             '            (void)kOff; (void)rOff;\n'
             '#else\n' + COPY_REAL2 + '#endif\n')
    text = text.replace(COPY_REAL2, probe)

TAIL_REAL = """                WholeReduceSum(rd, pf, vc, g, 1, 1, rowC / sfa::F32_PER_BLK);
                WholeReduceSum(rd2, prf, vr, g, 1, 1, rowR / sfa::F32_PER_BLK);
                Add(out, rd, rd2, g);
                Muls(out, out, scale_, g);
"""
if 'SFA_PROBE_NOTAIL' in defines:
    assert text.count(TAIL_REAL) == 1, 'tail anchor'
    text = text.replace(TAIL_REAL, '#ifdef SFA_PROBE_NOTAIL\n                (void)out; (void)rd2;\n'
                                   '#else\n' + TAIL_REAL + '#endif\n')

FLUSH_REAL = """        SoftmaxPv(o, vb, sc, ml, nbCur, m);
"""
if 'SFA_PROBE_NOSOFTMAX' in defines:
    assert text.count(FLUSH_REAL) == 1, 'flush anchor'
    text = text.replace(FLUSH_REAL, '#ifdef SFA_PROBE_NOSOFTMAX\n'
                                    '        (void)vb; (void)ml;\n'
                                    '#else\n' + FLUSH_REAL + '#endif\n')

sys.stdout.write(text)
