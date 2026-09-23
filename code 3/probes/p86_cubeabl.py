#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P86：cube 臂自己的段级消融（P85 那轮只砍到了 AIV 侧，AIC 侧的两段大头从没单独量过）。

用法: p86_cubeabl.py <mode>   —— 把探针 kernel 打到 stdout，**只推远端副本**
  cubekg    删每片的 K / K-rope 那条 `DataCopy`（GM->L1 的 ND2NZ gather，M1d 每片 nRun 条）
  cubek1    五片只跑第一片（content 0~127）⇒ MAC/L0 搬运各留 22 %，**结构一字不动**。
            这是"cube 计算份额"唯一能跑的档 —— 见下面 (P86 实测) 那条。
  cubel0    （已废）五片全短路 `j < 0u`：真机**挂死**，test_sfa_dev 连一行时间都不打。
  cubefx    （已废）删每片末尾那条 `Fixpipe`：真机**当场报错**，同样零读数。
  cubenull  （已废）= cubekg + cubel0 + cubefx：跟着 cubefx 一起废。
判据只有时间：这些档的输出必然是错的 ⇒ 一律 `act=none`/`diff` 读时间，绝不 `act=write`。
差分口径与 §15.73(m) 的 AIV 侧表同构：`full − 本档` = 该段在**当前锁步形态**下的边际上界
（编译器可能顺手把只被计算读的缓冲的搬运也优化掉 ⇒ 上界，不是份额）。
"""
import io
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
ALL = ('full', 'cubekg', 'cubel0', 'cubefx', 'cubek1', 'cubenull')
assert MODE in ALL, MODE

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
SRC = os.path.join(BASE, 'op_kernel', 'sparse_flash_attention.cpp')
s = io.open(SRC, encoding='utf-8').read()
n0 = s.count('\n')


def drop_like(text, needle, tag):
    """按"剥掉缩进后包含 needle"定位整行并注释掉；命中数不是 1 就当场挂（宁挂不读假数）。"""
    lines = text.split('\n')
    hit = [i for i, ln in enumerate(lines) if needle in ln.lstrip()]
    assert len(hit) == 1, 'needle 命中 %d 次（应为 1）: %s' % (len(hit), tag)
    i = hit[0]
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = ind + '// [ABL] %s 已删（档 %s）' % (tag, MODE)
    return '\n'.join(lines)


KG_K = 'DataCopy(ctx.l1ka[done * 16u], kGm_[rn'
KG_R = 'DataCopy(ctx.l1kr[done * 16u], krGm_[rn'
FX_LINE = 'Fixpipe('

if MODE in ('cubekg', 'cubenull'):
    s = drop_like(s, KG_K, '每片 K 的 ND2NZ gather')
    s = drop_like(s, KG_R, '每片 K-rope 的 ND2NZ gather')
if MODE in ('cubel0', 'cubenull'):
    # 五片 L1->L0 + Mmad 整段短路：把循环上界写成 0（体内三对旗标本成对 ⇒ 不留孤儿）
    old = 'for (uint32_t j = 0u; j < 5u; ++j) {'
    assert s.count(old) == 1, 'slice loop anchor miss: %d' % s.count(old)
    s = s.replace(old, 'for (uint32_t j = 0u; j < 0u; ++j) {   // [ABL] 五片 LoadData+Mmad 短路', 1)
if MODE == 'cubek1':
    # 只跑第一片，并且【把链尾标志改成本片】—— unitFlag=0b10 的意思是"操作数还要被链上
    # 后续单元用"，一片都不发 0b11 的话 L0C 永远等不到"可以搬出"⇒ 与 cubel0 同一种挂法。
    old = 'for (uint32_t j = 0u; j < 5u; ++j) {'
    assert s.count(old) == 1, 'slice loop anchor miss: %d' % s.count(old)
    s = s.replace(old, 'for (uint32_t j = 0u; j < 1u; ++j) {   // [ABL] 五片只跑第一片', 1)
    old2 = 'ctx.mp.unitFlag = rp ? 0b11u : 0b10u;'
    assert s.count(old2) == 1, 'unitFlag anchor miss: %d' % s.count(old2)
    s = s.replace(old2, 'ctx.mp.unitFlag = 0b11u;              // [ABL] 唯一一片就是链尾', 1)
if MODE in ('cubefx', 'cubenull'):
    lines = s.split('\n')
    hit = [i for i, ln in enumerate(lines) if ln.lstrip().startswith(FX_LINE)]
    assert len(hit) == 1, 'Fixpipe 命中 %d 次（cube 回程应恰好 1 条）' % len(hit)
    i = hit[0]
    ind = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = ind + '// [ABL] 每片 Fixpipe 已删（档 %s）' % MODE
    s = '\n'.join(lines)

assert 'struct CubeCtx' in s, '结构自检失败：CubeCtx 被删（锚点漂移）'
d = n0 - s.count('\n')
sys.stderr.write('[ABL] mode=%s 净删 %d 行 (0..12)\n' % (MODE, d))
assert 0 <= d <= 12, 'mode=%s 删了 %d 行 ⇒ 越界' % (MODE, d)
sys.stdout.write(s)
