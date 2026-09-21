#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P19 瓶颈定位探针（任务 #35）：平台形状（`p1/p4`，nb=1/32-token chunk）那 166 µs 到底花在
哪一段 —— 把 `FlushChunk` 里的五段计算**逐段删掉**重测，差分就是该段的**边际上界**。

为什么还要再量一遍：§15.10(b) 那个"score 占 71 %"是 **big1 + nb=8** 的账，而平台 6 点是
`nb=1、n_blk=32`，调用数拆分（score 22 / softmax≈6 / PV 32）暗示**大头是 PV**（M2 的靶子）。
M1 还是 M2 先做，取决于这一屏，不取决于我的记忆。

用法: mk_probe_abl.py <mode>   —— 把探针 kernel 打到 stdout（**只推远端副本**）
  full      不删（同场次基准）
  nosc      删 `ComputeScores`               → score 段（含 K/K-rope 的 fp32 展开与两级归约）
  nomax     删 SoftmaxPv 的「1) 行最大」
  noexp     删「2) P = exp(S-mNew)」          → vexp 那一趟
  nosum     删「3) ΣP + alpha」+「4) O 重缩放 + l 更新」
  nopv      删「5) PV 累加」                  → 32 条 Axpy + 一组 V 加宽
  nocalc    全删（只剩 gather + 标量索引扫描 + WriteOut）= 这条流水的**地板**
  —— 以下是 #36「地板再分解」那一组（P19 第二轮，全都在 `nocalc` 之上再砍搬运）——
  noK       只删 K 的那条 `CopyGm2Ub`         → K 的 MTE 份额（score 读的是脏 UB）
  noV       只删 V 的那条 `CopyGm2Ub`         → V 的 MTE 份额
  noMTE     删 K/V/K-rope 三条                → "整条流水不碰 GM 读" 的地板
  noW       删 `WriteOut` + `MergeToken` 调用  → "整条流水不写 GM" 的地板
⚠️ 判据只有时间：这些档的输出**必然是错的** ⇒ 一律 `act=none`，**绝不 `act=write`**（探针纪律：
   哑 expect 的计时用例不许锁进 golden）。删掉计算后编译器可能顺手把"只被计算读的缓冲"的搬运
   也优化掉 ⇒ 差分是**上界**，`nocalc` 那格才是搬运+扫描的下界。
"""
import io
import os
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else 'full'
ALL = ('full', 'nosc', 'nomax', 'noexp', 'nosum', 'nopv', 'nocalc',
       'noK', 'noV', 'noMTE', 'noW')
assert MODE in ALL, MODE

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'code')
SRC = os.path.join(BASE, 'op_kernel', 'sparse_flash_attention.cpp')
s = io.open(SRC, encoding='utf-8').read()


def cut(text, start_marker, end_marker, tag):
    """删掉 [start, end) 这一段（end 不删，因为它是下一段的标题）。"""
    i = text.find(start_marker)
    assert i >= 0, 'start marker miss: ' + tag
    j = text.find(end_marker, i)
    assert j > i, 'end marker miss: ' + tag
    return text[:i] + ('    // [ABL] %s 段已删（探针档 %s）\n' % (tag, MODE)) + text[j:]


R_SC = '        ComputeScores(q, kb, kr, sc, nbCur, m);\n'
M1 = '// ---- 1) 行最大'
M2 = '// ---- 2) P = exp'
M3 = '// ---- 3) ΣP'
M4 = '// ---- 4) O 重缩放'
M5 = '// ---- 5) PV 累加'
MEND = '    TPipe pipe_;'
# 5) 是 SoftmaxPv 的最后一段 ⇒ 尾巴上没有"下一段标题"可挂，只能挂到**函数收尾的那个 `    }`**。
# 直接拿 `    TPipe pipe_;` 当 end marker 会把函数闭合花括号一起吃掉（nopv 首跑就是这么挂的：
# op_kernel:1045 起满屏 "shadows template parameter" —— 类结构塌了）。
MCLOSE = '\n    }\n\n' + MEND

CALCLESS = MODE in ('nocalc', 'noK', 'noV', 'noMTE', 'noW')   # 这些档 = 地板再分解
drop_sc = MODE in ('nosc',) or CALCLESS
drop_max = MODE in ('nomax',) or CALCLESS
drop_exp = MODE in ('noexp',) or CALCLESS
drop_sum = MODE in ('nosum',) or CALCLESS
drop_pv = MODE in ('nopv',) or CALCLESS

if drop_sc:
    assert s.count(R_SC) == 1, 'score call anchor miss'
    s = s.replace(R_SC, '        // [ABL] ComputeScores 已删（档 %s）\n' % MODE, 1)
if drop_max:
    s = cut(s, M1, M2, '行最大')
if drop_exp:
    s = cut(s, M2, M3, 'P=exp')
if drop_sum:
    # 3) 与 4) 一起删：4) 读 3) 的 r0/av，只删一半会留一堆未定义变量
    s = cut(s, M3, M5, 'ΣP+alpha+O 重缩放')
    # 删完 3)+4) 之后 5) 还在，但 5) 要用 ml/r0？不 —— 它只用 p 与 vf
if drop_pv:
    s = cut(s, M5, MCLOSE, 'PV 累加')


def drop_line(text, line, tag):
    assert text.count(line) == 1, 'line anchor miss: ' + tag
    return text.replace(line, '        // [ABL] %s 已删（档 %s）\n' % (tag, MODE), 1)


# ---- #36：地板再分解。全部档都在"计算已清空"的地板之上再砍搬运，所以
#      `nocalc − 本档` = 这条搬运在**串行暴露**下的真实份额（计算在场时它多半被
#      MTE↔V 的双缓冲重叠吃掉一部分，那个数用地板差分是量不出来的）。
L_K = '                CopyGm2Ub(kb[done * D_],  kGm_[kOff],  run * D_);\n'
L_V = '                CopyGm2Ub(vb[done * D_],  vGm_[kOff],  run * D_);\n'
L_KR = '                CopyGm2Ub(kr[done * Dr_], krGm_[rOff], run * Dr_);\n'
L_WO = '            WriteOut(o, ml, lse, s1Base, lseBase + n0, n0, nbCur);\n'
L_MG = '            MergeToken(b, s, hb);\n'
if MODE in ('noK', 'noMTE'):
    s = drop_line(s, L_K, 'K 的 CopyGm2Ub')
if MODE in ('noV', 'noMTE'):
    s = drop_line(s, L_V, 'V 的 CopyGm2Ub')
if MODE == 'noMTE':
    s = drop_line(s, L_KR, 'K-rope 的 CopyGm2Ub')
if MODE == 'noW':
    s = drop_line(s, L_WO, 'WriteOut')
    s = drop_line(s, L_MG, 'MergeToken')

sys.stdout.write(s)
