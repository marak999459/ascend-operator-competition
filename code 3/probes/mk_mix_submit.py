#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""#31：把提交源 `code 3/code/` 打进 **最小 MIX 形态**（真·提交源，不是远端副本探针）。

与 `mk_probe_mix.py` 的分工：那一支是**探针**，只写远端副本、带 getenv/printf 调试打印；
这一支是**提交候选**，直接改本地四文件里的两处，产物里**一个 `printf`/`getenv` 都不许有**
（§3.2 第 3 条的禁用词闸门会当场拒收）。

四条目全部来自 §15.37(b) 已钉死的配方，没有新发明：
  K1  `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` + `matmul::clearWorkspace` 空桩
  K2  `coreNum = GetBlockNum() * 2`（MIX 下 `GetBlockNum()` = **组数** BD，
      AIV 的 `GetBlockIdx()` = `group*2 + sub ∈ [0, 2·BD)`；ratio **写死 2** 不查
      `GetTaskRatio()` —— AIC 侧它返回 1，两档核数会在同一份 tiling 下算歪）
  K3  AIC 分支 `ks_ = 1u`：本探针里 AIC 不产任何 tile，让它从 `Process()` 的
      `if (ks_ < 2u) return;` 处退出，**绝不进 `SyncAll<true>()`**（那是纯 AIV 硬件屏障，
      AIC 调它会等一块不存在的旗标）⇒ 即 §15.37 的 `mxa` 形态
  H1  `SetBlockDim(ceil(AIV块/2))`：折成**组数**口径，让一次 launch 起回原来那么多 AIV 块

用法：
  python3 mk_mix_submit.py check    # 只验锚点 + 花括号平衡 + 禁用词，不写盘
  python3 mk_mix_submit.py apply    # 写盘（先自己备份到 probes/backup/，见 -- 下方）
  python3 mk_mix_submit.py revert   # 从 probes/backup/<stamp>/ 还原（打印还原后的 md5）
"""
import hashlib
import io
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, '..', 'code'))
KER = os.path.join(CODE, 'op_kernel', 'sparse_flash_attention.cpp')
HOST = os.path.join(CODE, 'op_host', 'sparse_flash_attention.cpp')
BKROOT = os.path.join(HERE, 'backup')
STAMP = os.path.join(BKROOT, 'p49_p38_clean')
# 提交内容闸门（§3.2 第 3 条）—— 命中即拒绝落盘
FORBID = re.compile(r'printf|fprintf|fflush|std::cout|\bcout\b|TODO|FIXME|#if 0|调试|getenv')

# 还原备份里存的两份原文件名
BAK = {'ker': 'sparse_flash_attention.cpp.kernel', 'host': 'sparse_flash_attention.cpp.host'}


def rd(p):
    return io.open(p, encoding='utf-8').read()


def md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def braces(s):
    return s.count('{') - s.count('}')


def patch_kernel(s):
    inc = '#include "kernel_operator.h"\n'
    assert s.count(inc) == 1, 'K1a include anchor: %d' % s.count(inc)
    s = s.replace(inc, inc + (
        '\n// MIX 入口桩会调 matmul::clearWorkspace(workspace)；本算子不用 KFC/Matmul，给空实现\n'
        '// （真清空间的那条框架路径在 arch22 上会把下一次 launch 毒化）。\n'
        'namespace matmul { __aicore__ inline void clearWorkspace(GM_ADDR) {} }\n'), 1)

    ent = '    REGISTER_TILING_DEFAULT(SparseFlashAttentionTilingData);'
    assert s.count(ent) == 1, 'K1b entry anchor: %d' % s.count(ent)
    s = s.replace(ent, '    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);\n' + ent, 1)

    idx = ('        const uint32_t total0  = B_ * S1_ * nHeadBlk_;\n'
           '        const uint32_t coreNum = GetBlockNum();\n'
           '        const uint32_t coreIdx = GetBlockIdx();\n')
    assert s.count(idx) == 1, 'K2 idx anchor: %d' % s.count(idx)
    s = s.replace(idx, (
        '        const uint32_t total0  = B_ * S1_ * nHeadBlk_;\n'
        '        // MIX：GetBlockNum() 给的是【组数】，AIV 块号 = group*2 + sub ⇒ AIV 并行度 = 2·组数\n'
        '        const uint32_t coreNum = static_cast<uint32_t>(GetBlockNum()) * 2u;\n'
        '        const uint32_t coreIdx = GetBlockIdx();\n'), 1)

    aic = ('        if ASCEND_IS_AIC {\n'
           '            unitBegin_ = 0; unitEnd_ = 0; unitStep_ = 1;\n')
    assert s.count(aic) == 1, 'K3 aic anchor: %d' % s.count(aic)
    s = s.replace(aic, aic + '            ks_ = 1u;   // AIC 不进纯 AIV 的 SyncAll\n', 1)
    return s


def patch_host(s):
    a = '        context->SetBlockDim(blockDim);'
    assert s.count(a) == 1, 'H1 blockdim anchor: %d' % s.count(a)
    s = s.replace(a, (
        '        // MIX 的 SetBlockDim 是【组数】口径：1 组 = 1 AIC + 2 AIV；kernel 侧按 2·组数\n'
        '        // 还原 AIV 并行度，所以这里必须向上取整折半（奇数块数时多起一组，空转的那块\n'
        '        // 靠单元循环 unit < total 自然跳过）。\n'
        '        blockDim = (blockDim + 1u) / 2u;\n'
        '        if (blockDim == 0u) { blockDim = 1u; }\n' + a), 1)
    return s


def guarded_apply(name, path, fn):
    src = rd(path)
    b0 = braces(src)
    out = fn(src)
    b1 = braces(out)
    if b1 != b0:
        raise SystemExit('[MIX] %s 花括号不平：%d -> %d，拒绝落盘' % (name, b0, b1))
    hits = FORBID.findall(out)
    if hits:
        raise SystemExit('[MIX] %s 产物含禁用词 %s，拒绝落盘' % (name, sorted(set(hits))))
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    print('[MIX] %-5s %s  md5 %s -> %s  (+%d B)' %
          (name, os.path.relpath(path, CODE), md5(src)[:8], md5(out)[:8],
           len(out) - len(src)))


def do_check():
    for name, path, fn in (('kernel', KER, patch_kernel), ('host', HOST, patch_host)):
        src = rd(path)
        try:
            out = fn(src)
        except AssertionError as e:
            print('[CHECK] %-6s 锚点失配：%s' % (name, e))
            continue
        print('[CHECK] %-6s 锚点全中  花括号 %+d -> %+d  禁用词命中=%s  增 %d B' %
              (name, braces(src), braces(out),
               sorted(set(FORBID.findall(out))) or '无', len(out) - len(src)))


def do_apply():
    if not os.path.isdir(STAMP):
        os.makedirs(STAMP)
        shutil.copyfile(KER, os.path.join(STAMP, BAK['ker']))
        shutil.copyfile(HOST, os.path.join(STAMP, BAK['host']))
        print('[MIX] 原文件已备份 -> %s' % os.path.relpath(STAMP, HERE))
    else:
        raise SystemExit('[MIX] 备份已存在，拒绝二次 apply（先 revert 或换 stamp）')
    guarded_apply('kernel', KER, patch_kernel)
    guarded_apply('host', HOST, patch_host)
    print('[MIX] 下一步：md5 记录 → 推远端 → 双 dtype 闸门 → 禁用词 grep → --dry-run')


def do_revert():
    for key, path in (('ker', KER), ('host', HOST)):
        bak = os.path.join(STAMP, BAK[key])
        src, cur = rd(bak), rd(path)
        with io.open(path, 'w', encoding='utf-8') as f:
            f.write(src)
        print('[BACK] %-6s md5 %s -> %s' % (key, md5(cur)[:8], md5(src)[:8]))
    print('[BACK] 还原后请重跑 §3.2：md5 复验 + 闸门 + grep')


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    {'check': do_check, 'apply': do_apply, 'revert': do_revert}[mode]()
