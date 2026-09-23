#!/usr/bin/env python3
# P57 诊断：把 M1d 的三条跨核旗标按【侧】剪掉，用来把"卡在旗标"与"卡在 AIC 自己的
# MMA/Fixpipe 里"分开。只作用在远端副本 ~/sfa_real/code —— 本地提交源一行都不动。
#   cut aiv  = 只剪 AIV 侧（读环前的 READY wait + FlushChunk 尾的 CRED set）
#   cut aic  = 只剪 AIC 侧（CRED wait + READY set + 收工 drain）
#   cut all  = 两侧全剪 ⇒ 整个启动期没有任何跨核通信
# 判据（三档联合读数，见 §15.72）：
#   all 通、aiv 通、aic 卡  ⇒ AIC 自己的计算路径有问题（MMA/Fixpipe/InitBuffer/越界）
#   all 通、aiv 卡          ⇒ 旗标【路由】不对：AIC 的 set 到不了它在等的 ID
#   all 卡                  ⇒ 与旗标无关，是 AIC 侧死循环 / 计算陷阱
import sys

path, mode = sys.argv[1], sys.argv[2]
lines = open(path, encoding='utf-8').read().split('\n')
na = nv = 0
for i, ln in enumerate(lines):
    s = ln.strip()
    if not (s.startswith('CrossCoreSetFlag<') or s.startswith('CrossCoreWaitFlag<')
            or 'CrossCoreSetFlag<' in s or 'CrossCoreWaitFlag<' in s):
        continue
    if not s.endswith(';'):
        continue                                   # 注释里提到的名字，跳过
    aiv = 'sub_' in s                              # AIV 侧的两条都带下划线
    drop = (mode == 'all') or (mode == 'aiv' and aiv) or (mode == 'aic' and not aiv)
    if not drop:
        continue
    if aiv:
        nv += 1
    else:
        na += 1
    lines[i] = '// [CUT] ' + ln
open(path, 'w', encoding='utf-8').write('\n'.join(lines))
print('CUT mode=%s aic=%d aiv=%d' % (mode, na, nv))
