#!/bin/bash
# P64：p63 实测把"片数对不上"这条判死了 —— 现在只剩"旗标号是运行时算出来的"这一条。
#
# p63 真机读数（远端副本，门关 + AIV 不等 READY + 两侧各自打点）：
#   r6_multiB  每组：AIV sub0 telSets_=1、AIV sub1 telSets_=1、AIC cp_=1、N1=2 nb=1
#   p1         每组：AIV sub0 telSets_=43、AIV sub1 telSets_=43、AIC cp_=43、N1=4 nb=2
#   ⇒ 两侧片数【逐块相等】，两条 credit 线各自都真的走了 43 次 set。
# 把六档实测摆成一张表，唯一还在起作用的自变量就是"set 的号是常量还是算出来的"：
#   set 号          等待数                 结果
#   常量 6          收尾 1 次              通   (p60 free6)
#   运行时 5/7      收尾 1 次(等 5)        挂   (p60 free5)
#   运行时 6+2*sub  每片每线 1 次          挂   (p61 cred6)
#   运行时 5/7      每片每线 1 次          挂   (完整协议)
#   常量 6          每片 2 次              挂   (p61 dbl)
#   无 set          收尾 1 次              挂   (p61 nostale ⇒ free6 的"通"是真到达)
# ⇒ 决定性一发 `c57`：**只把 set 的号换成编译期常量**（`if (sub_==0) set(5) else set(7)`），
#   其余（门关不关、片数、每片每线各等一次、收尾配平）与完整协议【逐字节相同】。
#   通 ⇒ 运行时号这条路在 arch22 的 ffts 上就是不落地，改常量分支即可收尾；
#   挂 ⇒ 常量号也救不了"每片两等"，方向转去 p64b（单线单等，sub0/sub1 各验一条）。
# ⚠️ c57 是配平档（等 86 到 86）⇒ 不会给下一次启动留残值，后面接别的档也不受污染。
# ⚠️ 补丁只打在远端 ~/sfa_real 上，跑完一定 RESTORE + 重建成净版；本地提交源不动。
# ⚠️ 判据只有 rc：这一档的输出注定是错的（门关/AIV 不等 READY 的形态没有），超差不看。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
VARIANT="${1:-c57}"
CASES="${CASES:-r6_multiB p1}"
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p64_orig.cpp; md5sum /tmp/kernel_p64_orig.cpp"

echo "--- PATCH ($VARIANT) ---"
ns "$ENVR; python3 - $VARIANT <<'PYX'
import sys, io
v = sys.argv[1]
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open('/tmp/kernel_p64_orig.cpp', encoding='utf-8').read()

CSET = '            CrossCoreSetFlag<2, PIPE_MTE3>(sfa::CF_CRED + 2u * sub_);'

def one(t, a, n):
    assert t.count(a) == 1, '%s x%d' % (n, t.count(a))
    return t

if v == 'c57':
    new = ('            if (sub_ == 0u) {                     // [p64] 号换成编译期常量\n'
           '                CrossCoreSetFlag<2, PIPE_MTE3>(5u);\n'
           '            } else {\n'
           '                CrossCoreSetFlag<2, PIPE_MTE3>(7u);\n'
           '            }')
    out = one(src, CSET, 'cset').replace(CSET, new, 1)
elif v == 'c57plus':
    # 同 c57，但连 CF_READY 也一并写成常量 4u（set 侧）—— READY 那条本来就是常量号，
    # 这一档只是把"两处都常量"钉死，防止"运行时号会污染同一条 inline asm 的其他分支"。
    new = ('            if (sub_ == 0u) {                     // [p64]\n'
           '                CrossCoreSetFlag<2, PIPE_MTE3>(5u);\n'
           '            } else {\n'
           '                CrossCoreSetFlag<2, PIPE_MTE3>(7u);\n'
           '            }')
    t = one(src, CSET, 'cset').replace(CSET, new, 1)
    A_SET = '        CrossCoreSetFlag<2, PIPE_FIX>(sfa::CF_READY);'
    out = one(t, A_SET, 'aset').replace(A_SET, '        CrossCoreSetFlag<2, PIPE_FIX>(4u);', 1)
else:
    raise SystemExit('unknown variant ' + v)
io.open(p, 'w', encoding='utf-8').write(out)
print('PATCHED_%s lines %d -> %d' % (v.upper(), len(src.splitlines()), len(out.splitlines())))
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p64_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p64_build.log | head -6; tail -2 /tmp/p64_build.log")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 读数是上一个二进制的 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p64_orig.cpp $K; md5sum $K"
     ns "$ENVR; bash build.sh > /tmp/p64_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P64_ABORT"; exit 1;;
esac

for cs in $CASES; do
  ns "$ENVR; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 diff > /tmp/p64_$cs.log 2>&1; rc=\$?; \
      out=\$(grep -aoE '超差 [0-9]+/[0-9]+|PASS|FAIL|errorStr: [A-Za-z0-9_ .:-]+' /tmp/p64_$cs.log | sort -u | tr '\n' ' '); \
      printf '%-12s rc=%-4s %s\n' $cs \$rc \"\$out\""
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p64_orig.cpp $K; md5sum $K"
echo "--- REBUILD CLEAN ---"
ns "$ENVR; bash build.sh > /tmp/p64_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P64_DONE"
