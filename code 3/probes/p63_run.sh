#!/bin/bash
# P63 驱动：把 p63_patch.py 打到的遥测内核在真机上跑一遍，读回两侧的"累计片数"。
#
# 判据不是数值、不是 rc，而是**记录本身**（AIC 的 cp_ 对两颗 AIV 的 aChunk_）。
# 补丁与解码脚本都从本地推远端；跑完恢复源码 + 重建成净版（不留探针态）。
# ⚠️ test_sfa_dev 的 act=write 会把输出写进 golddir ⇒ 显式指到 /tmp/p63_gold，绝不碰 golden/。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
GOLD=/tmp/p63_gold
CASES="${CASES:-r6_multiB p1}"
TMO="${TMO:-60}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p63_orig.cpp; md5sum /tmp/kernel_p63_orig.cpp"

echo "--- PUSH ---"
( cd "$(dirname "$0")" && tar cf - p63_patch.py p63_decode.py ) | \
  timeout 180 ssh -F $S $H "cd ~/sfa_real && tar xf -" 2>&1 | grep -av Warning

echo "--- PATCH ---"
ns "$ENVR; python3 p63_patch.py"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p63_build.log 2>&1; echo build_rc=\$?; grep -m8 -E 'error|Error' /tmp/p63_build.log | head -8; tail -2 /tmp/p63_build.log")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 跑起来用的是上一个二进制，读数无意义 ⇒ 直接恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p63_orig.cpp $K; md5sum $K; rm -rf $GOLD p63_patch.py p63_decode.py"
     ns "$ENVR; bash build.sh > /tmp/p63_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P63_ABORT"; exit 1;;
esac

for cs in $CASES; do
  echo "--- RUN $cs ---"
  ns "$ENVR; rm -rf $GOLD; mkdir -p $GOLD; timeout $TMO ./test_sfa_dev cases/$cs.bin 1 write 2e-3 1e-2 $GOLD > /tmp/p63_$cs.log 2>&1; echo rc=\$?; ls -la $GOLD"
  ns "$ENVR; python3 p63_decode.py $GOLD"
done

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p63_orig.cpp $K; md5sum $K; rm -rf $GOLD p63_patch.py p63_decode.py"
echo "--- REBUILD CLEAN ---"
ns "$ENVR; bash build.sh > /tmp/p63_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P63_DONE"
