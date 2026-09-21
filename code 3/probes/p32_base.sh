#!/bin/bash
# P32 A/B 的补齐：p32_ab.sh 的单元格串把"P29 那一档"误吞进了 AUTO 项 ⇒ 单独测 P29 臂基线。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
for pass in 1 2; do
  if [ "$pass" = 1 ]; then order="p1 p2 q1h q2h p4 p6 big1"; else order="big1 p6 p4 q2h q1h p2 p1"; fi
  for cell in $order; do
    case $cell in
      p1|p2|q1h|q2h) nb=2; kb=32; ks=2 ;;
      p4)   nb=4; kb=32; ks=2 ;;
      p6)   nb=4; kb=32; ks=1 ;;
      big1) nb=8; kb=32; ks=1 ;;
    esac
    line=$(timeout 220 ssh -F $S $H "$ENVR; env SFA_FORCE_NB=$nb SFA_FORCE_NBLK=$kb SFA_FORCE_KS=$ks ./test_sfa_dev cases/$cell.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|FAIL|ERROR'" 2>&1 | grep -av Warning | tr '\n' ' ')
    printf 'B%s %-6s %-8s | %s\n' "$pass" "$cell" "$nb/$kb/$ks" "$line"
  done
done
echo P32_BASE_DONE
