#!/bin/bash
# P107 探针发次的闸门：把**要提交的那一份 sed**（不是 env 旋钮版）打进远端副本，
# 建一次、量一次"该动的动 / 不该动的逐字节不动"、再跑正确性。
# 用法：bash "code 3/probes/p107_gate.sh"
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1200}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
# 探针那一行 = 唯一改动；本地/远端/dry-run 三方哈希必须都等于这一式的结果
SEDEXP="s|    tiling->n_blk = nBlk;|    tiling->n_blk = (static_cast<uint64_t>(B) * Q_S <= 40u) ? 16u : nBlk;|"
GOLD="r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads p1 p2 p4 p6 big1 pm41 pm81 pm161"
EXPC="q1h q2h q3h p_n64 p_n512 p_n1024 e1empty e2one e3two e4odd e5s2one e6many e7padq e8padkv"

# 0) 本地先把期望哈希算出来（对一份临时副本打同一条 sed）
mkdir -p /tmp/p107h && cp "$REPO/code 3/code/op_host/sparse_flash_attention.cpp" /tmp/p107h/
sed -i "$SEDEXP" /tmp/p107h/sparse_flash_attention.cpp
EXP=$(sha256sum /tmp/p107h/sparse_flash_attention.cpp | cut -d' ' -f1)
echo ">>> 本地期望 host(探针态) sha256 = $EXP"
echo ">>> sed 命中数 = $(grep -c 'static_cast<uint64_t>(B) \* Q_S <= 40u' /tmp/p107h/sparse_flash_attention.cpp)（必须是 1）"

bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
# 1) 远端：SoC sed（只为本机建得起来）+ 探针 sed
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  sed -i '$SEDEXP' op_host/sparse_flash_attention.cpp && \
  echo HITS_PROBE=\$(grep -c 'Q_S <= 40u' op_host/sparse_flash_attention.cpp) \
       HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp) && \
  sha256sum op_host/sparse_flash_attention.cpp" | tail -3
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"

# 2) GATE-A 选档/计时自证：该动的动（p1/p6/big1...）、不该动的逐字节不动（w3/w4）
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; break
  fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
echo "########## GATE-A 探针态计时（对照 p107_calib.txt 的 off 列） ##########"
nssh "$ENVR; for cs in p1 p4 p6 w1 big1 w3 w4; do t=\$(./test_sfa_dev cases/\$cs.bin 5 none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '%-8s %10s\n' \$cs \"\$t\"; done" 2>&1 | grep -av "^Warning"

# 3) GATE-B 正确性（双 dtype）：判据只看 超差（golden 逐位一致只在 chunk 数不变的档上成立）
for f in 0 1; do
  tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
  echo "########## GATE-B $tag ##########"
  nssh "$ENVR; for cs in $GOLD $EXPC; do line=\$(env $([ "$f" = 1 ] && echo SFA_F32=1) ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+' | tr '\n' ' '); printf '%-10s %s\n' \$cs \"\$line\"; done" 2>&1 | grep -av "^Warning"
done
echo "P107_GATE_DONE  期望哈希=$EXP"
