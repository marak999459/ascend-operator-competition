#!/bin/bash
# P108 探针发次的闸门：要提交的那一份 sed = `rows<=40 ? 8u : nBlk`。
# 与 p107_gate.sh 的差别只有三件事：① 值从 16 换成 **8**（在 NBLK_MIN=16 之下 ⇒ 平台的
# 任何一档都 >8，所以它对 rows<=40 的点是**可证明非惰性**的，这正是 P107 那发缺的东西）；
# ② GATE-A 把 rows>=41 的三个档（big1/w3/w4）当成**同一发内的负对照**（必须严格 0.0 %）；
# ③ GATE-B 去掉 pm41/pm81/pm161 —— 那三档的 expect 是哑零（§1 表里 w3/w4/p1s* 同一族），
#    两份不同构建的"超差 6858/8192"逐字节相同 ⇒ 它们只能当时形档，别拿去看精度。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-1200}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
SEDEXP="s|    tiling->n_blk = nBlk;|    tiling->n_blk = (static_cast<uint64_t>(B) * Q_S <= 40u) ? 8u : nBlk;|"
GOLD="r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads p1 p2 p4 p6 big1 w3"
EXPC="q1h q2h q3h p_n64 p_n512 p_n1024 e1empty e2one e3two e4odd e5s2one e6many e7padq e8padkv"

mkdir -p /tmp/p108h && cp "$REPO/code 3/code/op_host/sparse_flash_attention.cpp" /tmp/p108h/
sed -i "$SEDEXP" /tmp/p108h/sparse_flash_attention.cpp
EXP=$(sha256sum /tmp/p108h/sparse_flash_attention.cpp | cut -d' ' -f1)
echo ">>> 本地期望 host(探针态) sha256 = $EXP"
echo ">>> sed 命中数 = $(grep -c 'Q_S <= 40u) ? 8u' /tmp/p108h/sparse_flash_attention.cpp)（必须是 1）"
echo ">>> 提交源基线 host = $(sha256sum "$REPO/code 3/code/op_host/sparse_flash_attention.cpp" | cut -c1-12)（须为 P105 的 3a8f5305…）"

bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  sed -i '$SEDEXP' op_host/sparse_flash_attention.cpp && \
  echo HITS_PROBE=\$(grep -c 'Q_S <= 40u' op_host/sparse_flash_attention.cpp) \
       HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" | tail -2
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"

for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; ALIVE=1; break
  fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
[ "${ALIVE:-0}" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }

echo "########## GATE-A 探针态计时（正对照 p1/p4/p6/w1 应大涨；负对照 big1/w3/w4 须 0.0 %） ##########"
nssh "$ENVR; for cs in p1 p4 p6 w1 big1 w3 w4; do t=\$(./test_sfa_dev cases/\$cs.bin 5 none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '%-8s %10s\n' \$cs \"\$t\"; done" 2>&1 | grep -av "^Warning"

for f in 0 1; do
  tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
  echo "########## GATE-B $tag ##########"
  nssh "$ENVR; for cs in $GOLD $EXPC; do line=\$(env $([ "$f" = 1 ] && echo SFA_F32=1) ./test_sfa_dev cases/\$cs.bin 1 diff 2>&1 | grep -aoE '超差 [0-9]+/[0-9]+' | tr '\n' ' '); printf '%-10s %s\n' \$cs \"\$line\"; done" 2>&1 | grep -av "^Warning"
done
echo "P108_GATE_DONE  期望哈希=$EXP"
