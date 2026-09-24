#!/bin/bash
# P113 = 造"平台的离线代理"：三臂（base / 计算×2 / gather×2）跑在 S2=131072 那一族上。
#
# 为什么是这一发：P112/P114 两个**逐位惰性**计量器在平台读出 计算×2 = +6.1 %、gather×2 = +17.8 %，
# 在本地 w3/w4/big1/w2/p6 读出 +44~50 % 与 +3.6~6.1 % ⇒ 两侧不同型，本地档位不能替平台说话。
# 判据（事先写死）：某个 case 上 **两臂 Δ% 同时**落进平台的两个数（计算 ≤10 % 且 gather ≥14 %）
# ⇒ 那一族就是代理，之后所有优化先在家里过；只有 gather 对上了 ⇒ 代理只覆盖搬运侧。
# ⛔ 全程不发次。补丁只落远端副本；退出 trap = 只做一次 npu.sh sync（不 build、不 sed）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

# 靶族：S2=131072（KV 池 285 MB，装不进缓存）+ sbs=1 + rows≥41；末尾两枚是"已知不同型"的对照
CASES="${CASES:-r64c65 b32n8s1 r32c65 p32s1c16384 big1 p6}"
REPS="${REPS:-3}"
PRISTINE="$REPO/code 3/probes/backup/p114_pre_probe/sparse_flash_attention.cpp"

restore() {
  echo "=== 还原：只做 sync（远端源码 = 本地提交字节，无 sed、无 build） ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sha256sum op_kernel/sparse_flash_attention.cpp | cut -c1-16; echo SEDHITS=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning | tail -2
}
trap restore EXIT

sha_local=$(sha256sum "$PRISTINE" | cut -c1-16)
cat "$PRISTINE" | nssh "cat > ~/sfa_real/kernel_pristine.cpp"
( cd "$REPO/code 3/probes" && tar cf - p110_score_rep.py p114_gather_rep.py ) | nssh "cd ~/sfa_real && tar xf -"
nssh "cd ~/sfa_real && sha256sum kernel_pristine.cpp | cut -c1-16" 2>&1 | grep -av Warning | tail -1
echo ">>> pristine sha 本地=$sha_local 远端应同"

for ARM in base score gather; do
  echo "########## ARM $ARM ##########"
  case "$ARM" in
    base)   GEN="python3 p114_gather_rep.py 1 code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" ;;
    score)  GEN="python3 p110_score_rep.py  2 code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" ;;
    gather) GEN="python3 p114_gather_rep.py 2 code/op_kernel/sparse_flash_attention.cpp kernel_pristine.cpp" ;;
  esac
  nssh "cd ~/sfa_real && $GEN" 2>&1 | grep -av Warning | tail -1
  nssh "cd ~/sfa_real && echo SED_TMP=\$(grep -c ascend910_93 code/op_host/sparse_flash_attention.cpp) && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' code/CMakeLists.txt && \
    grep -q ascend910_93 code/op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' code/op_host/sparse_flash_attention.cpp" >/dev/null
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3"
  ALIVE=0
  for try in 1 2 3 4 5 6; do
    if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 1 diff" 2>&1 | grep -qa 超差; then ALIVE=1; break; fi
    echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
  done
  [ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止（不要归因成改动挂了，§7.8）"; exit 1; }
  # 惰性自证 = **跨臂逐位比对**：base 臂把输出写进一次性目录 golden_p113/（⛔ 不是 durable 的
  # golden/，§7.3 那条"计时档绝不 write"讲的是前者），后两臂与它逐位比 ⇒ 冷大档上也拿到证据，
  # 而不只在 p6/big1 上。顺带打 header/pick ⇒ 形状是读来的、不是手抄的。
  ACT=write; [ "$ARM" = "base" ] || ACT=diff
  nssh "$ENVR; for c in $CASES; do printf '%-13s ' \$c; ./test_sfa_dev cases/\$c.bin 1 $ACT 2e-3 1e-2 golden_p113 2>&1 | grep -aE '^case=|pick\[|超差|逐位一致|不\*\*逐位一致\*\*|golden 写入|==>' | tr '\n' '|'; echo; done" 2>&1 | grep -av "^Warning" | tee "/tmp/p113_${ARM}_inert.txt"
  nssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    export ASCEND_CUSTOM_OPP_PATH=\$HOME/sfa_real/vendor/custom; \
    export LD_LIBRARY_PATH=\$HOME/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
    for c in $CASES; do t=\$(timeout 1800 ./test_sfa_dev cases/\$c.bin $REPS none 2>&1 | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//'); printf '  %-14s %10s ms\n' \$c \"\$t\"; done" 2>&1 | grep -av "^Warning" \
    | tee "/tmp/p113_${ARM}_time.txt"
done
