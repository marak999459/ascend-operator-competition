#!/bin/bash
# P126 闸门：本地提交源已钉档（p126_pin.py apply）-> npu.sh sync -> 远端 build（SoC sed 只落在远端副本）
# -> 探活 -> **命中数门** -> 全量数值门（复用 p32_gate.sh：13 golden 双 dtype + 13 真 expect）
# -> 顺手记三档时间（与 P109 的 FORCE_NB 臂对表，坐实"这一发在本地不是空操作"）。
# ⛔ 本脚本不做"最后一次 sync"，也不 submit —— 那是 p126_precheck.sh / p126_submit.sh 的活。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

echo "########## ① sync 钉档字节到远端 ##########"
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | grep -aE "md5|=== |OK|FAIL" | tail -8

echo "########## ② 命中数门 + kernel 三枚 sha（远端副本，sed 之前） ##########"
nssh "cd ~/sfa_real/code && \
  echo PIN=\$(grep -c 'NB_CAND\[\] = {1}' op_host/sparse_flash_attention.cpp) \
       NBLINES=\$(grep -c NB_CAND op_host/sparse_flash_attention.cpp) \
       OLDTAB=\$(grep -c '{32, 16, 8, 4, 2, 1}' op_host/sparse_flash_attention.cpp) ; \
  sha256sum op_host/sparse_flash_attention.cpp op_kernel/sparse_flash_attention.cpp \
            op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h | cut -c1-16" 2>&1 | grep -av Warning
echo ">>> 期望 PIN=1 NBLINES=4 OLDTAB=0 ；host sha=3969800637a4ce58，kernel 三枚 = f815bf1e / f2b28a86 / 1046b349"

echo "########## ③ SoC sed（只在远端）+ build ##########"
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  echo HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5" 2>&1 | grep -av Warning

echo "########## ④ 探活（§7.8：不通就中止，别带着坏构建跑完整批门） ##########"
ALIVE=0
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; ALIVE=1; break
  fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
[ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }

echo "########## ⑤ 三档时间（对表 P109 的 FORCE_NB 臂；REPS=${REPS:-5}） ##########"
for c in ${TIMCASES:-w3 big1 p6 r64c65}; do
  line=$(nssh "$ENVR; ./test_sfa_dev cases/$c.bin ${REPS:-5} none 2>&1" 2>&1 | grep -a 批量口径 | head -1)
  printf '%-9s %s\n' "$c" "$(printf '%s' "$line" | sed 's/.*平均 /平均 /; s/ *最小.*//')"
done

echo "########## ⑥ 全量数值门（p32_gate.sh：13 golden 双 dtype + 13 真 expect 双 dtype） ##########"
bash "$REPO/code 3/probes/p32_gate.sh" 2>&1 | tee "$REPO/code 3/probes/p126_gate.txt" | grep -av Warning
echo "P126_GATE_DONE"
