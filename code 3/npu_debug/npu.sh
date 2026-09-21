#!/bin/bash
# 第三题 SFA —— 真机（第二账号 910B）通道
# 用法: npu.sh <sync|build|reg|bench|probe> [args...]
set -u

HOST_ALIAS="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
SSH_CFG="$HOME/.atomgitdevenv/.ssh/config"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
RREM="sfa_real"          # 远端 ~ 下的工作区

nssh() { timeout "${TMO:-300}" ssh -F "$SSH_CFG" -o ConnectTimeout=25 "$HOST_ALIAS" "$@"; }
# 把本地目录/文件流式推到远端 ~ （tar 管道：隧道下比 scp 稳，改行尾风险也小）
npush() { ( cd "$REPO" && tar cf - "$@" ) | nssh "cd ~ && tar xf -"; }

MODE="${1:-sync}"; shift || true

case "$MODE" in
  sync)
    mkdir -p "$REPO/code 3/npu_debug/logs"
    {
      echo "=== 1) 推提交源 code 3/code -> ~/$RREM/code ==="
      mkdir -p "/tmp/sfa_stage/$RREM" && rm -rf "/tmp/sfa_stage/$RREM/code"
      cp -r "$REPO/code 3/code" "/tmp/sfa_stage/$RREM/code"
      ( cd /tmp/sfa_stage && tar cf - "$RREM" ) | nssh "cd ~ && tar xf -"
      echo "=== 2) 推 harness / 参考实现 ==="
      ( cd "$REPO/refs/sfa" && tar cf - build.sh run.sh test_sfa_real.cpp sfa_ref.py \
            bench.cpp gen_big.py gen_case.py ) | nssh "cd ~/$RREM && tar xf -"
      ( cd "$REPO/code 3/npu_debug" && tar cf - test_sfa_dev.cpp ) | nssh "cd ~/$RREM && tar xf -"
      echo "=== 3) md5 复验（本地提交源 vs 远端） ==="
      md5sum "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" \
             "$REPO/code 3/code/op_host/sparse_flash_attention.cpp" \
             "$REPO/code 3/code/op_kernel/sparse_flash_attention_tiling.h" \
             "$REPO/code 3/code/op_kernel/tiling_key_sparse_flash_attention.h" | sed "s#$REPO/##"
      nssh "cd ~/$RREM && md5sum code/op_kernel/sparse_flash_attention.cpp \
              code/op_host/sparse_flash_attention.cpp \
              code/op_kernel/sparse_flash_attention_tiling.h \
              code/op_kernel/tiling_key_sparse_flash_attention.h"
    } 2>&1 | tee -a "$REPO/code 3/npu_debug/logs/sync_$(date +%H%M%S).log"
    ;;

  # 本机 SoC = ascend910_93，而提交源只注册 ascend910b（比赛平台口径）。
  # ⚠️ 只在**远端副本**上打这个补丁，每次 sync 后重打；本地提交源保持 ascend910b 不动。
  build)
    nssh "cd ~/$RREM && \
      sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ${SOC:-ascend910_93})/' code/CMakeLists.txt && \
      sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"${SOC:-ascend910_93}\")|' code/op_host/sparse_flash_attention.cpp && \
      grep -c 'AddConfig' code/op_host/sparse_flash_attention.cpp && \
      bash build.sh && \
      g++ -std=c++17 -O2 bench.cpp -o test_bench \
        -I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include \
        -I\$HOME/sfa_real/vendor/custom/op_api/include \
        -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 \
        -L\$HOME/sfa_real/vendor/custom/op_api/lib \
        -lascendcl -lnnopbase -lcust_opapi 2>&1 | head -5 ; \
      ls -la test_bench" 2>&1 | tail -40
    ;;

  # 正确性回归：GEN=0 用已有用例；传用例名则只跑这些
  reg)
    nssh "cd ~/$RREM && GEN=${GEN:-0} bash run.sh $*" 2>&1 | tail -40
    ;;

  # 生成用例（在远端用 sfa_ref.py）
  gencase)
    nssh "cd ~/$RREM && GEN=1 bash run.sh" 2>&1 | tail -30
    ;;

  bigcase)
    nssh "cd ~/$RREM && mkdir -p cases && python3 gen_big.py ${1:-big1} ${2:-256}" 2>&1 | tail -5
    ;;

  bench)
    nssh "cd ~/$RREM && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
      export ASCEND_CUSTOM_OPP_PATH=~/sfa_real/vendor/custom; \
      export LD_LIBRARY_PATH=~/sfa_real/vendor/custom/op_api/lib:\$LD_LIBRARY_PATH; \
      ./test_bench cases/${CASE:-big1}.bin ${REPS:-5}" 2>&1 | tail -12
    ;;

  *)
    # 自由探针：npu.sh probe '<远端命令>'
    nssh "$@" 2>&1 | tail -60
    ;;
esac
