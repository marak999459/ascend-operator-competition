#!/bin/bash
# P107 标定 runner：**只在远端副本上打探针补丁**，退出时还原干净构建。
# 用法：bash "code 3/probes/p107_calib.sh"      （REPS=3 可加速）
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    grep -q ascend910_93 op_host/sparse_flash_attention.cpp || \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
  nssh "$ENVR; ./test_sfa_dev cases/p6.bin 3 none" 2>&1 | grep -aE "超差|批量口径|SFA_PICK" | tail -3
}
trap restore EXIT

bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | grep -aE "md5|=== " | tail -10
# ⚠️ sync 会把远端副本铺回**提交态**（只注册 ascend910b）⇒ 本机 SoC=ascend910_93，
#    不重打这两条 sed 的话 build 出来的 .so 注册不上，每一发都 `GetWorkspaceSize` rc=2
#    （形态与 §7.8 的"设备被占住"几乎一样，差别是它有 `[FAIL]` 行可看）。
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  echo HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" | tail -1
( cd "$REPO/code 3/probes" && tar cf - nb_sweep_patch.py p107_patch.py "${BODY:-p107_calib_body.sh}" ) | nssh "cd ~/sfa_real && tar xf -"
# 补丁 + **命中数当门**（P55 的自欺形态：不核命中数就发次）
nssh "cd ~/sfa_real && python3 nb_sweep_patch.py code/op_host/sparse_flash_attention.cpp && \
  python3 p107_patch.py code/op_host/sparse_flash_attention.cpp && \
  echo HITS_SFA_PICK=\$(grep -c 'SFA_PICK' code/op_host/sparse_flash_attention.cpp) \
       HITS_PROBE=\$(grep -c 'SFA_PROBE_ROWS' code/op_host/sparse_flash_attention.cpp) \
       HITS_FORCE=\$(grep -c 'SFA_FORCE' code/op_host/sparse_flash_attention.cpp)" | tail -3
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5"
# 探活门（§7.8：整批 rc=2 可能是设备被残留上下文占住）—— 撞不通就**中止**，
# 别像上一版那样带着坏构建把整批扫完（那是白烧一次场次）。
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; ALIVE=1; break
  fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
if [ "${ALIVE:-0}" != "1" ]; then echo ">>> 6 次探活全失败 ⇒ 中止（先看 [FAIL] 行是什么）"; exit 1; fi
nssh "REPS=${REPS:-5} bash ~/sfa_real/${BODY:-p107_calib_body.sh}" 2>&1 | grep -av "^Warning"
