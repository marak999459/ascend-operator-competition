#!/bin/bash
# P126 本地两臂对表（一发构建拿齐**归因**与**本地 Δ**）：
#   为什么要跑：闸门显示 pin 之后 `p1/p2/p4` 三例掉出"逐位一致"（超差仍 0），而 `p6/big1/w3/r64c65…`
#   逐位不变 ⇒ 必须把它归因到"模型重算出来的档变了"（n_blk 或 ks），不能停在"大概是 nb"。
#   构造：本地拉回 pristine → sync → 远端只打 P109 那枚 `SFA_FORCE_NB` + `SFA_PICK` 补丁（探针，不发货）
#   → 一次构建里跑 `auto` 与 `force_nb=1` 两臂。今天已实证 `force_nb=1` 的 tiling 与"把表钉成 {1}"
#   送进 CalcBlocking 是同一个参数（w3 5.9725 vs P109 的 5.9739、big1 0.9579 vs 0.9576，差 0.02~0.03 %）。
# ⚠️ 退出时 trap 还原远端干净构建，并把**本地提交源重新钉回 {1}**（这一发要发货的字节）。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
CASES="${CASES:-p1 p2 p4 p6 big1 w3 r64c65}"
NBS="${NBS:-1,2,4,8}"

repin() {
  echo "=== 收尾：远端还原干净构建 + 本地重新钉档 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  nssh "cd ~/sfa_real/code && sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp" >/dev/null 2>&1
  nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -3" 2>&1 | grep -av Warning
  python3 "$REPO/code 3/probes/p126_pin.py" apply 2>&1 | tail -2
}
trap repin EXIT

echo "=== ① 本地拉回 pristine 并 sync（探针要建在 pristine 上，auto 臂才是真的 auto） ==="
python3 "$REPO/code 3/probes/p126_pin.py" revert 2>&1 | tail -2
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | grep -aE "md5|=== " | tail -4
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  echo HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp) OLDTAB=\$(grep -c '{32, 16, 8, 4, 2, 1}' op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning
( cd "$REPO/code 3/probes" && tar cf - nb_sweep_patch.py p126_ab_body.sh ) | nssh "cd ~/sfa_real && tar xf -"
nssh "cd ~/sfa_real && python3 nb_sweep_patch.py code/op_host/sparse_flash_attention.cpp && \
  echo HITS_FORCE_NB=\$(grep -c 'SFA_FORCE_NB' code/op_host/sparse_flash_attention.cpp) \
       HITS_PICK=\$(grep -c 'SFA_PICK' code/op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning | tail -2
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5" 2>&1 | grep -av Warning
echo "=== ② 探活 ==="
ALIVE=0
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then echo ">>> 第 $try 次 OK"; ALIVE=1; break; fi
  echo ">>> 第 $try 次失败，等 60 s"; sleep 60
done
[ "$ALIVE" = "1" ] || { echo ">>> 探活全失败 ⇒ 中止"; exit 1; }
echo "=== ③ 两臂 ==="
nssh "CASES='$CASES' NBS='$NBS' REPS=${REPS:-5} bash ~/sfa_real/p126_ab_body.sh" 2>&1 | grep -av "^Warning"
