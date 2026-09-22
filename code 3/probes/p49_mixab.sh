#!/bin/bash
# P49 同场次 A/B：MIX 提交态 vs P38（纯 AIV）在同一屏里的设备时间。
# 动机：闸门已证 MIX 与 P38 逐位同（44 行日志字节相同），但 big1 读 0.6863 vs 13:0x 的 0.6576
#       —— 那是 5 小时前的另一场次，不能拿来定罪（§15.14(f) 的"禁止跨场次比时间"）。
# 只动远端副本 ~/sfa_real（本地 `code 3/code/` 全程保持 MIX 提交态不动），跑完把远端还原成 MIX。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
HOST_ALIAS="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
SSH_CFG="$HOME/.atomgitdevenv/.ssh/config"
BK="$REPO/code 3/probes/backup/p49_p38_clean"
SOC="ascend910_93"
rssh() { timeout 1500 ssh -F "$SSH_CFG" -o ConnectTimeout=25 "$HOST_ALIAS" "$@" 2>&1 | grep -av "^Warning: Permanently"; }
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

T3() {   # $1 = 档名
  for c in big1 d2048 w4; do
    rssh "$ENVR; ./test_sfa_dev cases/$c.bin 3 none 2>&1 | grep -a '时间' | sed 's|^|[$1] $c |'"
  done
}

putp38() {   # 远端 kernel/host 换回 P38 字节 + 重打 SoC 双注册 + 重建
  cat "$BK/sparse_flash_attention.cpp.kernel" | rssh "cat > ~/sfa_real/code/op_kernel/sparse_flash_attention.cpp"
  cat "$BK/sparse_flash_attention.cpp.host"  | rssh "cat > ~/sfa_real/code/op_host/sparse_flash_attention.cpp"
  rssh "cd ~/sfa_real/code && sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"$SOC\")|' op_host/sparse_flash_attention.cpp && grep -o 'AddConfig(\"[a-z0-9_]*\")' op_host/sparse_flash_attention.cpp | tr '\n' ' '"
  rssh "cd ~/sfa_real && bash build.sh" | grep -aE "构建 OK|FAIL|error" | head -3
}

putmix() {   # 远端换回本地 MIX 提交态（kernel/host 都从 code 3/code 推）
  cat "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" | rssh "cat > ~/sfa_real/code/op_kernel/sparse_flash_attention.cpp"
  cat "$REPO/code 3/code/op_host/sparse_flash_attention.cpp"  | rssh "cat > ~/sfa_real/code/op_host/sparse_flash_attention.cpp"
  rssh "cd ~/sfa_real/code && sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"$SOC\")|' op_host/sparse_flash_attention.cpp && grep -o 'AddConfig(\"[a-z0-9_]*\")' op_host/sparse_flash_attention.cpp | tr '\n' ' '"
  rssh "cd ~/sfa_real && bash build.sh" | grep -aE "构建 OK|FAIL|error" | head -3
}

echo "########## A/B 第 1 轮：MIX（当前远端就是它） ##########"
T3 "MIX-1"
echo "########## 换 P38 ##########"
putp38
T3 "P38-1"
echo "########## 换回 MIX ##########"
putmix
T3 "MIX-2"
echo "########## 第 2 轮交替：P38 / MIX ##########"
putp38
T3 "P38-2"
putmix
T3 "MIX-3"
echo "########## 收尾：远端应为 MIX 提交态（带 SoC sed） ##########"
rssh "md5sum ~/sfa_real/code/op_kernel/sparse_flash_attention.cpp ~/sfa_real/code/op_host/sparse_flash_attention.cpp; grep -c 'MIX_AIC_1_2' ~/sfa_real/code/op_kernel/sparse_flash_attention.cpp; grep -c 'blockDim + 1u' ~/sfa_real/code/op_host/sparse_flash_attention.cpp"
echo "P49_MIXAB_DONE"
