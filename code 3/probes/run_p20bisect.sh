#!/bin/bash
# P20 退步的机制二分（只改远端副本，本地提交源不动）：
#   base = pre_p20（标量 GM 读，p1 0.1652 / q1h 0.0895，见 p20ab_222206.log）
#   P20  = 窗口 + UB 标量读                → p1 0.2865（+121 µs ≈ 118 ns/token，比 GM 读还贵 3 倍）
#   V2   = 代码结构全在、idxUbOn_=false ⇒ 每 token 仍走标量 GM 读，只多一个分支 + 一次函数调用
#          ⇒ 若 V2 ≈ base：分支/调用是免费的，**贵的是 UB 标量读本身**
#   V3   = 补窗 DataCopy 照旧，但**读回仍读 GM**（窗口内容不用于取值）
#          ⇒ 若 V3 ≈ base：连"每 512 项一次 DataCopy"也是免费的 ⇒ 118 ns/token 全在 UB 标量读上
#          若 V3 ≈ P20：真正的元凶是 DataCopy 卡在循环里（管道排空），不是取值
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
C="-F $HOME/.atomgitdevenv/.ssh/config"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
rssh() { timeout "${T:-1500}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }

# 先确保远端是"P20 当前提交源"，再在其上做 sed 变体
cat "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" | rssh "cat > ~/sfa_real/$KER"
rssh "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"

for V in V2 V3; do
  echo "########## $V ##########"
  cat "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" | rssh "cat > ~/sfa_real/$KER"
  if [ "$V" = "V2" ]; then
    rssh "cd ~/sfa_real && sed -i 's|idxUbOn_ = ((reinterpret_cast<uint64_t>(sparse_indices) \& (sfa::UB_BLK - 1ULL)) == 0ULL);|idxUbOn_ = false;|' $KER"
  else
    # IdxVal 里两处窗口取值改成读 GM（保留 RefillIdxWin 的调用与 DataCopy）
    rssh "cd ~/sfa_real && sed -i 's|return idxWin_.GetValue(static_cast<uint32_t>(absPos - idxWinBeg_));|return idxGm_.GetValue(absPos);|g' $KER"
  fi
  printf '  sed 命中: '; rssh "grep -c 'idxUbOn_ = false;\|return idxGm_.GetValue(absPos);' ~/sfa_real/$KER"
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error:|Error" | head -5
  rssh "$ENVR; for n in p1 q1h big1; do printf '  %s %-6s ' \"$V\" \"\$n\"; \
      timeout 300 ./test_sfa_dev cases/\$n.bin 8 none >/tmp/v.txt 2>&1; \
      grep -aoE '平均 [0-9.]+ ms' /tmp/v.txt | head -2 | tr '\n' '|'; echo; done"
done

echo "=== 还原：当前提交源 + 干净构建 + 正确性 ==="
cat "$REPO/code 3/code/op_kernel/sparse_flash_attention.cpp" | rssh "cat > ~/sfa_real/$KER"
rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|FAIL" | head -3
rssh "$ENVR; ./test_sfa_dev cases/p1.bin 3 diff" 2>&1 | grep -aE "超差|逐位" | tail -3
