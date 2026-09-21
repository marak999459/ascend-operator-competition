#!/bin/bash
# P11 上机诊断：把 host tiling 的切核决策（nb/nBlk/ksplit/blockDim/wsBytes）打到 stderr。
# ⚠️ 探针只改远端副本，本地提交源不动；结束一律 npu.sh sync + build 还原干净构建。
# 用法: run_p11_diag.sh [case...]     （默认 p1）
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
HOSTF='code/op_host/sparse_flash_attention.cpp'
ANCHOR='    SparseFlashAttentionTilingData *tiling = context->GetTilingData<SparseFlashAttentionTilingData>();'
PRINTF='    std::fprintf(stderr, "[P11DIAG] nb=%u nBlk=%u ks=%u bd=%u ws=%llu units=%llu chunks=%llu\\n", nb, nBlk, ksplit, blockDim, (unsigned long long)wsBytes, (unsigned long long)((uint64_t)B * Q_S * ((Q_N + nb - 1U) / nb)), (unsigned long long)((nBlk == 0U) ? 0ULL : ((uint64_t)sparse_count + nBlk - 1ULL) / nBlk));'

restore() {
  echo "=== 还原干净构建 ==="
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
}
trap restore EXIT

# 远端插入 printf（幂等：已插过就跳过）
timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real && python3 - <<'PY'
import io
p='$HOSTF'
s=io.open(p,encoding='utf-8').read()
if '[P11DIAG]' in s:
    print('already patched'); raise SystemExit(0)
a=r'''$ANCHOR'''
b=r'''$PRINTF'''
assert s.count(a)==1, s.count(a)
s=s.replace(a, b+'\n'+a, 1)
if '#include <cstdio>' not in s:
    s=s.replace('#include <cstdint>', '#include <cstdint>\n#include <cstdio>', 1)
io.open(p,'w',encoding='utf-8').write(s)
print('patched')
PY" || { echo "PATCH FAIL"; exit 1; }

timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -5
for n in "${@:-p1}"; do
  echo "########## $n ##########"
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; ./test_sfa_dev cases/$n.bin 1 diff 2>&1 | grep -aE 'P11DIAG|FAIL|超差|不逐位|时间' | head -12"
done
