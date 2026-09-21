#!/bin/bash
# P11 真机二分诊断：定位 "DDR address of the MTE instruction is out of range" 落在哪一段。
#   nostore : StorePartial 空转（阶段一不碰 workspace）+ 关掉 MergeAll
#   nomerge : StorePartial 保留，MergeUnit 立刻返回（只看阶段一）
#   kfix    : kernel 里强行 ks_=1（host 仍下发 ks=2）→ 退化成 P14 路径（对照组，必须过）
#   wsmul   : 只改 host，workspace 尺寸 ×8 → 若故障消失，就是槽位尺寸/布局对不上
# ⚠️ 只改远端副本，本地提交源不动；结束一律还原干净构建。
# 用法: run_p11_bisect.sh [mode...]      （默认全部三种）
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

restore() {
  echo "=== 还原干净构建 ==="
  resync
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
}
# 推本地提交源 + 重打远端专属 SoC 补丁
resync() {
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
}
trap restore EXIT

[ $# -eq 0 ] && set -- wsmul nostore kfix
for m in "$@"; do
  echo "########## bisect = $m ##########"
  resync
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "MODE=$m python3 - <<'PY'
import io, os
base = os.path.expanduser('~/sfa_real')
p = os.path.join(base, 'code/op_kernel/sparse_flash_attention.cpp')
ph = os.path.join(base, 'code/op_host/sparse_flash_attention.cpp')
s = io.open(p, encoding='utf-8').read()
mode = os.environ['MODE']
A_STORE = '        const uint32_t rowC = static_cast<uint32_t>(D_);\n        const uint32_t accBase = slot * slotStride_;'
A_MERGE = '            MergeAll();'
A_KS    = '        ks_ = tiling_data.ksplit;'
A_WS    = '    SparseFlashAttentionTilingData *tiling = context->GetTilingData<SparseFlashAttentionTilingData>();'
guard = lambda n: '        if (%s == 0xFFFFFFFFu) { return; }' % n
if mode == 'nostore':
    # 阶段一完全不碰 workspace，阶段二也不读 -> 剩下的只有分片后的 gather/计算
    assert s.count(A_STORE) == 1 and s.count(A_MERGE) == 1
    s = s.replace(A_STORE, guard('slot') + '\n' + A_STORE, 1)
    s = s.replace(A_MERGE, '            if (unitBegin_ == 0xFFFFFFFFu) { MergeAll(); }', 1)
elif mode == 'nomerge':
    # 只关阶段二：StorePartial 照写，MergeUnit 不跑
    assert s.count(A_MERGE) == 1
    s = s.replace(A_MERGE, '            if (unitBegin_ == 0xFFFFFFFFu) { MergeAll(); }', 1)
elif mode == 'kfix':
    assert s.count(A_KS) == 1
    s = s.replace(A_KS, A_KS + '\n        ks_ = 1u;', 1)
elif mode == 'wsmul':
    # 只改 host：workspace 多给 8 倍 —— 若故障消失，就是尺寸/布局对不上
    sh = io.open(ph, encoding='utf-8').read()
    assert sh.count(A_WS) == 1
    sh = sh.replace(A_WS, '    wsBytes *= 8ULL;\n' + A_WS, 1)
    io.open(ph, 'w', encoding='utf-8').write(sh)
elif mode == 'allmine':
    # 关掉分片门（每个分片都处理全部块）：单元数翻倍但搬运地址不变
    assert s.count('                mine = (shardCnt == 0u);') == 1
    s = s.replace('                mine = (shardCnt == 0u);',
                  '                mine = (shardCnt == 65535u);', 1)
elif mode == 'nosync':
    # 整条 barrier 拿掉（结果必然错，但地址都合法）→ 只测"故障是不是 SyncAll"
    assert s.count('            SyncAll();\n            MergeAll();') == 1
    s = s.replace('            SyncAll();\n            MergeAll();',
                  '            if (unitBegin_ == 0xFFFFFFFFu) { SyncAll(); }\n            MergeAll();', 1)
elif mode == 'bd40':
    # 只改 host：ks>1 时不收缩块数（保持 40）→ 测"收缩块数 × barrier"的相互作用
    sh = io.open(ph, encoding='utf-8').read()
    assert sh.count(A_WS) == 1
    sh = sh.replace(A_WS, '    if (ksplit > 1U) { blockDim = 40U; }\n' + A_WS, 1)
    io.open(ph, 'w', encoding='utf-8').write(sh)
else:
    raise SystemExit('unknown mode')
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', mode)
PY" || { echo "PATCH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -5
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 2 none > /tmp/bs_p1.txt 2>&1; echo \"rc=\$?\"; grep -aE 'FAIL|超差|时间|平均|batch|批量' /tmp/bs_p1.txt | head -6"
done
