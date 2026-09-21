#!/bin/bash
# P11 二分诊断 · 第 2 批（修正上一批 nostore 守卫恒假、根本没关掉 store 的错误）
#   nostore2        : StorePartial 完全不调用 + MergeAll 关掉，保留 SyncAll
#                     → 阶段一不碰 workspace；还挂 = 故障在切分后的搬运/计算或 barrier
#   nostore2_nosync : 同上，再把 SyncAll 拿掉 → 只剩"单元翻倍 + 分片读"
#   wsasval         : workspace 指针换成 value 张量（一定合法且 33 MB，够大）
#                     → 若故障消失，就是 framework 下发的那个 workspace 指针/尺寸不可信
#   storeO / storeML: 只保留 O 那一条 / 只保留 m、l 那两条 DataCopy，定位到具体指令
# ⚠️ 只改远端副本，本地提交源不动；trap restore 一律还原干净构建。
# 用法: run_p11_bisect2.sh <mode...>
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

resync() {
  bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "cd ~/sfa_real/code && \
    sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
    sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
}
restore() {
  echo "=== 还原干净构建 ==="
  resync
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error" | head -3
}
trap restore EXIT

[ $# -eq 0 ] && set -- nostore2 wsasval
for m in "$@"; do
  echo "########## bisect2 = $m ##########"
  resync
  MODE=$m timeout 300 ssh $C -o ConnectTimeout=25 "$H" "MODE=$m python3 - <<'PY'
import io, os
base = os.path.expanduser('~/sfa_real')
p = os.path.join(base, 'code/op_kernel/sparse_flash_attention.cpp')
s = io.open(p, encoding='utf-8').read()
mode = os.environ['MODE']
A_CALL  = '            StorePartial(o, ml, slot, nbCur);'
A_MERGE = '            MergeAll();'
A_SYNC  = '            SyncAll();'
A_WS    = '            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));\n            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));'
A_STO   = '        DataCopy(wsAccGm_[accBase], o,'
A_ML1   = '        DataCopy(wsMlGm_[mlBase], ml, DataCopyParams{1, mlBlk, 0, 0});'
NEVER   = 'slot == 0xFFFFFFFFu'
NEVER2  = 'unitBegin_ == 0xFFFFFFFFu'
if mode == 'nostore2':
    assert s.count(A_CALL) == 1 and s.count(A_MERGE) == 1
    s = s.replace(A_CALL, '            if (%s) { StorePartial(o, ml, slot, nbCur); }' % NEVER, 1)
    s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
elif mode == 'nostore2_nosync':
    assert s.count(A_CALL) == 1 and s.count(A_MERGE) == 1 and s.count(A_SYNC) == 1
    s = s.replace(A_CALL, '            if (%s) { StorePartial(o, ml, slot, nbCur); }' % NEVER, 1)
    s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
    s = s.replace(A_SYNC, '            if (%s) { SyncAll(); }' % NEVER2, 1)
elif mode == 'wsasval':
    assert s.count(A_WS) == 1
    s = s.replace(A_WS, '            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value) + 4096 * 1024);\n'
                        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value) + 4096 * 1024);', 1)
elif mode == 'storeO':
    assert s.count(A_CALL) == 1 and s.count(A_MERGE) == 1 and s.count(A_ML1) == 1
    s = s.replace(A_CALL, '            if (%s) { StorePartial(o, ml, slot, nbCur); }' % NEVER, 1)
    s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
    s = s.replace(A_STO, '        DataCopy(%s' % 'wsMlGm_[0], ml,', 1)   # O 那条改指到合法小地址
elif mode == 'storeML':
    assert s.count(A_STO) == 1
    s = s.replace(A_STO, '        if (%s) {\n            DataCopy(wsAccGm_[accBase], o,' % NEVER, 1)
else:
    raise SystemExit('unknown mode ' + mode)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', mode)
PY" || { echo "PATCH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh > /tmp/b2.log 2>&1; echo build_rc=\$?; grep -aE '构建 OK|error' /tmp/b2.log | head -3" 2>&1 | tail -4
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 2 none > /tmp/b2_p1.txt 2>&1; echo run_rc=\$?; grep -aE 'FAIL|PASS|超差|批量|平均' /tmp/b2_p1.txt | head -6" 2>&1 | tail -8
done
