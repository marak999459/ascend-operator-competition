#!/bin/bash
# P11 二分诊断 · 第 4 批：故障已锁在 StorePartial 里（阶段一 + SyncAll 不碰 ws 跑通；
# 只要 StorePartial 一开就挂，O 大写/ml 小写都挂）。这一批分清两种可能：
#   A. kernel 看到的 workspace 实际可比 67584 小（框架没照 host 的申请给）
#      → 把槽位下标压到低位（slot0 / slotmask7）就该过
#   B. 那个指针本身不可信，或者 UB->GM 的 DataCopy 在这条路上有别的问题
#      → scalar（只用标量 SetValue 写 ws，不发 MTE）能过而 slot0 不过，说明是指针；
#        全都不过，说明是写指令本身。
# 一律同时关掉 MergeAll，保证只看阶段一。
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

[ $# -eq 0 ] && set -- slot0 scalar
for m in "$@"; do
  echo "########## bisect4 = $m ##########"
  resync
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "MODE=$m python3 - <<'PY'
import io, os, re
base = os.path.expanduser('~/sfa_real')
p = os.path.join(base, 'code/op_kernel/sparse_flash_attention.cpp')
s = io.open(p, encoding='utf-8').read()
mode = os.environ['MODE']
NEVER2 = 'unitBegin_ == 0xFFFFFFFFu'
A_BASE = '        const uint32_t accBase = slot * slotStride_;'
A_MERGE = '            MergeAll();'
A_O = ('        DataCopy(wsAccGm_[accBase], o,\n'
       '                 DataCopyParams{1, static_cast<uint16_t>(nbCur * rowC * sizeof(float) / sfa::UB_BLK), 0, 0});')
A_ML1 = '        DataCopy(wsMlGm_[mlBase], ml, DataCopyParams{1, mlBlk, 0, 0});'
A_ML2 = '        DataCopy(wsMlGm_[mlBase + halfOff_], ml[halfOff_], DataCopyParams{1, mlBlk, 0, 0});'
assert s.count(A_BASE) == 2, ('anchor miss BASE', s.count(A_BASE))   # StorePartial 与 MergeUnit 各一处
for a in (A_MERGE, A_O, A_ML1, A_ML2):
    assert s.count(a) == 1, ('anchor miss', a[:40])
# 一律只留阶段一
s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
if mode == 'slot0':
    s = s.replace(A_BASE, '        const uint32_t accBase = 0u;')   # 两处一起改，槽位口径保持一致
elif mode == 'slotmask7':
    s = s.replace(A_BASE, '        const uint32_t accBase = (slot & 7u) * slotStride_;')
elif mode == 'scalar':
    s = s.replace(A_O, '        wsMlGm_.SetValue(slot, 1.0f);', 1)
    s = s.replace(A_ML1, '        if (%s) { %s }' % (NEVER2, A_ML1.strip()), 1)
    s = s.replace(A_ML2, '        if (%s) { %s }' % (NEVER2, A_ML2.strip()), 1)
else:
    raise SystemExit('unknown mode ' + mode)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', mode)
PY" || { echo "PATCH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh > /tmp/b4.log 2>&1; echo build_rc=\$?; grep -aE '构建 OK|error' /tmp/b4.log | head -3" 2>&1 | tail -4
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 2 none > /tmp/b4_p1.txt 2>&1; echo run_rc=\$?; grep -aE 'FAIL|PASS|超差|批量' /tmp/b4_p1.txt | head -6" 2>&1 | tail -8
done
