#!/bin/bash
# P11 二分诊断 · 第 5 批
# 前情：bisect4 里 slot0（写到槽 0）和 scalar（只发一条标量 SetValue）都挂 —— 但这俩用的
# 都是框架下发的那个 ws 指针；而 bisect3 的 "valid 指针" 实验把基址放到 value+16 MB，
# p1 的 V 张量只有 8 MB（KV 头数=1），那个地址本身就是非映射的，等于没测。这一批重测。
#   flagsonly   : StorePartial 只留 SetFlag/WaitFlag(V_MTE3/MTE3_V, id 3)，一条 GM 访问都不发
#                 → 挂 = 事件旗标就是元凶（跟地址无关）
#   validmid    : ws 基址换成 value + 1,000,000 个 float（一定在 8 MB 张量内），三条 DataCopy 全留
#                 → 过 = 框架下发的那个指针不可信
#   scalarvalid : 同 validmid 的基址，但只留那条标量 SetValue
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

[ $# -eq 0 ] && set -- flagsonly validmid
for m in "$@"; do
  echo "########## bisect5 = $m ##########"
  resync
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "MODE=$m python3 - <<'PY'
import io, os
base = os.path.expanduser('~/sfa_real')
p = os.path.join(base, 'code/op_kernel/sparse_flash_attention.cpp')
s = io.open(p, encoding='utf-8').read()
mode = os.environ['MODE']
NEVER  = 'slot == 0xFFFFFFFFu'
NEVER2 = 'unitBegin_ == 0xFFFFFFFFu'
A_WS = ('            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));\n'
        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));')
A_MERGE = '            MergeAll();'
A_O = ('        DataCopy(wsAccGm_[accBase], o,\n'
       '                 DataCopyParams{1, static_cast<uint16_t>(nbCur * rowC * sizeof(float) / sfa::UB_BLK), 0, 0});')
A_ML1 = '        DataCopy(wsMlGm_[mlBase], ml, DataCopyParams{1, mlBlk, 0, 0});'
A_ML2 = '        DataCopy(wsMlGm_[mlBase + halfOff_], ml[halfOff_], DataCopyParams{1, mlBlk, 0, 0});'
for a in (A_WS, A_MERGE, A_O, A_ML1, A_ML2):
    assert s.count(a) == 1, ('anchor miss', a[:40])
s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
off = ' + 1000000u'
if mode == 'flagsonly':
    s = s.replace(A_O, '        (void)accBase; (void)mlBase; (void)mlBlk;', 1)
    s = s.replace(A_ML1, '', 1)
    s = s.replace(A_ML2, '', 1)
elif mode == 'validmid':
    s = s.replace(A_WS, '            (void)workspace;\n'
                        '            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value)%s);\n'
                        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value)%s);'
                        % (off, off), 1)
elif mode == 'scalarvalid':
    s = s.replace(A_WS, '            (void)workspace;\n'
                        '            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value)%s);\n'
                        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value)%s);'
                        % (off, off), 1)
    s = s.replace(A_O, '        wsMlGm_.SetValue(slot, 1.0f);', 1)
    s = s.replace(A_ML1, '        if (%s) { %s }' % (NEVER, A_ML1.strip()), 1)
    s = s.replace(A_ML2, '        if (%s) { %s }' % (NEVER, A_ML2.strip()), 1)
else:
    raise SystemExit('unknown mode ' + mode)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', mode)
PY" || { echo "PATCH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh > /tmp/b5.log 2>&1; echo build_rc=\$?; grep -aE '构建 OK|error' /tmp/b5.log | head -3" 2>&1 | tail -4
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 2 none > /tmp/b5_p1.txt 2>&1; echo run_rc=\$?; grep -aE 'FAIL|PASS|超差|批量' /tmp/b5_p1.txt | head -6" 2>&1 | tail -8
done
