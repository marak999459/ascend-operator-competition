#!/bin/bash
# P11 二分诊断 · 第 3 批：已确认
#   nostore2（阶段一 + SyncAll，不碰 workspace）→ 跑通（只是结果错）
#   nomerge（StorePartial 开、MergeAll 关）      → 挂
#   ⇒ 故障在 StorePartial 这三条 GM 写里。这一批分清是"指针不可信"还是"指令/尺寸"。
#   svalid_nomerge : ws 指针换成 value 张量 + MergeAll 关 → 只留 StorePartial，且地址一定合法
#                    过 = framework 那个 workspace 指针不可信；挂 = 写指令本身/尺寸
#   smlonly        : 只留 m、l 两条小写（砍掉 O 那条 8 KB 大写）
#   soccopyonly    : 只留 O 那条大写（砍掉两条 ml 小写）
# ⚠️ 只改远端副本；trap restore 还原干净构建。
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

[ $# -eq 0 ] && set -- svalid_nomerge
for m in "$@"; do
  echo "########## bisect3 = $m ##########"
  resync
  timeout 300 ssh $C -o ConnectTimeout=25 "$H" "MODE=$m python3 - <<'PY'
import io, os
base = os.path.expanduser('~/sfa_real')
p = os.path.join(base, 'code/op_kernel/sparse_flash_attention.cpp')
s = io.open(p, encoding='utf-8').read()
mode = os.environ['MODE']
NEVER  = 'slot == 0xFFFFFFFFu'
NEVER2 = 'unitBegin_ == 0xFFFFFFFFu'
A_MERGE = '            MergeAll();'
A_WS = ('            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));\n'
        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));')
A_O = ('        DataCopy(wsAccGm_[accBase], o,\n'
       '                 DataCopyParams{1, static_cast<uint16_t>(nbCur * rowC * sizeof(float) / sfa::UB_BLK), 0, 0});')
A_O_OFF = ('        if (%s) {\n            DataCopy(wsAccGm_[accBase], o,\n'
           '                     DataCopyParams{1, static_cast<uint16_t>(nbCur * rowC * sizeof(float) / sfa::UB_BLK), 0, 0});\n'
           '        }') % NEVER
A_ML1 = '        DataCopy(wsMlGm_[mlBase], ml, DataCopyParams{1, mlBlk, 0, 0});'
A_ML2 = '        DataCopy(wsMlGm_[mlBase + halfOff_], ml[halfOff_], DataCopyParams{1, mlBlk, 0, 0});'
for a in (A_MERGE, A_WS, A_O, A_ML1, A_ML2):
    assert s.count(a) == 1, ('anchor miss', a[:40])
if mode == 'svalid_nomerge':
    s = s.replace(A_WS, '            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value) + 4096 * 1024);\n'
                        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(value) + 4096 * 1024);', 1)
    s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
elif mode == 'smlonly':
    s = s.replace(A_O, A_O_OFF, 1)
    s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
elif mode == 'soccopyonly':
    s = s.replace(A_ML1, '        if (%s) { %s }' % (NEVER, A_ML1.strip()), 1)
    s = s.replace(A_ML2, '        if (%s) { %s }' % (NEVER, A_ML2.strip()), 1)
    s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
else:
    raise SystemExit('unknown mode ' + mode)
io.open(p, 'w', encoding='utf-8').write(s)
print('patched', mode)
PY" || { echo "PATCH FAIL"; continue; }
  timeout 1200 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; bash build.sh > /tmp/b3.log 2>&1; echo build_rc=\$?; grep -aE '构建 OK|error' /tmp/b3.log | head -3" 2>&1 | tail -4
  timeout 600 ssh $C -o ConnectTimeout=25 "$H" "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 2 none > /tmp/b3_p1.txt 2>&1; echo run_rc=\$?; grep -aE 'FAIL|PASS|超差|批量|error code|out of range' /tmp/b3_p1.txt | head -8" 2>&1 | tail -10
done
