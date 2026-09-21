#!/bin/bash
# P11 诊断 · 第 6 批：把 kernel 实际收到的 workspace 指针"写回"到 softmax_max 张量里，
# 由 harness 原样打印，判断 aicore 的 workspace 形参到底是不是调用方给的那块。
# 做法：Init 里 ks>1 且 blockId==0 时，把 workspace 与 attention_out 的低 32 位各拆成
#       两个 16 位段写成 float（<65536，float 里精确），放在 maxGm_[12..15]；
#       同时把 StorePartial/MergeAll 关掉（nostore2），保证没人覆盖这四个槽。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
rssh() { timeout "${T:-600}" ssh $C -o ConnectTimeout=25 "$H" "$1" 2>&1 | grep -av "^Warning:"; }

bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"

echo "=== 1) 打 kernel + harness 补丁 ==="
T=300 rssh "python3 - <<'PY'
import io, os
base = os.path.expanduser('~/sfa_real')
p = os.path.join(base, 'code/op_kernel/sparse_flash_attention.cpp')
s = io.open(p, encoding='utf-8').read()
NEVER  = 'slot == 0xFFFFFFFFu'
NEVER2 = 'unitBegin_ == 0xFFFFFFFFu'
A_WS = ('            wsAccGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));\n'
        '            wsMlGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));')
A_CALL  = '            StorePartial(o, ml, slot, nbCur);'
A_MERGE = '            MergeAll();'
for a in (A_WS, A_CALL, A_MERGE):
    assert s.count(a) == 1, ('anchor miss', a[:40])
PROBE = A_WS + '''
            if (GetBlockIdx() == 0u && softmax_max_out != nullptr) {
                const uint32_t wl = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(workspace));
                const uint32_t ol = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(attention_out));
                maxGm_.SetValue(15u, static_cast<float>(wl & 0xFFFFu));
                maxGm_.SetValue(14u, static_cast<float>(wl >> 16));
                maxGm_.SetValue(13u, static_cast<float>(ol & 0xFFFFu));
                maxGm_.SetValue(12u, static_cast<float>(ol >> 16));
                maxGm_.SetValue(11u, static_cast<float>(ks_));
                maxGm_.SetValue(10u, static_cast<float>(slotStride_));
                maxGm_.SetValue(9u,  static_cast<float>(total));
            }'''
s = s.replace(A_WS, PROBE, 1)
# total 是在这段之后才算出来的局部量 —— 探针里就地重算一份
s = s.replace('maxGm_.SetValue(9u,  static_cast<float>(total));',
              'maxGm_.SetValue(9u,  static_cast<float>(B_ * S1_ * nHeadBlk_ * ks_));', 1)
s = s.replace(A_CALL, '            if (%s) { StorePartial(o, ml, slot, nbCur); }' % NEVER, 1)
s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
io.open(p, 'w', encoding='utf-8').write(s)

h = os.path.join(base, 'test_sfa_dev.cpp')
t = io.open(h, encoding='utf-8').read()
A_H = '    std::printf(\"case=%s B=%u'
assert t.count(A_H) == 1, 'harness anchor miss'
io.open(h, 'w', encoding='utf-8').write(t)
print('kernel patched')
PY"

echo "=== 2) harness：跑完后原样打印 max[9..15] ==="
T=300 rssh 'cd ~/sfa_real && python3 - <<PY
import io
h = "test_sfa_dev.cpp"
t = io.open(h, encoding="utf-8").read()
anchor = "    Stat so;"
assert t.count(anchor) == 1, "anchor"
inj = """    if (nlse >= 16) {
        std::printf("[P11PROBE] max[9..15] ks=%.0f slotStride=%.0f total=%.0f outHi=%.0f outLo=%.0f wsHi=%.0f wsLo=%.0f\\n",
                    gmax[9], gmax[10], gmax[11], gmax[12], gmax[13], gmax[14], gmax[15]);
    }
"""
t = t.replace(anchor, inj + anchor, 1)
io.open(h, "w", encoding="utf-8").write(t)
print("harness patched")
PY'

echo "=== 3) build kernel + harness ==="
T=1500 rssh "$ENVR; bash build.sh > /tmp/b6.log 2>&1; echo build_rc=\$?; grep -aE '构建 OK|error' /tmp/b6.log | head -5"
T=300 rssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh >/dev/null 2>&1; \
  g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev \
    -I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include \
    -I\$HOME/sfa_real/vendor/custom/op_api/include \
    -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 \
    -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi 2>&1 | head -8; ls -la test_sfa_dev | head -2"

echo "=== 4) 跑 p1，读回指针 ==="
T=600 rssh "$ENVR; timeout 200 ./test_sfa_dev cases/p1.bin 2 none 2>&1 | grep -aE 'P11PROBE|HARNESS|FAIL|PASS|超差' | head -8"

echo "=== 5) 还原 ==="
bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1
T=300 rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"
T=1500 rssh "$ENVR; bash build.sh > /tmp/b6r.log 2>&1; echo restore_rc=\$?; grep -aE '构建 OK|error' /tmp/b6r.log | head -3"
