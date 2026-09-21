import io, os
# P11 探针：kernel 收到的 workspace 形参到底是不是调用方给的那块？
# aicore 里禁止 float<->整数转换，所以不做数值编码，改用"一位答案"的问法：
#   harness 在 launch 前把 ws 全部填成 0xBE 字节 —— 若 kernel 读 ws[0] 得到 0xBEBEBEBE，
#   就说明它拿到的是同一个 buffer；否则指针被框架换掉了。
#   判定结果用两个 float 常量写回 maxGm_[15]（7.0=同一个 buffer，3.0=不是）。
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
    assert s.count(a) == 1, ('kernel anchor miss', a[:40])
PROBE = A_WS + '\n'.join([
    '            if (GetBlockIdx() == 0u && softmax_max_out != nullptr) {',
    '                const uintptr_t wp = reinterpret_cast<uintptr_t>(workspace);',
    '                const uintptr_t op = reinterpret_cast<uintptr_t>(attention_out);',
    '                const uintptr_t kp = reinterpret_cast<uintptr_t>(key);',
    '                maxGm_.SetValue(0u, ((wp - op) == 0x7000u) ? 7.0f : 3.0f);',
    '                maxGm_.SetValue(1u, ((wp > op) && ((wp - op) < 0x100000u)) ? 7.0f : 3.0f);',
    '                maxGm_.SetValue(2u, ((wp > kp) && ((wp - kp) < 0x100000u)) ? 7.0f : 3.0f);',
    '                maxGm_.SetValue(3u, (wp == 0u) ? 7.0f : 3.0f);',
    '            }',
])
s = s.replace(A_WS, PROBE, 1)
s = s.replace(A_CALL, '            if (%s) { StorePartial(o, ml, slot, nbCur); }' % NEVER, 1)
s = s.replace(A_MERGE, '            if (%s) { MergeAll(); }' % NEVER2, 1)
io.open(p, 'w', encoding='utf-8').write(s)

h = os.path.join(base, 'test_sfa_dev.cpp')
t = io.open(h, encoding='utf-8').read()
A1 = '    auto RunOnce = [&]() { return aclnnSparseFlashAttention(ws, wsSize, ex, stream) == ACL_SUCCESS; };'
A2 = '    Stat so;'
assert t.count(A1) == 1 and t.count(A2) == 1, 'harness anchor miss'
t = t.replace(A1, '    if (wsSize > 0) { aclrtMemset(ws, wsSize, 0xBE, wsSize); }\n' + A1, 1)
inj = ('    if (nlse >= 16) {\n'
       '        std::printf("[P11PROBE] wsMinusOut0x7000=%.1f wsNearOut=%.1f wsNearKey=%.1f wsIsNull=%.1f\\n", gmax[0], gmax[1], gmax[2], gmax[3]);\n'
       '    }\n')
t = t.replace(A2, inj + A2, 1)
io.open(h, 'w', encoding='utf-8').write(t)
print('patched kernel + harness')
