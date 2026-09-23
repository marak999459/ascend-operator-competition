#!/bin/bash
# P68：cube 数据面的最后一层黑盒 = "AIV 到底从环里读到了什么"。
#
# 现状（p68 = 扇入修好 + ScoreFromRing 补 MTE2_V 之后）：
#   r6_multiB（1 片/单元）  超差 0/4096 + 0/8  ⇒ 单片链路**全对**
#   p1 (N1=4, 43 片)        每行 head0=全 0(l=0)、head1/head2 = 一行"≈0 的分数"
#                            (max≈0.35, sum≈1450 ⇒ 2048·exp(-0.35)=1443 对得上)、
#                            **head3 = 与 gold 逐位相同**
#   p2 (N1=2, 多片)         head0=全 0、head1=同一族"≈0"
# ⇒ 环里只有**一部分**行是真的，其余行读回来像 fp16 位模式被当 fp32 看的碎屑。
#   光靠输出反推不出行映射，所以这一档把**末片的 sc 原值**直接印到 attention_out 上：
#   WriteOut 里把 `Muls(oi, 1/l)` 换成 `Adds(oi, sc[i*nBlk_], 0, 64) + Duplicate(尾)` ⇒
#   dumped out 的前 64 个 fp16 = 本头末片读到的原始 score（LSE 那一栏照旧，仍给 m/l）。
#   本地再用 p1.bin 的 Q/K/idx 逐格对：真 score[h][t] vs 环里读回的那一行 ⇒ 一次定死
#   "行距 / 行号 / 片号奇偶（槽位）"三选一。
# ⚠️ 只打远端 ~/sfa_real；跑完 RESTORE + 重建成净版。判据是**读回来的数**，不是 PASS/FAIL。
set -u
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
K=code/op_kernel/sparse_flash_attention.cpp
TMO="${TMO:-75}"
ns() { timeout "${T:-$((TMO + 300))}" ssh -F $S $H "$@" 2>&1 | grep -av Warning; }

echo "--- BACKUP ---"
ns "$ENVR; cp -f $K /tmp/kernel_p68_orig.cpp; md5sum /tmp/kernel_p68_orig.cpp"

echo "--- PATCH ---"
ns "$ENVR; python3 - <<'PYX'
import io
p = 'code/op_kernel/sparse_flash_attention.cpp'
src = io.open(p, encoding='utf-8').read()
OLD = '''                const float l = ml.GetValue(halfOff_ + i);
                LocalTensor<float> oi = o[i * rowC];
                if (l > 0.0f) {
                    Muls(oi, oi, 1.0f / l, rowC);
                } else {
                    Duplicate(oi, 0.0f, rowC);      // 整行被 mask：输出 0
                }'''
NEW = '''                const float l = ml.GetValue(halfOff_ + i);   // [p68] LSE 照旧
                LocalTensor<float> oi = o[i * rowC];
                Adds(oi, sBuf_.Get<float>()[i * nBlk_], 0.0f, 64u);   // [p68] 末片原值
                Duplicate(oi[64], 0.0f, rowC - 64u);                  // [p68]'''
assert src.count(OLD) == 1, 'anchor x%d' % src.count(OLD)
io.open(p, 'w', encoding='utf-8').write(src.replace(OLD, NEW, 1))
print('PATCHED_RINGDUMP')
PYX"

echo "--- BUILD ---"
BOUT=$(ns "$ENVR; bash build.sh > /tmp/p68_build.log 2>&1; echo build_rc=\$?; grep -m6 -E 'error|Error' /tmp/p68_build.log | head -6")
echo "$BOUT"
case "$BOUT" in
  *build_rc=0*) ;;
  *) echo "!! 编译没过 ⇒ 恢复退出"
     ns "$ENVR; cp -f /tmp/kernel_p68_orig.cpp $K; bash build.sh > /tmp/p68_rebuild.log 2>&1; echo rebuild_rc=\$?"
     echo "P68_ABORT"; exit 1;;
esac

echo "--- RUN (write -> /tmp/p68_act2) ---"
ns "$ENVR; mkdir -p /tmp/p68_act2; for cs in p1 p2 r6_multiB; do timeout $TMO ./test_sfa_dev cases/\$cs.bin 1 write 2e-3 1e-2 /tmp/p68_act2 > /tmp/p68r_\$cs.log 2>&1; echo \"\$cs rc=\$?\"; done; ls /tmp/p68_act2"

echo "--- PULL ---"
timeout 300 ssh -F $S $H "cd /tmp/p68_act2 && tar cf - ." 2>/dev/null | (mkdir -p /tmp/p68/dump && tar xf - -C /tmp/p68/dump) && ls -la /tmp/p68/dump

echo "--- RESTORE ---"
ns "$ENVR; cp -f /tmp/kernel_p68_orig.cpp $K; md5sum $K"
ns "$ENVR; bash build.sh > /tmp/p68_rebuild.log 2>&1; echo rebuild_rc=\$?"
echo "P68_DONE"
