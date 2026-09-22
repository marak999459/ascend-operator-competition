#!/bin/bash
# P36 探针：§15.54(4) 假设② —— **V 复用 `kBuf_`（P32 内核）在大形状上是不是净负的？**
# 与 P34 唯一不同的地方：这一份跑的是 **P29 三件套**（独立 `vBuf_`、`SFA_STAGE_MAX=40`），
# 对照格取 **同 (nb,k) 的 P34 读数**（同一场次、同一构建流程、只换那三个文件）。
#   * `k=48` 档在 P29 上**物理装不下**（`vBuf_` 要单独 48 KB）⇒ 只扫 16/32/40；
#   * 判读口径：`nb=1/k=40` 与 `nb=2/k=40` 两行，P29 若稳定快于 P32 ⇒ V 复用有真实代价，
#     那就该只在"装得下 48"时复用、装得下独立 vBuf 时不复用（host+kernel 各一处条件分配）。
# 全程只动远端副本 ~/sfa_real，本地 `code 3/code/` 一个字节不变；跑完 sync 回 P32 并重建。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p36_p29big.txt
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
B="$REPO/code 3/probes/backup"
: > $OUT
echo "=== A) 把 P29 三件套铺到远端 code/（本地提交源不动）===" | tee -a $OUT
for pair in "host_sparse_flash_attention.cpp.bak_pre_p32:op_host/sparse_flash_attention.cpp" \
            "kernel_sparse_flash_attention.cpp.bak_pre_p32:op_kernel/sparse_flash_attention.cpp" \
            "kernel_sparse_flash_attention_tiling.h.bak_pre_p32:op_kernel/sparse_flash_attention_tiling.h"; do
  src=${pair%%:*}; dst=${pair##*:}
  timeout 120 ssh -F $S $H "mkdir -p ~/sfa_real/code/$(dirname $dst) && cat > ~/sfa_real/code/$dst && md5sum ~/sfa_real/code/$dst" \
    < "$B/$src" 2>&1 | grep -av Warning | tee -a $OUT
done
echo "=== B) SoC 双注册 + FORCE 旋钮 + 重建 ===" | tee -a $OUT
timeout 300 ssh -F $S $H "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  python3 ~/code\ 3/probes/p23_sweep_patch.py op_host/sparse_flash_attention.cpp && \
  grep -c SFA_FORCE_NB op_host/sparse_flash_attention.cpp" 2>&1 | grep -av Warning | tee -a $OUT
timeout 900 ssh -F $S $H "cd ~/sfa_real && bash build.sh" 2>&1 | grep -aE "构建 OK|FAIL|error:" | head -5 | tee -a $OUT
echo "=== C) P29 在大形状上的 (nb,k) 网格（与 P34 同格对照）===" | tee -a $OUT
timeout 2400 ssh -F $S $H "$ENVR; \
for cs in w1 w2 w3 w4 w5; do \
  a=\$(timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms' | head -3 | tr '\n' ' '); \
  echo \"AUTO29 \$cs \$a\"; \
  for nb in 1 2; do for kb in 16 32 40; do \
    line=\$(env SFA_FORCE_NB=\$nb SFA_FORCE_NBLK=\$kb SFA_FORCE_KS=2 timeout 300 ./test_sfa_dev cases/\$cs.bin 3 none 2>&1 | grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|ERROR|err=' | head -3 | tr '\n' ' '); \
    echo \"GRID29 \$cs nb=\$nb kb=\$kb :: \$line\"; \
  done; done; \
done" 2>&1 | grep -av Warning | tee -a $OUT
echo "P36_DONE"
