#!/bin/bash
# P45：§15.59(4) 的第一问 —— "不等回执 / 双缓冲"能把 §15.43(c) 那 1.35 µs/轮的锁步税压到多少。
# 同场次连跑 prod / lock / send / credit 四档（XR=32、BD=8、big1、只看批量口径）：
#   prod   = AIV 停工，纯 Cube 产数基线
#   lock   = 锁步（每轮 set(5) → 扇入 wait(6)）= §15.43 那档，逐字节同形 ⇒ 拿来对表
#   send   = 只广播不等回执（**不是**可实现形态，给"旗标广播 + 唤醒"定价 = 税的下界）
#   credit = 两槽乒乓 + AIC 每轮开头只等"两轮前"的回执 = **M1 可实现的形态**
# ⚠️ 探针档不算算子 ⇒ 对拍必 FAIL 是预期；判据只有时间，绝不锁 golden。
# ⚠️ 只改远端副本 ~/sfa_real/...（kernel 每档重推、host 只 sed 一次 blockDim），trap 还原干净构建。
# 用法: p45_xfer_tax.sh ["prod:32 lock:32 send:32 credit:32"]      默认 XR=32 四档
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
C="-F $HOME/.atomgitdevenv/.ssh/config"
KER="code/op_kernel/sparse_flash_attention.cpp"
HOST="op_host/sparse_flash_attention.cpp"
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}'
LINK="-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lcust_opapi"
PE='SFA_CBASE=32 SFA_CSTRIDE=16 SFA_QUIET_XC3=1'      # 探针只用 [0]/[64+bi] 两格 ⇒ 关掉 XC3/P6 两屏
BDN="${BDN:-8}"
rssh() { timeout "${T:-1600}" ssh $C -o ConnectTimeout=25 "$H" "$@"; }
SOC() { rssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp"; }
CLEAN() { bash "$REPO/code 3/npu_debug/npu.sh" sync >/dev/null 2>&1; SOC >/dev/null; }
BUILDH() { local o; o=$(rssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; \
    rm -f /tmp/tsd; \
    { g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd $LINK -lnnopcapbase 2>/dev/null || \
      g++ -std=c++17 -O2 test_sfa_dev.cpp -o /tmp/tsd $LINK -lnnopbase 2>&1 | head -8; } ; \
    if [ -x /tmp/tsd ]; then mv -f /tmp/tsd test_sfa_dev; echo 'HARNESS OK'; \
    else echo 'HARNESS BUILD FAIL'; fi" 2>&1 | tail -3)
  printf '%s\n' "$o"; case "$o" in *"HARNESS OK"*) return 0;; *) return 1;; esac; }
restore() { echo "=== 还原干净构建 ==="; CLEAN; \
  rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error:" | head -3; \
  rssh "$ENVR; ./test_sfa_dev cases/big1.bin 3 diff" 2>&1 | grep -aE "超差|批量口径|逐位" | head -3; \
  rssh "cd ~/sfa_real && grep -ac CUBEPROBE code/op_kernel/sparse_flash_attention.cpp; md5sum code/op_kernel/sparse_flash_attention.cpp"; }
trap restore EXIT

if [ "$#" -eq 0 ]; then set -- prod:32 lock:32 send:32 credit:32; fi

echo "=== 0) 干净态 + 探针 harness + host blockDim=$BDN ==="
CLEAN
rssh "$ENVR; bash build.sh" 2>&1 | grep -aE "构建 OK|error:" | head -3 || { echo "前置构建失败"; exit 1; }
rssh "cat > /tmp/cube_harness_patch.py" < "$REPO/code 3/probes/cube_harness_patch.py"
rssh "cd ~/sfa_real && python3 /tmp/cube_harness_patch.py" || { echo "harness 补丁失败"; exit 1; }
BUILDH || { echo "harness 构建失败 ⇒ 放弃（否则读到旧二进制的假数）"; exit 1; }
rssh "cd ~/sfa_real/code && sed -i 's|context->SetBlockDim(blockDim);|context->SetBlockDim($BDN);  // CUBEPROBE BD|' $HOST && grep -n 'SetBlockDim' $HOST | head -2"

for a in "$@"; do
  kind="${a%%:*}"; xr="${a##*:}"
  echo "########## cubexfer XFKIND=$kind XR=$xr BD=$BDN ##########"
  CLEAN   # 只复位 host/kernel 源；harness 在 ~/sfa_real 根目录，不受影响
  XFKIND="$kind" XR="$xr" python3 "$REPO/code 3/probes/mk_probe_cube.py" cubexfer \
      | rssh "cat > ~/sfa_real/$KER" || { echo "PUSH FAIL $a"; continue; }
  rssh "cd ~/sfa_real/code && sed -i 's|context->SetBlockDim(blockDim);|context->SetBlockDim($BDN);  // CUBEPROBE BD|' $HOST && grep -c 'CUBEPROBE BD' $HOST"
  BOUT=$(rssh "$ENVR; bash build.sh" 2>&1)
  printf '%s\n' "$BOUT" | grep -aE "构建 OK|error:|Error" | head -8
  printf '%s\n' "$BOUT" | grep -aq "构建 OK" || { echo ">>> $a 构建失败 ⇒ 跳过（防旧 .so 读假数）"; continue; }
  rssh "$ENVR; $PE timeout ${TO:-240} stdbuf -o0 -e0 ./test_sfa_dev cases/big1.bin 5 none >/tmp/xf_${kind}_${xr}.txt 2>&1; \
    echo \"rc=\$?\"; grep -a '批量口径' /tmp/xf_${kind}_${xr}.txt | grep -aoE '平均 [0-9.]+ ms' | head -1; \
    grep -aoE '时间: 平均 [0-9.]+ ms' /tmp/xf_${kind}_${xr}.txt | head -1; \
    grep -aoE '\[CUBE\] AIC alive = [0-9.]+|\[CUBE\] AIV bi=[0-9]+ = [0-9.]+' /tmp/xf_${kind}_${xr}.txt | head -4; \
    grep -aiE 'error|fail' /tmp/xf_${kind}_${xr}.txt | head -3"
done
