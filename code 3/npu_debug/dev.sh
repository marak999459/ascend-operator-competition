#!/bin/bash
# SFA 真机开发闭环：sync → 打 SoC 补丁 → 构建 → 编译 dev harness → 跑矩阵/计时
# 用法:
#   dev.sh build                 同步 + 构建 + 编 dev harness
#   dev.sh matrix <write|diff|none> [cases...]   跑正确性矩阵（默认 r1..r8，fp16 实例）
#   dev.sh f32    <write|diff|none> [cases...]   同一批用例再跑一遍 fp32 实例（§15.13）
#   dev.sh one <case> <reps> <act>              单用例
#   dev.sh big                   生成 big1 性能用例
set -u
HOST_ALIAS="devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0"
SSH_CFG="$HOME/.atomgitdevenv/.ssh/config"
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
SOC="${SOC:-ascend910_93}"
nssh() { timeout "${TMO:-1800}" ssh -F "$SSH_CFG" -o ConnectTimeout=25 "$HOST_ALIAS" "$@"; }
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

case "${1:-build}" in
  build)
    bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | grep -aE "^[0-9a-f]{32}|FAIL" | tail -8
    # 远端副本专属：SoC 双注册（提交源只有 ascend910b，比赛平台口径不动）
    nssh "cd ~/sfa_real/code && \
      sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT $SOC)/' CMakeLists.txt && \
      sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"$SOC\")|' op_host/sparse_flash_attention.cpp && \
      grep -o 'AddConfig(\"[a-z0-9_]*\")' op_host/sparse_flash_attention.cpp"
    nssh "cd ~/sfa_real && bash build.sh" 2>&1 | grep -aE "构建 OK|CMAKE FAIL|MAKE FAIL|error|Error" | head -20
    LINK="-I\$HOME/Ascend/cann-9.0.0/aarch64-linux/include -I\$HOME/sfa_real/vendor/custom/op_api/include -L\$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -L\$HOME/sfa_real/vendor/custom/op_api/lib -lascendcl -lnnopbase -lcust_opapi"
    # ⚠️ 必须先 source set_env.sh：ld 靠 LD_LIBRARY_PATH 解 libascendcl/libnnopbase 的传递依赖，
    #    否则报一屏 "undefined reference to mmDlsym / ge::AscendString..."（实测踩坑）
    nssh "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real && g++ -std=c++17 -O2 test_sfa_dev.cpp -o test_sfa_dev $LINK 2>&1 | head -20; ls -la test_sfa_dev"
    ;;

  matrix)
    ACT="${2:-diff}"; shift 2 || true
    CASES="${*:-r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads}"
    nssh "$ENVR; p=0; f=0; for n in $CASES; do \
        timeout 900 ./test_sfa_dev cases/\$n.bin 3 $ACT >/tmp/dv_\$n.txt 2>&1; rc=\$?; \
        if [ \$rc -eq 0 ]; then p=\$((p+1)); st=PASS; else f=\$((f+1)); st=FAIL; fi; \
        printf '%-14s %-6s %s\n' \"\$n\" \"\$st\" \"\$(grep -aE 'out :|LSE :|时间|逐位|不逐位' /tmp/dv_\$n.txt | tr '\n' '|')\"; \
      done; echo \"-------- PASS=\$p FAIL=\$f\""
    ;;

  one)
    # ⚠️ 这里必须本地展开用例名：远端 shell 的 $1 恒为空（旧写法把 cases/.bin 发出去，已修）
    shift; C="${1:-r1_min}"; R="${2:-5}"; A="${3:-diff}"
    nssh "$ENVR; ./test_sfa_dev cases/$C.bin $R $A" 2>&1 | tail -15
    ;;

  f32)
    # fp32 模板实例的第二遍（code3.md §15.13）：同一批用例把输入按 float 重送。
    # ⚠️ DT_QUERY=float 是与 fp16 并列的另一条模板实例，只跑 fp16 等于只测了一半。
    ACT="${2:-none}"; shift 2 || true
    CASES="${*:-r1_min r2_chunk r3_mode3 r4_shortkv r5_blocks r6_multiB r7_norope r8_heads}"
    nssh "$ENVR; p=0; f=0; for n in $CASES; do \
        SFA_F32=1 timeout 900 ./test_sfa_dev cases/\$n.bin 3 $ACT >/tmp/dvf_\$n.txt 2>&1; rc=\$?; \
        if [ \$rc -eq 0 ]; then p=\$((p+1)); st=PASS; else f=\$((f+1)); st=FAIL; fi; \
        printf '%-14s %-6s %s\n' \"\$n\" \"\$st\" \"\$(grep -aE 'out :|LSE :|时间|不逐位' /tmp/dvf_\$n.txt | tr '\n' '|')\"; \
      done; echo \"-------- fp32 PASS=\$p FAIL=\$f\""
    ;;

  big)
    nssh "cd ~/sfa_real && source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; python3 gen_big.py big1 256" 2>&1 | tail -4
    ;;
esac
