#!/bin/bash
# 真机 preflight（非提交文件）：一次连接把"能不能编、按什么口径编"的判据全部打回来。
# 只读信息，不编译、不跑算子、不改任何文件。
echo "### host"
hostname
uname -srm
echo "### cpu/mem"
nproc
free -g 2>/dev/null | head -2
echo "### npu device"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info 2>&1 | head -30
    echo "--- board info (chip name / arch) ---"
    npu-smi info -t board -i 0 2>&1 | head -20
else
    echo "npu-smi MISSING"
fi
echo "### device nodes"
ls /dev/davinci* /dev/devmm_svm /dev/hisi_hdc 2>&1 | head -12
echo "### set_env.sh candidates"
for p in "$HOME/Ascend/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    [ -f "$p" ] && echo "FOUND $p"
done
SRC=""
for p in "$HOME/Ascend/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh" /usr/local/Ascend/ascend-toolkit/set_env.sh; do
    if [ -f "$p" ]; then source "$p" 2>/dev/null; SRC="$p"; break; fi
done
echo "SET_ENV_USED=$SRC"
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "### tools"
for t in ccec bisheng-compiler atc msopst msprof cmake gcc g++ python3; do
    if command -v "$t" >/dev/null 2>&1; then echo "TOOL $t = $(command -v $t)"; else echo "TOOL $t = MISSING"; fi
done
cmake --version 2>&1 | head -1
ccec --version 2>&1 | head -2
atc --version 2>&1 | tail -2
echo "### ASC cmake module (npu_op_package 依赖)"
find "$ASCEND_HOME_PATH" -maxdepth 6 -name "ASCConfig*.cmake" 2>/dev/null | head -3
find "$ASCEND_HOME_PATH" -maxdepth 5 -type d -name "cmakemod*" 2>/dev/null | head -3
echo "### soc dirs shipped with this CANN (910b? 910c?)"
find "$ASCEND_HOME_PATH" -maxdepth 4 -type d \( -iname "*ascend910*" -o -iname "*davinci_22*" -o -iname "*davinci_35*" \) 2>/dev/null | head -12
echo "### space"
df -h "$HOME" 2>/dev/null | tail -1
echo "### preflight done"
