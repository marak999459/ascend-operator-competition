#!/bin/bash
# 诊断用（非提交）：打印脚本实际收到的位置参数，排查分组驱动拿到 "1000" 的原因
echo "ARGC=$#"
i=0
for a in "$@"; do
    i=$((i + 1))
    echo "ARG[$i]=[$a]"
done
echo "ZERO=[$0]"
echo "PWD=[$(pwd)]"
echo "SHELL_VAR=[$SHELL] BASH_ENV=[$BASH_ENV] ENV=[$ENV]"
echo "BASH_VER=[$(bash --version | head -1)]"
