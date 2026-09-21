#!/bin/bash
# 探测本机 CANN 9.0.0 到底提供哪些归约 / 广播 / 逐元素原语（决定 kernel 能用哪条路）
# 只读头文件，不编译、不跑核。仿真机或真机都可执行。
CANN_ENV=""
for p in "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann-9.0.0/set_env.sh" "$HOME/Ascend/cann/set_env.sh"; do
    [ -f "$p" ] && CANN_ENV="$p" && break
done
[ -n "$CANN_ENV" ] || { echo "ERROR: set_env.sh not found"; exit 1; }
source "$CANN_ENV"

ROOT="$ASCEND_HOME_PATH/$(uname -m)-linux"
[ -d "$ROOT" ] || { echo "ERROR: no $ROOT"; exit 1; }
ASC="$ROOT/asc"
ADC="$ROOT/ascendc"
OUT="$(cd "$(dirname "$0")" && pwd)/probe_cann_api.txt"
: > "$OUT"

{
echo "### CANN=$ASCEND_HOME_PATH ARCH=$(uname -m)"
echo
echo "=== 1) ReduceSum / ReduceMax 重载签名（关注是否有 workBuffer / keepDims / isAutoSync） ==="
grep -rn --include=*.h --include=*.hpp -E "^\s*(template|__aicore__).*Reduce(Sum|Max)" "$ASC" "$ADC" 2>/dev/null \
  | grep -v "impl/" | head -40
echo
echo "--- ReduceSum 声明原文（含模板参数，取前 3 处上下文） ---"
grep -rn -A3 "inline void ReduceSum(" "$ASC" "$ADC" 2>/dev/null | head -40
echo
echo "=== 2) MoveMask / SetVectorBuf / Queue 广播三件套 ==="
grep -rn --include=*.h --include=*.hpp -E "MoveMask|SetVectorBuf" "$ASC" "$ADC" 2>/dev/null \
  | grep -E "inline|template" | head -20
echo
echo "=== 3) 是否存在现成的 softmax / 归一化高阶 API ==="
find "$ASC" "$ADC" -iname "*softmax*" -o -iname "*layer_norm*" 2>/dev/null | head -10
echo
echo "=== 4) 逐元素 Add/Div 的非 32B 偏移是否放行（找对齐约束注释） ==="
grep -rn -B2 -A6 "align" "$ASC/include/basic_api" 2>/dev/null | grep -iE "32|align|对齐" | head -20
echo
echo "=== 5) Exp / Reciprocal / Cast 可用性 ==="
grep -rn --include=*.h --include=*.hpp -E "inline void (Exp|Reciprocal)\(" "$ASC" "$ADC" 2>/dev/null | head -12
echo
echo "=== 6) 仿真机是否已实现 MoveMask（CPU debug 路径） ==="
grep -rln "MoveMask" "$ROOT/../toolkit/tools/tikicpulib" 2>/dev/null | head -5
} 2>&1 | tee "$OUT"

echo "----"
echo "完整输出: $OUT"
