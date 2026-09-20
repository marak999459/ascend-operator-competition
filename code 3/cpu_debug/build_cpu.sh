#!/bin/bash
# SFA compile gate for a CPU-only box (cloud 仿真机).
# cmake + make runs the real ccec/bisheng cross-compiler for ascend910b, so it
# catches the errors the tikicpulib CPU debug build silently ignores
# (code3.md 4: float/int implicit conversion). Running still needs an NPU.
# ASCII-only on purpose (code3.md 4.4: Chinese literals break on transfer).
set -u
CANN=${CANN:-/home/developer/Ascend/cann-9.0.0}
source "$CANN/set_env.sh" 2>/dev/null
B=${B:-$HOME/sfa_cpu/code}
L=${L:-$HOME/sfa_cpu/logs}
mkdir -p "$L"

cd "$B" || { echo "NO CODE DIR: $B"; exit 2; }
rm -rf build && mkdir build && cd build || exit 2

echo "=== CMAKE ==="
if ! cmake .. -DCMAKE_BUILD_TYPE=Release > "$L/sfa_cmake.log" 2>&1; then
  echo "CMAKE FAIL"; grep -a -i -m8 "error" "$L/sfa_cmake.log" | head -20
  exit 1
fi
echo "cmake ok"

echo "=== MAKE ==="
if ! make -j8 > "$L/sfa_make.log" 2>&1; then
  echo "MAKE FAIL -- error summary:"
  grep -a -i "error" "$L/sfa_make.log" | head -40
  echo "full log: $L/sfa_make.log ($(wc -l < "$L/sfa_make.log") lines)"
  exit 1
fi
echo "BUILD OK"

echo "--- compiler actually used ---"
grep -a -o -m3 "[^ ]*ccec[^ ]*\|bisheng" "$L/sfa_make.log" | head -5
echo "--- kernel objects produced ---"
find "$B/build" -name "*.o" 2>/dev/null | head -5
find "$B/build" -name "*.json" -path "*sparse*" 2>/dev/null | head -3
echo "--- built libs ---"
ls -la "$B/build/libcust_opapi.so" "$B/build/op_host/"*.so 2>/dev/null
echo "--- warning count ---"
grep -a -ci "warning" "$L/sfa_make.log"
