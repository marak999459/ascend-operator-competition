#!/bin/bash
# 真机侧编译：算子包（缺则编）+ ACL 启动器。非提交脚本。
# 防假成功（工作流 §4.6）：构建前后看产物时间戳，error 不吞。
# 不能开 set -u：CANN 的 set_env.sh 直接引用 LD_LIBRARY_PATH/PYTHONPATH 等未定义变量，
# 开了会在 source 时报 "unbound variable" 并让后面所有路径推导失效（2026-09-20 实测）
set +u
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

for f in "$HOME/Ascend/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/cann/cann-9.0.0/set_env.sh" \
         "$HOME/Ascend/ascend-toolkit/set_env.sh" \
         /usr/local/Ascend/cann-9.0.0/set_env.sh; do
    [ -f "$f" ] && { source "$f"; break; }
done
if [ -z "${ASCEND_HOME_PATH:-}" ]; then echo "NO set_env.sh sourced"; exit 2; fi
ARCH=$(uname -m)
INC=$ASCEND_HOME_PATH/${ARCH}-linux/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
echo "### root=$ROOT home=$ASCEND_HOME_PATH arch=$ARCH"

build_pkg() {
    echo "### building op package"
    rm -rf build_out
    cmake -S . -B build_out -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH" > npu_debug/cmake_cfg.log 2>&1 ||
        { echo "CMAKE CONFIGURE FAILED"; tail -20 npu_debug/cmake_cfg.log; exit 3; }
    cmake --build build_out -j8 > npu_debug/cmake_build.log 2>&1
    rc=$?
    grep -a "error:" npu_debug/cmake_build.log | head -20
    [ $rc -eq 0 ] || { echo "OP BUILD FAILED rc=$rc"; exit 3; }
}

# ---- 1) 算子包（vendor + libcust_opapi.so，kernel 二进制已内嵌）----
# 陈旧判定必须看**源文件 mtime**，不能只看"包在不在"：包存在就跳过会让改过的
# kernel 静默不进产物（2026-09-20 实测踩过，产物时间戳 19:40:44 一整天没变）
BEFORE=$(stat -c %Y build_out/libcust_opapi.so 2>/dev/null || echo 0)
NEWEST_SRC=$(find op_kernel op_host CMakeLists.txt -type f \( -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) \
             -newer build_out/libcust_opapi.so 2>/dev/null | head -1)
if [ "$BEFORE" = "0" ]; then
    build_pkg
elif [ -n "$NEWEST_SRC" ]; then
    echo "### stale: $NEWEST_SRC 比 libcust_opapi.so 新 -> 强制重建"
    build_pkg
else
    echo "### op package up-to-date, ts=$(date -d @$BEFORE +%H:%M:%S)"
fi
AFTER=$(stat -c %Y build_out/libcust_opapi.so 2>/dev/null || echo 0)
echo "### libcust_opapi.so ts: $BEFORE -> $AFTER"
[ "$AFTER" = "0" ] && { echo "MISSING libcust_opapi.so"; exit 3; }
nm -D build_out/libcust_opapi.so | grep -a aclnnMhcExpand | head -3

# ---- 1.5) 刷新 npu_debug/pkg：运行期真正加载的是这里，不是 build_out ----
# 为什么每次无条件重建：MhcExpand_<hash>.o 里那段哈希只覆盖 tiling/key，**不覆盖内核源码**，
# 所以改过 kernel 重编后 pkg 里的同名 .o 可以是完全不同的一份二进制 ⇒ 静默跑旧核。
# 2026-09-21 03:36 实测：optA 树 pkg 停在 02:16 那版，把"未打补丁的对照"跑成挂死，
# 连带 R5/R6/R7/R8 四组读数全部作废（都执行的是同一份 02:16 二进制）。
rm -rf npu_debug/pkg
mkdir -p npu_debug/pkg/custom/op_api/lib npu_debug/pkg/custom/op_impl/ai_core/tbe
cp build_out/libcust_opapi.so npu_debug/pkg/custom/op_api/lib/ ||
    { echo "PKG SO COPY FAILED"; exit 5; }
for d in kernel config; do
    [ -d "build_out/tmp/vendors/custom/op_impl/ai_core/tbe/$d" ] &&
        cp -r "build_out/tmp/vendors/custom/op_impl/ai_core/tbe/$d" \
              npu_debug/pkg/custom/op_impl/ai_core/tbe/
done
# 硬不变式：pkg 内嵌 .o 必须与 build_out 逐字节一致，否则绝不放行
PKGO=$(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null |
       awk '{print $1}' | sort | tr -d '\n')
SRCO=$(md5sum build_out/tmp/vendors/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o 2>/dev/null |
       awk '{print $1}' | sort | tr -d '\n')
if [ -z "$SRCO" ] || [ "$PKGO" != "$SRCO" ]; then
    echo "PKG KERNEL MISMATCH pkg=$PKGO src=$SRCO"; exit 5
fi
echo "### pkg synced, kernel .o md5 = $(md5sum npu_debug/pkg/custom/op_impl/ai_core/tbe/kernel/ascend910b/mhc_expand/*.o | awk '{print substr($1,1,12)}' | tr '\n' ' ')"

# ---- 2) ACL 启动器 ----
mkdir -p npu_debug/logs
g++ -std=c++17 -O2 -I build_out/autogen -I "$INC" \
    npu_debug/test_mhc_expand_npu.cpp -o npu_debug/test_npu \
    -L build_out -L "$ASCEND_HOME_PATH/lib64" -L "$ASCEND_HOME_PATH/${ARCH}-linux/lib64" \
    -lnnopbase -lascendcl -ldl 2>&1 | tee npu_debug/build_launcher.log
rc=${PIPESTATUS[0]}
grep -a "error:" npu_debug/build_launcher.log | head -30
if [ $rc -ne 0 ]; then echo "LAUNCHER BUILD FAILED rc=$rc"; exit 4; fi
ls -la --time-style=+%H:%M:%S npu_debug/test_npu
echo "### build_npu done"
