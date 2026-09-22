#!/bin/bash
# [题1/p20 临时探针，非提交面] c5 固定开销分解第 1 阶：**0 构建、只用现成旋钮**（MHC_SHAPE / MHC_FORCE_BLK）。
#
# 要回答的问题（code1.md §24.5）：反向小档上那笔"与 io 无关"的 ≥2.2µs 到底存不存在。
#   模型：T(io,blk) = A + B·blk + C·io/blk，A=1.10~1.20µs、B=75ns/核、C=19ns/KiB（§23.5，前向标定）
#   ⇒ 若 T 的超出量在不同 io 上**相等**，它就是与 io 无关的固定项；若随 io 走，§24.5 的形状窗推断错。
#
# 为什么取 m=1：反向 io=(m+1)·S·D·2 ⇒ m=1 时每行只有 1 读 + 1 写，把 §22 那本"每核发起次数"账
#   压成常数 ⇒ 剩下的超出量不可能再由发起次数解释。（R18 已证 m=1 是恒等对照的天然材料。）
#
# 装置口径（§23.27-6 硬规定）：同构建两臂 A/B 里"后跑的那一发"固定快 ≈0.45µs ⇒
#   每个 cell 与参考 R 跑**两种块序各一块**：`R,X,X,R` 与 `X,R,R,X`
#   ⇒ 每个臂各占一次"慢位 {1,4}"和一次"快位 {2,3}"，块内均值对相位无偏。
#   外加一条**构造性恒等对照** C ≡ R（同形状同 blk，真增益必为 0）用来读 residual 相位 p。
set +u
cd ~/ops_comp/probe2 || exit 1
SO=npu_debug/pkg/custom/op_api/lib/libcust_opapi.so
strings "$SO" | grep -q MHC_FORCE_BLK || { echo "ABORT: .so 没有旋钮，先跑 apply_h1_knobs.py on + build_npu.sh"; exit 8; }
strings "$SO" | grep -q MHC_BWD_GATHER && { echo "ABORT: H3 gather 探针还开着，会污染反向路径"; exit 9; }

# 参考格：io=96KiB(S=64,m=1,D=384) @ blk=16 —— 正落在 §24.5 推出来的 [48,130]KiB 窗中间
RSH=bwd,fp16,64,384,1
RBLK=16
one() {   # $1=shape $2=blk $3=tag $4=round $5=slot
  local line mean got ver
  line=$(env MHC_SHAPE="$1" MHC_PCASE=0 MHC_REPS=41 MHC_FORCE_BLK="$2" MHC_CORE_FLOOR=8 MHC_MERGE_FLOOR=8 \
        bash npu_debug/prof_matrix.sh "p20_$3_b${2}_r${4}_$5" 0 2>&1 |
        grep -E "剔首 mean=|ALL PASS|FAIL=" | head -2 | tr '\n' '|')
  mean=$(echo "$line" | sed -n 's/.*剔首 mean=\([0-9.]*\).*/\1/p')
  got=$(echo "$line" | sed -n 's/.*blk=\([0-9]*\).*/\1/p')
  ver=$(echo "$line" | grep -oE "ALL PASS|FAIL=[0-9]+" | head -1)
  echo "S $3 b$2 r$4 slot$5 arm=$3 got=$got ${ver:-NOVERDICT} mean=$mean shape=$1"
}
pair() {  # $1=shape $2=blk $3=tag
  one "$RSH" "$RBLK" REF 1 1; one "$1" "$2" "$3" 1 2
  one "$1" "$2" "$3" 1 3; one "$RSH" "$RBLK" REF 1 4
  one "$1" "$2" "$3" 2 1; one "$RSH" "$RBLK" REF 2 2
  one "$RSH" "$RBLK" REF 2 3; one "$1" "$2" "$3" 2 4
}

# 自检一发：形状表能不能被 harness 接受（D=192/384/640 这些非 2 幂档）
one "bwd,fp16,64,192,1" 16 SELFTEST 0 1

# io 阶梯（m=1,S=64 ⇒ io=256·D 字节）：48 / 96 / 160 KiB  ×  blk 4 / 8 / 16 / 32
for blk in 4 8 16 32; do
  for d in 192 384 640; do
    shp="bwd,fp16,64,${d},1"
    kib=$(( d / 4 ))                     # io(KiB) = 256*D / 1024 = D/4
    if [ "$d" = "384" ] && [ "$blk" = "16" ]; then
      pair "bwd,fp16,64,384,1" 16 IDENT  # 构造性恒等对照：X ≡ R ⇒ 真增益必为 0
    else
      pair "$shp" "$blk" "io${kib}k"
    fi
  done
done
echo "### p20_ladder done"
