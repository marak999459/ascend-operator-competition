#!/bin/bash
# P127 闸门（钉档字节 = 这一发要发货的字节，探针 printf 一个都不加）：
#   p127_pin.py apply（已做）→ npu.sh sync → 远端命中数门 + 四枚 sha（sed 之前）
#   → SoC sed（只在远端副本）→ build → 探活
#   → ④ 补两个"头数 ≥16"的用例（现有 26 例最大 qN=8，而这一发把 nb 抬到 16、n_blk 压到 32 ⇒ 必须真跑）
#   → ⑤ **时间指纹对表**：不打印选档（那要往提交源加 printf），改用 P126 两臂已经量到的
#        "每一档的批量时间"当指纹：p1 auto(nb=2)=0.1192 / nb=4=0.1298 / 退化档=0.3754
#        ⇒ 钉档生效应当落在 ≈0.1298 那一侧；落在 0.375 一侧 = 掉进 L434 退化档 = 硬失败。
#   → ⑥ 全量数值门（p32_gate.sh：13 golden 双 dtype + 13 真 expect 双 dtype）+ 两枚新用例双 dtype
# ⛔ 本脚本不做"最后一次 sync"，也不 submit —— 那是 p127_precheck.sh / p127_submit.sh 的活。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'
nssh() { timeout "${TMO:-3000}" ssh -F "$S" -o ConnectTimeout=25 "$H" "$@"; }
HOST_SHA_PINNED=f0a25b0771b644d0
KSHAS="f815bf1eaba0f8bc f2b28a86e998c194 1046b349538f3007"

echo "########## ① 本地钉档状态 + sync ##########"
python3 "$REPO/code 3/probes/p127_pin.py" check
bash "$REPO/code 3/npu_debug/npu.sh" sync 2>&1 | grep -aE "md5|=== |OK|FAIL" | tail -8

echo "########## ② 命中数门 + 四枚 sha（远端副本，sed 之前） ##########"
nssh "cd ~/sfa_real/code && \
  echo P127=\$(grep -c 'P127 单元数下界档' op_host/sparse_flash_attention.cpp) \
       OLDANCHOR=\$(grep -c 'cost \* 20ULL' op_host/sparse_flash_attention.cpp) \
       BC0=\$(grep -c 'bestCost == 0' op_host/sparse_flash_attention.cpp) \
       OLDTAB=\$(grep -c '{32, 16, 8, 4, 2, 1}' op_host/sparse_flash_attention.cpp) \
       FORBID=\$(grep -acE 'printf|fflush|fprintf|std::cout|cerr|TODO|FIXME|#if 0|调试' op_host/sparse_flash_attention.cpp) ; \
  sha256sum op_host/sparse_flash_attention.cpp op_kernel/sparse_flash_attention.cpp \
            op_kernel/sparse_flash_attention_tiling.h op_kernel/tiling_key_sparse_flash_attention.h | cut -c1-16" 2>&1 | grep -av Warning
echo ">>> 期望 P127=1 OLDANCHOR=0 BC0=2 OLDTAB=1 FORBID=0 ；host=$HOST_SHA_PINNED，kernel 三枚 = $KSHAS"

echo "########## ③ SoC sed（只在远端）+ build ##########"
nssh "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  echo HITS_SOC=\$(grep -c ascend910_93 op_host/sparse_flash_attention.cpp)" 2>&1 | grep -av Warning
nssh "$ENVR; bash build.sh 2>&1 | grep -aE '构建 OK|FAIL|error' | head -5" 2>&1 | grep -av Warning

echo "########## ④ 探活 + 补 qN>=16 用例 ##########"
ALIVE=0
for try in 1 2 3 4 5 6; do
  if nssh "$ENVR; ./test_sfa_dev cases/p6.bin 2 none" 2>&1 | grep -qa 批量口径; then
    echo ">>> 探活第 $try 次 OK"; ALIVE=1; break
  fi
  echo ">>> 探活第 $try 次失败，等 60 s"; sleep 60
done
[ "$ALIVE" = "1" ] || { echo ">>> 6 次探活全失败 ⇒ 中止"; exit 1; }
( cd "$REPO/code 3/probes" && tar cf - gen_n16.py ) | nssh "cd ~/sfa_real && tar xf - && \
  ls cases/pn16a.bin cases/pn32a.bin 2>/dev/null | wc -l" 2>&1 | grep -av Warning
nssh "$ENVR; python3 gen_n16.py 2>&1 | tail -3" 2>&1 | grep -av Warning

echo "########## ⑤ 时间指纹对表（REPS=${REPS:-5}；auto 值取自本目录 p126_ab.txt 同机两臂） ##########"
# ⚠️ 只有"带 golden 或带真 expect"的用例才跑 diff：w3/r64c65 两者都没有，diff 打空会被
#    读成"逐位 0/3"（P126 的探针自欺就是这个口径，见 LOG 已知为错表）。
NUMSET="${NUMSET:-p1 p2 p4 p6 big1 pn16a pn32a}"
printf '%-9s %14s  %s\n' case 批量 数值
for c in ${FINGER_CASES:-p1 p2 p4 p6 big1 w3 r64c65 pn16a pn32a}; do
  bm=$(nssh "$ENVR; ./test_sfa_dev cases/$c.bin ${REPS:-5} none 2>&1" 2>&1 \
        | grep -a 批量口径 | head -1 | sed 's/.*平均 //; s/ *最小.*//')
  dif="-" bit="-" bad="-"
  case " $NUMSET " in
    *" $c "*)
      out=$(nssh "$ENVR; ./test_sfa_dev cases/$c.bin 1 diff 2>&1" 2>&1 | grep -av "^Warning")
      dif=$(printf '%s' "$out" | grep -aoE '超差 [0-9]+/[0-9]+' | tr '\n' ' ')
      bit=$(printf '%s' "$out" | grep -a 逐位一致 | grep -av 不逐位一致 | wc -l)
      bad=$(printf '%s' "$out" | grep -ac 不逐位一致) ;;
    *) bit="无golden" ;;
  esac
  printf '%-9s %14s  超差[%s] 逐位%s 不逐位%s\n' "$c" "$bm" "$dif" "$bit" "$bad"
done
echo ">>> 参考（P126 同机两臂）：p1 auto=0.1192 / nb=4=0.1298 / 退化档=0.3754 ；"
echo ">>>                  p2/p4/w3/r64c65 钉档后应**与 auto 同值**（镜像：可证惰性），"
echo ">>>                  big1 应落在 nb=8 那一档（P109 网格：比 auto 慢 6.4 %）"

echo "########## ⑥ 全量数值门（p32_gate.sh 26 例双 dtype） ##########"
bash "$REPO/code 3/probes/p32_gate.sh" 2>&1 | tee "$REPO/code 3/probes/p127_gate.txt" | grep -av Warning
echo "########## ⑥b 两枚新用例双 dtype ##########"
for f in 0 1; do
  tag=$([ "$f" = 1 ] && echo fp32 || echo fp16)
  for c in pn16a pn32a; do
    line=$(nssh "$ENVR; env $( [ "$f" = 1 ] && echo SFA_F32=1 ) ./test_sfa_dev cases/$c.bin 1 diff 2>&1" 2>&1 \
      | grep -aoE '超差 [0-9]+/[0-9]+|逐位一致 [^ ]+|不\*\*逐位一致\*\*|FAIL|PASS' | tr '\n' ' ')
    printf '%-5s %-7s %s\n' "$tag" "$c" "$line"
  done
done
echo "P127_GATE_DONE"
