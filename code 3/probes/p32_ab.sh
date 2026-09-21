#!/bin/bash
# P32 A/B：V 复用 kBuf_ 之后 n_blk=48 第一次可达 —— 但代价模型的选档同时从 nb=2 翻到 nb=1，
# 所以必须把"k 变大"与"nb 变小"两个效应拆开测。同一份构建内用 SFA_FORCE_* 扫格（P31 口径），
# 第二遍把每案的格序反转以抵消场次内漂移。补丁只作用在远端副本 ~/sfa_real/code。
set -u
REPO="/home/fszqsn/ops_comp/ascend-operator-competition"
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
OUT=/tmp/p32_ab.txt
# 把本地 probes/ 推一份到远端 ~（补丁脚本、后续复盘都要用）
tar cf - -C "$REPO/code 3" probes | timeout 180 ssh -F $S -o ConnectTimeout=25 $H "cd ~ && tar xf -" 2>&1 | grep -av Warning || true
ENVR='source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/sfa_real; export ASCEND_CUSTOM_OPP_PATH=$HOME/sfa_real/vendor/custom; export LD_LIBRARY_PATH=$HOME/sfa_real/vendor/custom/op_api/lib:$LD_LIBRARY_PATH'

# 1) 远端补丁 + 重编（本地提交源不动）
timeout 300 ssh -F $S -o ConnectTimeout=25 $H "cd ~/sfa_real/code && \
  sed -i 's/set(ASCEND_COMPUTE_UNIT ascend910b)/set(ASCEND_COMPUTE_UNIT ascend910_93)/' CMakeLists.txt && \
  sed -i 's|\.AddConfig(\"ascend910b\")|.AddConfig(\"ascend910b\").AddConfig(\"ascend910_93\")|' op_host/sparse_flash_attention.cpp && \
  python3 ~/code\ 3/probes/p23_sweep_patch.py op_host/sparse_flash_attention.cpp && \
  grep -c SFA_FORCE_NB op_host/sparse_flash_attention.cpp" 2>&1 | grep -av Warning
timeout 900 ssh -F $S $H "cd ~/sfa_real && bash build.sh" 2>&1 | grep -aE "构建 OK|FAIL|error:" | head -5

# 2) 扫格。单元格 = "nb kb ks"，AUTO 表示不 force（走 host 自己的选档，读 SFA_PICK）
run_cell() {   # $1=case $2=nb $3=kb $4=ks $5=pass
  local cs=$1 nb=$2 kb=$3 ks=$4 ps=$5 envs=""
  if [ "$nb" != AUTO ]; then envs="SFA_FORCE_NB=$nb SFA_FORCE_NBLK=$kb SFA_FORCE_KS=$ks"; fi
  local line
  line=$(timeout 220 ssh -F $S $H "$ENVR; env $envs ./test_sfa_dev cases/$cs.bin 3 none 2>&1 | \
    grep -aoE 'SFA_PICK nb=[0-9]+ nblk=[0-9]+ ks=[0-9]+|平均 [0-9.]+ ms|超差 [0-9]+/[0-9]+|FAIL|ERROR'" 2>&1 | grep -av Warning | tr '\n' ' ')
  printf 'P%s %-6s %-16s | %s\n' "$ps" "$cs" "$nb/$kb/$ks" "$line"
}
: > $OUT
for pass in 1 2; do
  for cs in p1 p2 q1h q2h p4 p6 big1; do
    case $cs in
      p1|p2|q1h|q2h) cells="AUTO 2 32 2|2 48 2|1 48 2|1 40 2" ;;
      p4)            cells="AUTO 4 32 2|4 40 2|4 48 2" ;;
      p6)            cells="AUTO 4 32 1|4 40 1|4 48 1" ;;
      big1)          cells="AUTO 8 32 1|8 40 1" ;;
    esac
    IFS='|' read -ra arr <<< "$cells"
    if [ $pass -eq 2 ]; then rev=(); for ((i=${#arr[@]}-1;i>=0;i--)); do rev+=("${arr[i]}"); done; arr=("${rev[@]}"); fi
    for c in "${arr[@]}"; do
      if [ "$c" = AUTO ]; then run_cell $cs AUTO - - $pass | tee -a $OUT
      else run_cell $cs $c $pass | tee -a $OUT; fi
    done
  done
done
echo "P32_AB_DONE"
