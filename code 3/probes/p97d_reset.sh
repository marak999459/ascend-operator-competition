#!/bin/bash
# P97d：设备恢复尝试 —— 不等看门狗，直接用 aclrtResetDevice() 清掉本容器的设备上下文。
#
# 触发条件：P97c 里 cube 臂已经**全死**（p6 连 10 遍 rc=2，w3 也秒挂），而 18:49 那一轮
# 同一份字节还能跑完 p6 ⇒ 状态是被前面几次挂死/SIGKILL 累积毒化的，不是代码回退。
# 先观察"光等"能不能自愈（P97c 的 4)/5) 档），不行再试这条 —— 它只影响我们自己那块卡
# （npu-smi 里只有 NPU 3，且 "No running processes found"），不动主机、不重置整芯片。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ns() { timeout "${T:-300}" ssh -F $S -o ConnectTimeout=25 $H "$@" 2>&1 | grep -av Warning; }
smi() { ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; npu-smi info 2>/dev/null | sed -n '7p;8p' | tr -s ' ' | tr '\n' '|'"; }

echo "########## 0) 复位前健康 $(date +%H:%M:%S) ##########"; smi
cat "$REPO/code 3/probes/p97_reset.c" | ns "mkdir -p ~/p97 && cat > ~/p97/reset.c && echo pushed"
ns 'source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; cd ~/p97 && \
    g++ -x c++ reset.c -o reset -I$HOME/Ascend/cann-9.0.0/aarch64-linux/include \
      -L$HOME/Ascend/cann-9.0.0/aarch64-linux/lib64 -lascendcl 2>&1 | head -5; \
    echo "---- 跑复位（0=ACL_SUCCESS）----"; timeout 120 ./reset; echo "rc=$?"'
echo "########## 1) 复位后立刻 p6x3 $(date +%H:%M:%S) ##########"
ns "bash ~/p97/probe.sh p6 3 60"
echo "########## 2) 再 p6x3 $(date +%H:%M:%S) ##########"
ns "bash ~/p97/probe.sh p6 3 60"
# ⛔ 这里刻意不放 w3：§7 第 8 条的 ② —— 会挂的档塞在批次中间，下一次读到的是"被自己毒死的
#    设备"。要读 w3 请单独开一批，放在该批最后一次。
echo "########## 3) 纯 AIV 对照 p1x2 $(date +%H:%M:%S) ##########"
ns "bash ~/p97/probe.sh p1 2 60"
echo "########## 4) 结束健康 ##########"; smi
