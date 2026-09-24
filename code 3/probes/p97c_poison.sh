#!/bin/bash
# P97c：毒化假说定量化 —— 一次 w3 挂死会不会让后续【与它无关】的 cube launch 也失败？
#
# 为什么先做这一发：P97 的对照读数里，head(P91 已验证态) 与 p96 **同样**在 p6/k_n16s64 上
# 确定性失败，而 20 分钟前同一份 p96 字节的 p6 还能跑完 ⇒ 单发读数不可信。回看全部失败
# 序列，有一条共同点：**每一批失败都排在一次 w3 挂死之后**，而 cube 关掉（不可能挂死）的
# 那一轮四个例全部跑完。若这条成立，那么"cube 线的任何 A/B 读数"都必须配一套干净的场次
# 协议，否则 P95/P96 都是在沙子上做减法。
#
# 读数口径：只用 p6（cube 臂里最小、最便宜的例）当"探针"，rc=0/1 都算活着、rc=2/124 算死。
set -u
REPO=/home/fszqsn/ops_comp/ascend-operator-competition
H=devenvc_3gfcn.bd23984d636345e490f7c1d42dbef049.atomgit.0
S=$HOME/.atomgitdevenv/.ssh/config
ns() { timeout "${T:-300}" ssh -F $S -o ConnectTimeout=25 $H "$@" 2>&1 | grep -av Warning; }
smi() { ns "source ~/Ascend/cann-9.0.0/set_env.sh 2>/dev/null; npu-smi info 2>/dev/null | sed -n '7p;8p' | tr -s ' ' | tr '\n' '|'"; }

# 远端逻辑推上去单独成文件：上一版把模板塞进本地 printf，远端内层的 %s 被一并吃掉 ⇒ 四遍读数全空。
cat "$REPO/code 3/probes/p97_probe.sh" | ns "mkdir -p ~/p97 && cat > ~/p97/probe.sh && wc -l < ~/p97/probe.sh"
echo "########## 0) kernel sha $(date +%H:%M:%S) ##########"
ns "cd ~/sfa_real && sha256sum code/op_kernel/sparse_flash_attention.cpp | cut -c1-12"
smi
echo "########## 1) 干净窗口 p6x5 $(date +%H:%M:%S) ##########"
ns "bash ~/p97/probe.sh p6 5 60"
echo "########## 2) 制造挂死 w3 (timeout 90 后 SIGKILL) $(date +%H:%M:%S) ##########"
ns "bash ~/p97/probe.sh w3 1 90"
echo "########## 3) 挂死后立刻 p6x5 $(date +%H:%M:%S) ##########"
ns "bash ~/p97/probe.sh p6 5 60"
echo "########## 4) 等 120s 后 p6x4 $(date +%H:%M:%S) ##########"; sleep 120
ns "bash ~/p97/probe.sh p6 4 60"
echo "########## 5) 再等 240s 后 p6x4 $(date +%H:%M:%S) ##########"; sleep 240
ns "bash ~/p97/probe.sh p6 4 60"
echo "########## 6) 结束健康 $(date +%H:%M:%S) ##########"
smi
