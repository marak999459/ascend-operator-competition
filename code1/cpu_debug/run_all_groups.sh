#!/bin/bash
# 全量分组仿真驱动（非提交）：在 code1/ 根目录下由 setsid nohup 拉起，逐组写独立日志。
# 用法：bash cpu_debug/run_all_groups.sh [组1 组2 ...]（不传参 = 5 组全跑）
# 变量名不能叫 GROUPS：bash 会把它重置成当前用户的 gid 列表（容器 developer = 1000），
# 实测 2026-09-20 因此把 5 组跑成 1 个不存在的组名，且 harness 对未知组名回 "ALL PASS"。
cd "$(dirname "$0")/.." || exit 1
LOGDIR=cpu_debug/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M)
VALID="quick medium mtile ub48 large full"
MODES=("$@")
[ ${#MODES[@]} -gt 0 ] || MODES=(quick medium mtile ub48 large)
for m in "${MODES[@]}"; do
    case " $VALID " in
        *" $m "*) ;;
        *) echo "$m SKIP bad-mode $(date +%H:%M:%S)" >> "$LOGDIR/driver.log"; continue ;;
    esac
    LF="$LOGDIR/${m}_r4_${STAMP}.log"
    echo "$m START $(date +%H:%M:%S)" >> "$LOGDIR/driver.log"
    bash "cpu_debug/run_cpu.sh" 20 "$m" > "$LF" 2>&1
    rc=$?
    n=$(grep -c 'maxdiff=' "$LF")
    echo "$m DONE rc=$rc cases=$n $(date +%H:%M:%S)" >> "$LOGDIR/driver.log"
    [ "$n" -gt 0 ] || echo "$m SUSPECT zero-cases" >> "$LOGDIR/driver.log"
done
echo "ALL_DONE $(date +%H:%M:%S)" >> "$LOGDIR/driver.log"
