#!/bin/bash
# 本地看门狗（非提交）：低频轮询云端仿真机，一旦 sshd 能应答就把分组日志抓回本地目录。
# 用途：large 组会把 16 核压满 -> ssh banner 超时；此时不能重试轰炸，也不能把日志只留在容器上。
HOST=devenvc_tpm0u.724ca0f549d7458fb9ffa7e903a49689.atomgit.0
ROOT="D:/Projects/算子比赛"
OUT="$ROOT/code1/cpu_debug/logs"
RLOG='~/ops_comp/code1/cpu_debug/logs'
mkdir -p "$OUT"
log() { echo "$(date +%H:%M:%S) $*" >> "$OUT/pull_watch.log"; }
grab() {
    ssh -o ConnectTimeout=25 "$HOST" "cd $RLOG && md5sum *_r4_*.log" > "$OUT/remote_md5_r4.txt" 2>/dev/null
    ssh -o ConnectTimeout=25 "$HOST" "cd $RLOG && tar cf - *_r4_*.log" 2>/dev/null | tar xf - -C "$OUT/" 2>/dev/null
    log "pulled $(grep -c . "$OUT/remote_md5_r4.txt") logs, local now $(ls "$OUT" | grep -c '_r4_.*\.log')"
}
for i in $(seq 1 24); do
    if ssh -o ConnectTimeout=25 -o BatchMode=yes "$HOST" "cat $RLOG/driver.log" > "$OUT/driver_r4.log" 2>/dev/null; then
        log "attempt $i ssh OK"
        grab
        if grep -q ALL_DONE "$OUT/driver_r4.log"; then
            log "ALL_DONE reached -> final check"
            ssh -o ConnectTimeout=25 "$HOST" "cd $RLOG && cat driver.log" > "$OUT/driver_r4.log" 2>/dev/null
            grab
            log "DONE"
            exit 0
        fi
    else
        log "attempt $i ssh busy/unreachable"
    fi
    sleep 180
done
log "GAVE UP after 24 attempts"
