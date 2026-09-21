# DevSpace 隧道自举脚本：不起 VS Code，按需拉起云端开发环境的本地转发并验证。
# 能力边界：能救"环境还活着、隧道没起来/断了"；救不了"环境已被回收"（那需要 OAuth 登录态，只能在 IDE 或网页点一次 Start）。
# ⚠️ 本文件**必须存成 UTF-8 with BOM**：PowerShell 5.1 无 BOM 时会把中文注释的字节读乱，
#    连下面的 param 块都解析失败（实测报 "表达式或语句中包含意外的标记 ')'"）。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File devspace_tunnel.ps1 -List              # 只看状态
#   ... -File devspace_tunnel.ps1 -Role cpu                                                   # 全部**可派活**的 CPU 环境
#   ... -File devspace_tunnel.ps1 -Role npu -Env 02aeb                                        # 只起某一个（点名可越过 use 闸门）
#   ... -File devspace_tunnel.ps1 -Role cpu -Probe                                            # 顺带实测角色（防表写错）
#   ... -File devspace_tunnel.ps1 -Role cpu -Watch -IntervalSec 60                            # 保温
#
# 派活闸门：$EnvTable 里 use='free' 才会被 `-Role` 选中；use='reserved'（正给别人当通道）
#           和未登记的新环境只能 `-Env <短名>` 显式点名。加新环境 = 在 $EnvTable 补一行。
param(
    [ValidateSet('all', 'cpu', 'npu', 'unknown', 'deleted')]
    [string]$Role = 'all',         # 角色筛选：cpu / npu / all(=cpu+npu) / unknown / deleted
    [string]$Env = '',             # 按环境短名精确/子串筛选（如 02aeb）；留空 = 该角色全部
    [switch]$List,                 # 只报告状态，不动手
    [switch]$Probe,                # 连上后实测角色（NPU 设备节点 / 架构）
    [switch]$Diag,                 # 附带 bootstrap.log 判读
    [switch]$Watch,                # 保温循环
    [int]$IntervalSec = 60,
    [int]$Iterations = 20,
    [int]$WaitSec = 40
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# ===== 环境表（2026-09-20 用户逐一口述确认；**加新环境只改这里，一行一个**）=====
# 键 = 别名短名（小写，取自 devenvc_<短名>.<envId>）
#   role     = cpu / npu           （没登记的会解析成 unknown，永远不会被 -Role 选中）
#   use      = free（可派活）/ reserved（正给别人当通道，-Role 会跳过，必须 -Env 点名）
#   note     = 显示在 -List 的备注列
$EnvTable = @{
    'tpm0u' = @{ role='cpu'; use='free';     note='题1 全量仿真用（large 组会把 16 核压满 -> ssh banner 超时属预期）。⚠️ 18:56 起当前登录账号列表里查不到它的 devEnvId => 需桌面 VS Code 切回它所属账号' }
    'e6z6k' = @{ role='cpu'; use='free';     note='18:53 实测可自举成功（forward.ready）；下午那次 no longer exists 是账号可见性问题而非环境回收 => 与 tpm0u 分属不同账号，谁可见取决于 VS Code 当前登录态' }
    '02aeb' = @{ role='npu'; use='free';     note='NPU 真机：隧道可自举，但上机跑东西前仍按纪律先问用户' }
    'bna7c' = @{ role='deleted'; use='no';   note='已删除，不要再用' }
}
# 用户提到过 "3GFCN"（说是 NPU），但 config 里没有它的条目 => 从没在 IDE 打开过；出现后补进 $EnvTable
$NamedButAbsent = @('3gfcn')

$Base    = Join-Path $env:USERPROFILE '.atomgitdevenv'
$Config  = Join-Path $Base '.ssh\config'
$BootDir = Join-Path $Base 'forward-bootstrap\vscode'
$BootLog = Join-Path $BootDir 'bootstrap.log'
$HubDir  = Join-Path $Base '.hub\vscode'

if (-not (Test-Path $Config)) { Write-Host "[X] 找不到 SSH config：$Config"; exit 1 }

function Test-LocalPort([int]$Port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try {
        $r = $c.BeginConnect('127.0.0.1', $Port, $null, $null)
        return ($r.AsyncWaitHandle.WaitOne(400) -and $c.Connected)
    } catch { return $false } finally { $c.Close() }
}

# ===== 解析 SSH config：别名 / 端口；环境 ID = 别名里的 32 位十六进制段 =====
function Get-Envs {
    $raw = @(); $cur = $null
    foreach ($line in (Get-Content $Config)) {
        $t = $line.Trim()
        if ($t -like 'Host *') {
            if ($cur -and $cur.EnvId) { $raw += $cur }
            $alias = $t.Substring(5).Trim()
            $m = [regex]::Match($alias, '([0-9a-f]{32})')
            $cur = [pscustomobject]@{
                Alias = $alias; Short = ''; Port = 0; EnvId = ''; JsonPath = ''; CmdPath = ''
                Role = 'unknown'; Use = ''; Note = ''
            }
            if ($m.Success) {
                $cur.EnvId = $m.Value
                $cur.Short = ($alias -replace '^devenvc_', '' -replace '\..*$', '')
                $k = $cur.Short.ToLower()
                if ($EnvTable.ContainsKey($k)) {
                    $e2 = $EnvTable[$k]
                    $cur.Role = $e2.role; $cur.Use = $e2.use; $cur.Note = $e2.note
                } else {
                    $cur.Use = 'unlisted'   # 没登记 -> 不派活，只提示
                }
            }
            continue
        }
        if (-not $cur) { continue }
        if ($t -match '^Port\s+(\d+)') { $cur.Port = [int]$Matches[1] }
    }
    if ($cur -and $cur.EnvId) { $raw += $cur }
    foreach ($i in $raw) {
        $i.JsonPath = Join-Path $BootDir ($i.EnvId + '.json')
        $i.CmdPath  = Join-Path $BootDir ($i.EnvId + '.cmd')
    }
    return $raw
}

function Get-HubPid {
    if (-not (Test-Path $HubDir)) { return $null }
    $f = Get-ChildItem $HubDir -Filter '*.LOCK' -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $f) { return $null }
    try { return (Get-Content $f.FullName -Raw | ConvertFrom-Json).pid } catch { return $null }
}

function Get-CfgAgeHour($path) {
    if (-not (Test-Path $path)) { return -1 }
    try {
        $j = Get-Content $path -Raw | ConvertFrom-Json
        if (-not $j.updatedAt) { return -1 }
        $e = [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$j.updatedAt).UtcDateTime
        return [math]::Round(((Get-Date).ToUniversalTime() - $e).TotalHours, 1)
    } catch { return -1 }
}

function Report($e) {
    $hasJson = Test-Path $e.JsonPath
    $age = Get-CfgAgeHour $e.JsonPath
    $note = [string]$e.Note
    if ($note.Length -gt 44) { $note = $note.Substring(0, 44) + '...' }
    [pscustomobject]@{
        环境     = $e.Short
        角色     = $e.Role
        派活     = $e.Use
        端口     = $e.Port
        监听     = $(if (Test-LocalPort $e.Port) { 'UP' } else { 'down' })
        转发配置 = $(if ($hasJson) { "${age}h 前" } else { '缺失' })
        自举脚本 = $(if (Test-Path $e.CmdPath) { '有' } else { '缺失' })
        备注     = $note
    }
}

$FatalBoot = 'no longer exists|forwardBootstrap.unhandled|connectUrlUnavailable|timeout waiting for refreshed connect_url'

function Boot($e) {
    Write-Host "[>] $($e.Short) : 自举转发"
    $startUtc = (Get-Date).ToUniversalTime()
    $p = Start-Process -FilePath $env:ComSpec -ArgumentList '/c', ('"' + $e.CmdPath + '"') `
                       -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds($WaitSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-LocalPort $e.Port) { return $true }
        # 服务端已判死刑就别空等：只认本次启动之后的日志行（按 time 字段比对，避免上一条失败记录误伤）
        if (Test-Path $BootLog) {
            try {
                foreach ($l in @(Get-Content $BootLog -Tail 6)) {
                    if ($l -notmatch '"time":"([^"]+)"') { continue }
                    $ts = [DateTime]::Parse($matches[1], [System.Globalization.CultureInfo]::InvariantCulture).ToUniversalTime()
                    if ($ts -ge $startUtc -and $l -match $FatalBoot) { return $false }
                }
            } catch {}
        }
        Start-Sleep -Seconds 2
    }
    if ($p.HasExited -and $p.ExitCode -ne 0) { Write-Host "    bootstrap 退出码 = $($p.ExitCode)" }
    elseif (-not $p.HasExited) { Write-Host "    bootstrap 仍在运行（多半在等服务端刷新 connect_url），已等 ${WaitSec}s" }
    return (Test-LocalPort $e.Port)
}

$Noise = 'post-quantum|Permanently added|store now|may need|^Warning|__ALIVE__|__PROBE__'

function Verify($e) {
    $cmd = 'echo __ALIVE__; hostname; nproc; free -g | sed -n 2p'
    if ($Probe) {
        $cmd += '; if [ -e /dev/davinci0 ] || command -v npu-smi >/dev/null 2>&1; then echo "__PROBE__ npu"; else echo "__PROBE__ cpu"; fi; ls /dev/davinci* 2>/dev/null | head -3; uname -m'
    }
    # PS 5.1 会把原生命令的 stderr（如 ssh 的 known hosts 提示）在 ErrorActionPreference=Stop 下抛成异常 → 局部降级
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $out = @(ssh -F "$Config" -o ConnectTimeout=15 -o BatchMode=yes $e.Alias $cmd 2>&1 | ForEach-Object { "$_" })
    $ErrorActionPreference = $prevEap
    # ⚠️ 数组上的 -notmatch 是"过滤出元素"不是"取反布尔"（PS 语义坑）：必须先 join 成整串再判
    $joined = $out -join "`n"
    if ($joined -notmatch '__ALIVE__') {
        $out | Select-Object -Last 3 | ForEach-Object { Write-Host "    $_" }
        return $false
    }
    $out | Where-Object { $_ -and $_ -notmatch $Noise } | ForEach-Object { Write-Host "    $_" }
    if ($Probe) {
        $m = $out | Where-Object { $_ -match '__PROBE__' } | Select-Object -First 1
        $real = ($m -replace '.*__PROBE__\s*', '').Trim()
        if ($real -ne $e.Role) { Write-Host "    [!] 实测角色 = $real，与 `$EnvTable 里的 $($e.Role) 不一致 -> 请更正表里的 role" }
        else { Write-Host "    [i] 实测角色 = $real（与角色表一致）" }
    }
    return $true
}

# 扩展自己的日志比 bootstrap.log 更准：它能区分"环境在列表里但没开机"和"当前登录账号里查不到这个环境"
function EnvListCheck($envs) {
    $log = Join-Path $Base 'logs\vscode\latest.log'
    if (-not (Test-Path $log)) { return }
    foreach ($e in $envs) {
        if (-not $e.EnvId) { continue }
        $hits = @(Get-Content $log -Tail 600 | Where-Object {
            $_ -match ([regex]::Escape($e.EnvId)) -and $_ -match 'envNotRunning|deleted\.requestFailed|envListUnavailable' })
        if (-not $hits) { continue }
        $last = $hits[-1]
        $st = [regex]::Match($last, '"status":\s*(\d+)')
        $tm = [regex]::Match($last, '"time":"([^"]+)"')
        $when = if ($tm.Success) { $tm.Groups[1].Value } else { '?' }
        if ($st.Success) {
            Write-Host "[i] $($e.Short) 扩展日志：环境**在列表里**但 status=$($st.Groups[1].Value)（$when）=> 控制台点『启动』即可"
        } else {
            Write-Host "[i] $($e.Short) 扩展日志：账号环境列表里**查不到** devEnvId $($e.EnvId.Substring(0,8))…（$when）"
            Write-Host "    => 十有八九是**桌面 VS Code 现在登录的账号不是它的主人**（切过账号），不是环境被回收"
        }
    }
}

function Diagnose {
    Write-Host "`n===== bootstrap.log 判读（最后 25 行）====="
    if (-not (Test-Path $BootLog)) { Write-Host "没有日志文件：$BootLog"; return }
    $tail = @(Get-Content $BootLog -Tail 25)
    $tail | ForEach-Object { if ($_.Length -gt 170) { Write-Host ($_.Substring(0, 170) + ' ...') } else { Write-Host $_ } }
    $joined = $tail -join "`n"
    Write-Host "`n----- 结论 -----"
    if ($joined -match 'no longer exists|bootstrap-forward-status|timeout waiting for refreshed connect_url|connectUrlUnavailable') {
        Write-Host '[X] 服务端换不出 connect_url。先区分下面两种（处置完全不同）：'
        Write-Host '    · 环境**已关机** => 控制台点『启动』，隧道随后可自举'
        Write-Host '    · 当前登录账号**列表里没这个环境** => 桌面 VS Code 要切回它所属账号，并点一次『连接』重新签发转发凭据'
    } elseif ($joined -match 'ECONNREFUSED|socket hang up|network|ETIMEDOUT') {
        Write-Host '[!] 网络/服务端不可达 => 可重试（网络恢复后再跑本脚本）。'
    } else {
        Write-Host '[!] 未见 404 特征 => 更像转发侧抖动，重试本脚本；仍不通就核对 hub pid 与端口占用。'
    }
    EnvListCheck $all
}

# ===== 主流程 =====
$all = @(Get-Envs)
# Role=all 只含 cpu + npu（deleted / unknown 一律不入选，须显式点名）
$allowed = if ($Role -eq 'all') { @('cpu', 'npu') } else { @($Role) }
$byRole = @($all | Where-Object { $allowed -contains $_.Role })
# 闸门：-Role 只派给 use='free' 的环境；reserved / unlisted / no 必须用 -Env 显式点名
$gate = @($byRole | Where-Object { $_.Use -ne 'free' })
if ($Env) {
    # 点名 = 越过 role 和 use 两道筛选（新环境还没登记时也能连），只做提示不做拦截
    $pick = @($all | Where-Object { $_.Short -like "*$Env*" })
    foreach ($e in $pick) {
        if ($e.Use -ne 'free') { Write-Host "[!] $($e.Short) 标记为 use=$($e.Use)（$($e.Note)）—— 你用 -Env 点名了，按你的意思继续" }
        if ($allowed -notcontains $e.Role) { Write-Host "[!] $($e.Short) 的 role=$($e.Role) 不在本次 Role=$Role 范围内 —— 同上，点名优先" }
    }
} else {
    $pick = @($byRole | Where-Object { $_.Use -eq 'free' })
}

$gated = @($gate | Where-Object { $pick.Short -notcontains $_.Short })
Write-Host "DevSpace 隧道  ·  config = $Config  ·  Role=$Role  Env=$(if($Env){$Env}else{'*'})  ·  可用 $($pick.Count) 个 / 被闸门挡下 $($gated.Count) 个 / config 共 $($all.Count) 个"
$all | ForEach-Object { Report $_ } | Format-Table -AutoSize | Out-String -Width 220 | Write-Host

$orphans = @($EnvTable.Keys | Where-Object { $all.Short -notcontains $_ })
if ($orphans) { Write-Host "[i] 表里有、config 里没有的短名：$($orphans -join ', ')（环境未创建或没在 IDE 打开过）" }
$unlisted = @($all | Where-Object { $_.Use -eq 'unlisted' })
foreach ($u in $unlisted) { Write-Host "[+] **新环境** $($u.Short)（端口 $($u.Port)）config 里有但表里没登记 -> 在 `$EnvTable 补一行：'$($u.Short)' = @{ role='cpu 或 npu'; use='free'; note='...' }" }
if ($NamedButAbsent) { Write-Host "[i] 用户提过但 config 无条目：$($NamedButAbsent -join ', ') -> 在 IDE 里打开过一次后，短名补进 `$EnvTable" }
foreach ($s in @($all | Where-Object { $allowed -notcontains $_.Role })) { Write-Host "[-] 角色不匹配 $($s.Short) (role=$($s.Role))：$($s.Note)" }
foreach ($s in $gated) { Write-Host "[门] 跳过 $($s.Short) (use=$($s.Use))：$($s.Note)  -> 要用它必须 -Env $($s.Short)" }
$hub = Get-HubPid
if ($hub) { Write-Host "转发 hub pid = $hub" }
if ($List) { if ($Diag) { Diagnose }; exit 0 }
if (-not $pick) {
    Write-Host "[X] 没有可派活的环境（role 匹配的为 0，或全被 use 闸门挡下）"
    Write-Host "    -> 看上面的表：`-Env <短名>` 点名可越过闸门；表里没登记的新环境先在 `$EnvTable` 补一行"
    exit 1
}

$fail = 0; $round = 0
while ($true) {
    $round++
    $ready = @(); $pending = @()
    foreach ($e in $pick) {
        if (Test-LocalPort $e.Port) { Write-Host "[=] $($e.Short) : 端口 $($e.Port) 已在监听，跳过自举"; $ready += $e }
        else { $pending += $e }
    }
    $bootFail = 0
    foreach ($e in $pending) {
        if (-not (Test-Path $e.CmdPath)) {
            Write-Host "[X] $($e.Short) : 缺自举脚本 $($e.CmdPath) —— 该环境从没在 IDE 里打开过，脚本无法凭空生成"
            $bootFail++; continue
        }
        # 单个环境起不来不拦其它环境：记数后继续
        if (Boot $e) { $ready += $e } else { Write-Host "[X] $($e.Short) : 等 ${WaitSec}s 后端口 $($e.Port) 仍无监听"; $bootFail++ }
    }
    if (-not $ready) { Write-Host "[X] 本轮没有任一环境可用"; Diagnose; exit 1 }
    if ($bootFail) { Write-Host "[!] $($bootFail) 个环境隧道起不来，下面只验证已就绪的"; Diagnose }
    $fail = $bootFail
    foreach ($e in $ready) {
        if (Verify $e) { Write-Host "[OK] $($e.Short) : 可用 -> ssh -F `"$Config`" $($e.Alias)" }
        else { Write-Host "[X] $($e.Short) : 端口在监听但 ssh 不通（容器内 sshd 被仿真压满，或环境已停）"; $fail++ }
    }
    if (-not $Watch) { break }
    if ($round -ge $Iterations) { Write-Host "[i] 保温到达 $Iterations 轮，退出"; break }
    Start-Sleep -Seconds $IntervalSec
}
if ($Diag) { Diagnose }
exit $(if ($fail) { 1 } else { 0 })
