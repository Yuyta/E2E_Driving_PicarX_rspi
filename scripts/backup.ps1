<#
.SYNOPSIS
    走行データ、モデル、記憶 DB を PC 内の別の物理ディスクへ複製する（ROADMAP の M3）。

.DESCRIPTION
    robocopy で追加と更新だけを複製し、複製先のファイルは消さない（走行データは追記のみ）。
    記憶 DB を書き込み中に複製しないよう、GUI を閉じてから実行する。
    -RestoreTo を付けると、複製先から指定したフォルダへ戻す（復元の確認用）。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\backup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\backup.ps1 -RestoreTo D:\picarx-restore-test
#>
param(
    [string]$Destination = $(if ($env:PICARX_BACKUP_DIR) { $env:PICARX_BACKUP_DIR } else { 'D:\PiCarX-backup' }),
    [string]$RestoreTo
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

# 複製する対象（リポジトリからの相対パス）。いずれも git の管理外か、個人データを含む
$targets = @('python\data', 'python\models', 'data')

function Invoke-Copy([string]$from, [string]$to) {
    if (-not (Test-Path $from)) {
        Write-Host "SKIP  $from (not found)"
        return
    }
    # /E: 空のフォルダも含める。/XO: 複製先の方が新しいファイルは上書きしない
    robocopy $from $to /E /XO /COPY:DAT /DCOPY:T /R:2 /W:2 /NP /NFL /NDL /XJ | Out-Null
    # robocopy は 8 以上が失敗
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed ($LASTEXITCODE): $from -> $to"
    }
    Write-Host "OK    $from -> $to"
}

if ($RestoreTo) {
    foreach ($t in $targets) {
        Invoke-Copy (Join-Path $Destination $t) (Join-Path $RestoreTo $t)
    }
    Write-Host "Restored to $RestoreTo"
    exit 0
}

$destRoot = [System.IO.Path]::GetPathRoot((Resolve-Path -LiteralPath (Split-Path -Parent $Destination) -ErrorAction SilentlyContinue).Path)
$repoRoot = [System.IO.Path]::GetPathRoot($repo)
if ($destRoot -and ($destRoot -eq $repoRoot)) {
    Write-Warning "The backup is on the same drive as the repository ($repoRoot). It does not protect against a disk failure."
}

foreach ($t in $targets) {
    Invoke-Copy (Join-Path $repo $t) (Join-Path $Destination $t)
}
Write-Host "Backup finished: $Destination"
