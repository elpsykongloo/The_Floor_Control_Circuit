[CmdletBinding(SupportsShouldProcess = $true)]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$allowedRootNames = @(
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".hypothesis"
)

$rootTargets = Get-ChildItem -LiteralPath $repoRoot -Force -Directory | Where-Object {
    $_.Name -in $allowedRootNames -or $_.Name -like ".pytest_tmp_*"
}

$sourceRoots = @("src", "tests", "scripts", "runners") | ForEach-Object {
    $path = Join-Path $repoRoot $_
    if (Test-Path -LiteralPath $path -PathType Container) {
        Get-Item -LiteralPath $path
    }
}
$bytecodeTargets = $sourceRoots | ForEach-Object {
    Get-ChildItem -LiteralPath $_.FullName -Recurse -Force -Directory -Filter "__pycache__"
}

$targets = @($rootTargets) + @($bytecodeTargets)
foreach ($target in $targets) {
    $fullPath = [System.IO.Path]::GetFullPath($target.FullName)
    $repoPrefix = $repoRoot.TrimEnd("\") + "\"
    if (-not $fullPath.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "拒绝清理仓库范围外路径：$fullPath"
    }
    if ($target.Name -ne "__pycache__" -and $target.Parent.FullName -ne $repoRoot) {
        throw "拒绝清理非根级工具目录：$fullPath"
    }
}

if ($targets.Count -eq 0) {
    Write-Host "仓库内没有需要清理的工具缓存。"
    exit 0
}

$removedCount = 0
foreach ($target in $targets | Sort-Object { $_.FullName.Length } -Descending) {
    if ($PSCmdlet.ShouldProcess($target.FullName, "删除可再生工具缓存")) {
        Remove-Item -LiteralPath $target.FullName -Recurse -Force
        $removedCount += 1
    }
}

if ($removedCount -gt 0) {
    Write-Host "已清理 $removedCount 个仓库内工具缓存目录。"
}
else {
    Write-Host "未执行实际清理。"
}
