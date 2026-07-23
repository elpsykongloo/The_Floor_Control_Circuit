[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("all", "pytest", "ruff", "mypy")]
    [string]$Task = "all",

    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$ToolArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$dataRoot = if ([string]::IsNullOrWhiteSpace($env:FCC_DATA_ROOT)) {
    "D:\data_storage\The_Floor_Control_Circuit"
}
else {
    [System.IO.Path]::GetFullPath($env:FCC_DATA_ROOT)
}
$repoPrefix = $repoRoot.TrimEnd("\") + "\"
if (
    $dataRoot.Equals($repoRoot, [System.StringComparison]::OrdinalIgnoreCase) -or
    $dataRoot.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)
) {
    throw "FCC_DATA_ROOT 不得指向仓库内部：$dataRoot"
}

$pytestTemp = Join-Path $dataRoot "tmp\pytest"
$toolCacheRoot = Join-Path $dataRoot "cache\tooling"
$pytestCache = Join-Path $toolCacheRoot "pytest"
$ruffCache = Join-Path $toolCacheRoot "ruff"
$mypyCache = Join-Path $toolCacheRoot "mypy"
$pytestLock = Join-Path $toolCacheRoot "pytest.lock"

New-Item -ItemType Directory -Force -Path $pytestTemp, $pytestCache, $ruffCache, $mypyCache | Out-Null

# 测试与静态检查不应在源码树中生成 __pycache__。
$env:PYTHONDONTWRITEBYTECODE = "1"

function Invoke-RuffCheck {
    param([string[]]$Arguments)

    if ($Arguments | Where-Object { $_ -like "--cache-dir*" }) {
        throw "统一入口禁止覆盖 ruff 缓存目录；固定位置为 $ruffCache"
    }
    $effectiveArgs = if ($Arguments.Count -eq 0) { @(".") } else { $Arguments }
    $commandArgs = @("run", "ruff", "check", "--cache-dir", $ruffCache) + $effectiveArgs
    & uv @commandArgs
    $script:ToolExitCode = $LASTEXITCODE
}

function Invoke-MypyCheck {
    param([string[]]$Arguments)

    if ($Arguments | Where-Object { $_ -like "--cache-dir*" }) {
        throw "统一入口禁止覆盖 mypy 缓存目录；固定位置为 $mypyCache"
    }
    $effectiveArgs = if ($Arguments.Count -eq 0) { @("src", "tests") } else { $Arguments }
    $commandArgs = @("run", "mypy", "--cache-dir", $mypyCache) + $effectiveArgs
    & uv @commandArgs
    $script:ToolExitCode = $LASTEXITCODE
}

function Invoke-PytestCheck {
    param([string[]]$Arguments)

    if ($Arguments | Where-Object { $_ -like "--basetemp*" -or $_ -like "*cache_dir*" }) {
        throw "统一入口禁止覆盖 pytest 临时目录或缓存目录；固定位置为 $pytestTemp 和 $pytestCache"
    }

    $lockStream = $null
    try {
        try {
            $lockStream = [System.IO.File]::Open(
                $pytestLock,
                [System.IO.FileMode]::OpenOrCreate,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
        }
        catch [System.IO.IOException] {
            throw "已有 pytest 正在使用固定临时目录；请等待其结束后重试。锁文件：$pytestLock"
        }

        $commandArgs = @(
            "run",
            "pytest",
            "--basetemp=$pytestTemp",
            "-o",
            "cache_dir=$pytestCache"
        ) + $Arguments
        & uv @commandArgs
        $script:ToolExitCode = $LASTEXITCODE
    }
    finally {
        if ($null -ne $lockStream) {
            $lockStream.Dispose()
        }
    }
}

$script:ToolExitCode = 0
Push-Location $repoRoot
try {
    switch ($Task) {
        "ruff" {
            Invoke-RuffCheck -Arguments $ToolArgs
        }
        "mypy" {
            Invoke-MypyCheck -Arguments $ToolArgs
        }
        "pytest" {
            Invoke-PytestCheck -Arguments $ToolArgs
        }
        "all" {
            if ($ToolArgs.Count -gt 0) {
                throw "all 模式不接受额外参数；需要定向检查时请选择 pytest、ruff 或 mypy。"
            }
            Invoke-RuffCheck -Arguments @(".")
            if ($script:ToolExitCode -eq 0) {
                Invoke-PytestCheck -Arguments @("-q")
            }
        }
    }
}
finally {
    Pop-Location
}

exit $script:ToolExitCode
