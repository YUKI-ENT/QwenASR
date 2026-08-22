[CmdletBinding()]
param(
    [switch]$InstallDependencies,
    [ValidateSet("all", "cli", "server")]
    [string]$Target = "all"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($env:OS -ne "Windows_NT") {
    throw "Windows executables must be built on Windows."
}

$Python = Get-Command python -ErrorAction Stop
function Invoke-Python {
    & $Python.Source @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: python $args"
    }
}

$PythonInfo = & $Python.Source -c "import platform,sys; print(f'{sys.version_info.major}.{sys.version_info.minor}|{platform.architecture()[0]}')"
if ($LASTEXITCODE -ne 0) {
    throw "Unable to run Python."
}
if ($PythonInfo -ne "3.12|64bit") {
    throw "64-bit Python 3.12 is required (detected: $PythonInfo)."
}

if ($InstallDependencies) {
    Invoke-Python -m pip install --upgrade pip
    Invoke-Python -m pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    Invoke-Python -m pip install -r requirements.txt
}
Invoke-Python -m pip install -r packaging/requirements-build.txt

function Build-QwenASRExecutable {
    param(
        [string]$Name,
        [string]$EntryPoint,
        [string[]]$ExtraArguments
    )

    $Arguments = @(
        "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir", "--console",
        "--name", $Name,
        "--paths", $ProjectRoot,
        "--additional-hooks-dir", (Join-Path $ProjectRoot "packaging/hooks"),
        "--hidden-import", "qwen_asr",
        "--exclude-module", "vllm",
        "--exclude-module", "flash_attn"
    ) + $ExtraArguments + @($EntryPoint)

    Invoke-Python @Arguments

    $OutputDir = Join-Path $ProjectRoot "dist/$Name"
    $NagisaTagger = Join-Path $OutputDir "_internal/nagisa/tagger.py"
    if (Test-Path -LiteralPath $NagisaTagger) {
        $NagisaSource = [IO.File]::ReadAllText($NagisaTagger)
        $NagisaSource = $NagisaSource.Replace(
            "w.replace('(', '\(').replace(')', '\)')",
            "w.replace('(', r'\(').replace(')', r'\)')"
        )
        [IO.File]::WriteAllText(
            $NagisaTagger,
            $NagisaSource,
            [Text.UTF8Encoding]::new($false)
        )
    }
    Copy-Item "config.json.sample" (Join-Path $OutputDir "config.json") -Force
    New-Item -ItemType Directory -Force (Join-Path $OutputDir "models") | Out-Null
    New-Item -ItemType Directory -Force (Join-Path $OutputDir "results") | Out-Null

    $BuiltExecutable = Join-Path $OutputDir "$Name.exe"
    & $BuiltExecutable --version
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start $Name."
    }
}

if ($Target -in @("all", "cli")) {
    Build-QwenASRExecutable -Name "QwenASR" -EntryPoint "app.py" -ExtraArguments @()
}
if ($Target -in @("all", "server")) {
    Build-QwenASRExecutable -Name "QwenASR-Server" -EntryPoint "server.py" -ExtraArguments @(
        "--collect-all", "uvicorn",
        "--hidden-import", "multipart"
    )
}

Write-Host "Build complete: $ProjectRoot\dist"
