[CmdletBinding()]
param(
    [string]$ISCCPath
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ServerSource = Join-Path $ProjectRoot "dist/QwenASR-Server"
$InstallerScript = Join-Path $PSScriptRoot "QwenASR-Server.iss"
$OutputDir = Join-Path $ProjectRoot "release"

if ($env:OS -ne "Windows_NT") {
    throw "The Windows installer must be built on Windows."
}

foreach ($RequiredPath in @(
    (Join-Path $ServerSource "QwenASR-Server.exe"),
    (Join-Path $ServerSource "_internal"),
    (Join-Path $ProjectRoot "config.json.sample"),
    $InstallerScript
)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) {
        throw "Required server distribution file is missing: $RequiredPath"
    }
}

$VersionText = [IO.File]::ReadAllText((Join-Path $ProjectRoot "version.py"))
$VersionMatch = [regex]::Match($VersionText, 'APP_VERSION\s*=\s*"([^"]+)"')
if (-not $VersionMatch.Success) {
    throw "APP_VERSION was not found in version.py."
}
$AppVersion = $VersionMatch.Groups[1].Value
if ($AppVersion -notmatch '^[0-9A-Za-z._-]+$') {
    throw "APP_VERSION contains characters that cannot be used in an artifact name: $AppVersion"
}
$ArtifactBaseName = "QwenASR-Server-$AppVersion-win-x64"

if ($ISCCPath) {
    $Compiler = (Resolve-Path -LiteralPath $ISCCPath -ErrorAction Stop).Path
} else {
    $Command = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    $Candidates = @()
    if ($Command) {
        $Candidates += $Command.Source
    }
    if (${env:ProgramFiles(x86)}) {
        $Candidates += Join-Path ${env:ProgramFiles(x86)} "Inno Setup 6/ISCC.exe"
    }
    if ($env:ProgramFiles) {
        $Candidates += Join-Path $env:ProgramFiles "Inno Setup 6/ISCC.exe"
    }
    $Compiler = $Candidates |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $Compiler) {
        throw "Inno Setup 6 (ISCC.exe) was not found. Install it or specify -ISCCPath."
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputRoot = [IO.Path]::GetFullPath($OutputDir).TrimEnd('\')
$ArtifactPattern = '^' + [regex]::Escape($ArtifactBaseName) +
    '(?:\.exe|-\d+\.bin|-SHA256\.txt)$'
foreach ($ExistingFile in Get-ChildItem -LiteralPath $OutputDir -File) {
    if ($ExistingFile.Name -match $ArtifactPattern) {
        $Parent = [IO.Path]::GetFullPath($ExistingFile.DirectoryName).TrimEnd('\')
        if ($Parent -ne $OutputRoot) {
            throw "Refusing to remove an artifact outside the release directory: $($ExistingFile.FullName)"
        }
        Remove-Item -LiteralPath $ExistingFile.FullName -Force
    }
}

& $Compiler "/DAppVersion=$AppVersion" $InstallerScript
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup compilation failed with exit code $LASTEXITCODE."
}

$PayloadPattern = '^' + [regex]::Escape($ArtifactBaseName) + '(?:\.exe|-\d+\.bin)$'
$PayloadFiles = @(
    Get-ChildItem -LiteralPath $OutputDir -File |
        Where-Object { $_.Name -match $PayloadPattern } |
        Sort-Object Name
)
if ($PayloadFiles.Count -lt 2 -or -not ($PayloadFiles.Name -contains "$ArtifactBaseName.exe")) {
    throw "The expected split installer files were not generated."
}

$GitHubAssetLimit = 2GB
$OversizedFiles = @($PayloadFiles | Where-Object { $_.Length -ge $GitHubAssetLimit })
if ($OversizedFiles.Count -gt 0) {
    $Names = ($OversizedFiles.Name -join ", ")
    throw "Release assets must each be under 2 GiB. Oversized files: $Names"
}

$ChecksumPath = Join-Path $OutputDir "$ArtifactBaseName-SHA256.txt"
$ChecksumLines = foreach ($PayloadFile in $PayloadFiles) {
    $Hash = (Get-FileHash -LiteralPath $PayloadFile.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$Hash  $($PayloadFile.Name)"
}
[IO.File]::WriteAllLines(
    $ChecksumPath,
    $ChecksumLines,
    [Text.UTF8Encoding]::new($false)
)

Write-Host "Installer build complete: $OutputDir"
$ReportFiles = @($PayloadFiles) + @(Get-Item -LiteralPath $ChecksumPath)
$ReportFiles |
    Select-Object Name, Length |
    Format-Table -AutoSize
