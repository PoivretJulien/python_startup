<#
.SYNOPSIS
    Windows bootstrap of the Python GLFW + OpenGL (core profile) toolchain.

.DESCRIPTION
    Creates .venv, installs the tutorial stack from requirements.txt and verifies
    a live OpenGL 3.3 core context. Never needs Administrator: opengl32.dll and
    every native dependency ship with Windows or inside the wheels.

    Windows equivalents of the Linux script's checks:
      ldconfig -p libGL.so.1   ->  C:\Windows\System32\opengl32.dll
      ldconfig -p libX11.so.6  ->  (n/a: GLFW uses win32/dwmapi, part of Windows)
      uv / python3 -m venv     ->  uv / py -3 -m venv, venv at .venv\Scripts

.PARAMETER Legacy
    Also install pyrr, so the original 2016 scripts run.

.PARAMETER Python
    Explicit interpreter path, e.g. -Python C:\Python312\python.exe

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1

.EXAMPLE
    .\setup-windows.ps1 -Legacy
#>
[CmdletBinding()]
param(
    [switch]$Legacy,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Venv    = ".venv"
$VenvPy  = Join-Path $Venv "Scripts\python.exe"

# ------------------------------------------------------------------ output style
function Write-Say  ($m) { Write-Host ([char]0x25B8 + " " + $m) -ForegroundColor DarkYellow }
function Write-Ok   ($m) { Write-Host ("  " + [char]0x2713 + " " + $m) -ForegroundColor Green }
function Write-Warn ($m) { Write-Host ("  " + [char]0x21 + " " + $m) -ForegroundColor Yellow }
function Write-Fail ($m) { Write-Host ("  " + [char]0x2717 + " " + $m) -ForegroundColor Red }
function Die       ($m) { Write-Fail $m; exit 1 }

if ($PSVersionTable.PSVersion.Major -lt 5) { Die "PowerShell >= 5.1 required" }

# ---------------------------------------------------------------- 1. interpreter
Write-Say "checking interpreter"

$candidates = @()
if ($Python) { $candidates += ,@($Python) }
$candidates += ,@("py", "-3")   # Python Launcher: honours the newest 3.x install
$candidates += ,@("python")
$candidates += ,@("python3")

$PyExe  = $null
$PyPre  = @()
foreach ($cand in $candidates) {
    $exe = $cand[0]
    $cmd = Get-Command $exe -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }
    $pre = @()
    if ($cand.Count -gt 1) { $pre = $cand[1..($cand.Count - 1)] }
    & $exe @pre -c "import sys" 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { continue }
    $PyExe = $cmd.Source
    $PyPre = $pre
    break
}

if (-not $PyExe) {
    Die @"
no Python interpreter found.
    install one (no admin needed for the per-user install):
        winget install --scope user Python.Python.3.12
        or  py -3  after installing from https://www.python.org/downloads/windows/
    then re-run:  .\setup-windows.ps1 -Python C:\Path\To\python.exe
"@
}

$ver = & $PyExe @PyPre -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
$okVer = & $PyExe @PyPre -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    Die "Python $ver is too old (need >= 3.11). Install 3.12: winget install --scope user Python.Python.3.12"
}
Write-Ok "Python $ver"
Write-Host ("    exe    : " + $PyExe + $(if ($PyPre) { " " + ($PyPre -join " ") } else { "" })) -ForegroundColor DarkGray

$arch = & $PyExe @PyPre -c "import platform; print(platform.machine())"
Write-Host ("    arch   : " + $arch + " / " + [System.Environment]::OSVersion.VersionString) -ForegroundColor DarkGray
if ($arch -match "ARM64") {
    Write-Ok "Windows on ARM: arm64 wheels exist for numpy/Pillow; glfw ships a win-arm64 build"
}

# ------------------------------------------------------ 2. native libs (verify)
Write-Say "checking native OpenGL layer"
$sys32 = Join-Path $env:SystemRoot "System32"
foreach ($dll in @("opengl32.dll", "dwmapi.dll", "gdi32.dll")) {
    if (Test-Path (Join-Path $sys32 $dll)) {
        Write-Ok "$dll (system)"
    } elseif ($dll -eq "opengl32.dll") {
        Die "$dll missing from $sys32 - the OpenGL loader is broken (run: sfc /scannow)"
    } else {
        Write-Warn "$dll missing (unexpected)"
    }
}
Write-Host ("    note   : GLFW's win32 backend needs no extra runtime; wheels bundle glfw3.dll") -ForegroundColor DarkGray

if ($env:SESSIONNAME -and $env:SESSIONNAME.ToUpper().StartsWith("RDP")) {
    Write-Warn "Remote Desktop session: GPU contexts are usually limited to OpenGL 1.1"
    Write-Warn "run this locally (or install Mesa3D's llvmpipe opengl32.dll) for the probe in step 4"
}

# ------------------------------------------------------------------ 3. venv
function New-Venv {
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Write-Say "creating virtualenv with uv"
        & uv venv --python $PyExe $Venv | Out-Null
    } else {
        Write-Say "creating virtualenv with stdlib venv (uv not found)"
        & $PyExe @PyPre -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { Die "virtualenv creation failed" }
}

if (Test-Path $VenvPy) {
    & $VenvPy -c "import sys" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Say "reusing existing virtualenv ($Venv)"
    } else {
        Write-Warn "$Venv is broken, recreating it"
        Remove-Item -Recurse -Force $Venv
        New-Venv
    }
} else {
    New-Venv
}
Write-Ok "venv -> $VenvPy"

# --------------------------------------------------------------- 4. packages
$reqs = @("requirements.txt")
if ($Legacy) { $reqs += "requirements-legacy.txt" }

if (Get-Command uv -ErrorAction SilentlyContinue) {
    foreach ($r in $reqs) {
        Write-Say "installing $r"
        if ($r -eq "requirements-legacy.txt") { Write-Say "installing pyrr for the legacy scripts" }
        & uv pip install --quiet --python $VenvPy -r $r
        if ($LASTEXITCODE -ne 0) { Die "uv pip install -r $r failed" }
    }
} else {
    Write-Say "installing requirements.txt"
    & $VenvPy -m pip install --quiet --upgrade pip
    foreach ($r in $reqs) {
        if ($r -eq "requirements-legacy.txt") { Write-Say "installing pyrr for the legacy scripts" }
        & $VenvPy -m pip install --quiet -r $r
        if ($LASTEXITCODE -ne 0) { Die "pip install -r $r failed" }
    }
}
Write-Ok "packages installed"

# ------------------------------------------------------------------ 5. verify
Write-Say "probing the live OpenGL context"
& $VenvPy "check_env.py"
if ($LASTEXITCODE -ne 0) { Die "check_env.py reported problems (see above)" }

Write-Host ""
Write-Host "Ready. Activate and run things with:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1        # PowerShell"
Write-Host "  .\.venv\Scripts\activate.bat        # cmd.exe"
Write-Host "  .\.venv\Scripts\python.exe check_env.py --window"
Write-Host ""
Write-Host "If activation is blocked by the execution policy (default on Windows):" -ForegroundColor DarkGray
Write-Host "  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass" -ForegroundColor DarkGray
Write-Host "  # or skip activation entirely and call .\.venv\Scripts\python.exe directly" -ForegroundColor DarkGray