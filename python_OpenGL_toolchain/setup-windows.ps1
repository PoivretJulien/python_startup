<#
.SYNOPSIS
    Windows bootstrap of the Python GLFW + OpenGL (core profile) toolchain.

.DESCRIPTION
    Creates .venv, installs the tutorial stack from requirements.txt and verifies
    a live OpenGL 3.3 core context. No Administrator rights needed: opengl32.dll
    ships with Windows and the glfw wheel bundles glfw3.dll.

    Windows equivalents of the Linux script's checks:
      ldconfig -p libGL.so.1   ->  C:\Windows\System32\opengl32.dll
      ldconfig -p libglfw.so.3 ->  glfw3.dll on PATH, or the wheel's glfw\glfw3.dll
      uv / python3 -m venv     ->  uv / py -3 -m venv, venv at .venv\Scripts

    Warning worth knowing: on win32 PyGLFW tries ctypes.CDLL("glfw3.dll") against
    the default DLL search path FIRST, so any glfw3.dll on PATH silently wins
    over the one inside the wheel. Use -SystemGlfw to make that explicit, or
    -BundledGlfw to pin the wheel's copy.

.PARAMETER Legacy
    Also install pyrr, so the original 2016 scripts run.

.PARAMETER SystemGlfw
    Require a real glfw3.dll. If none is found, the official glfw 3.5.1 prebuilt
    binary (lib-static-ucrt, no VC++ runtime needed) is downloaded into
    .\vendor\glfw and used via PYGLFW_LIBRARY.

.PARAMETER BundledGlfw
    Pin PYGLFW_LIBRARY to the glfw3.dll that ships inside the glfw wheel.

.PARAMETER Python
    Explicit interpreter path, e.g. -Python C:\Python312\python.exe

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\setup-windows.ps1

.EXAMPLE
    .\setup-windows.ps1 -Legacy -SystemGlfw
#>
[CmdletBinding()]
param(
    [switch]$Legacy,
    [switch]$SystemGlfw,
    [switch]$BundledGlfw,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Venv      = ".venv"
$VenvPy    = Join-Path $Venv "Scripts\python.exe"
$VendorDir = Join-Path "vendor" "glfw"
$GlfwVer   = "3.5.1"
$GlfwZip   = "https://github.com/glfw/glfw/releases/download/$GlfwVer/glfw-$GlfwVer.bin.WIN64.zip"

if ($SystemGlfw -and $BundledGlfw) { throw "-SystemGlfw and -BundledGlfw are mutually exclusive" }

# ------------------------------------------------------------------ output style
function Write-Say  ($m) { Write-Host ([char]0x25B8 + " " + $m) -ForegroundColor DarkYellow }
function Write-Ok   ($m) { Write-Host ("  " + [char]0x2713 + " " + $m) -ForegroundColor Green }
function Write-Warn ($m) { Write-Host ("  " + [char]0x21 + " " + $m) -ForegroundColor Yellow }
function Write-Fail ($m) { Write-Host ("  " + [char]0x2717 + " " + $m) -ForegroundColor Red }
function Die        ($m) { Write-Fail $m; exit 1 }

if ($PSVersionTable.PSVersion.Major -lt 5) { Die "PowerShell >= 5.1 required" }

# Finds a glfw3.dll the way Windows' loader would (app dir, then PATH).
function Find-SystemGlfw {
    $here = (Get-Location).Path
    if (Test-Path (Join-Path $here "glfw3.dll")) { return (Join-Path $here "glfw3.dll") }
    foreach ($dir in ($env:PATH -split ';')) {
        if (-not $dir) { continue }
        $cand = Join-Path $dir "glfw3.dll"
        if (Test-Path $cand) { return $cand }
    }
    return $null
}

# ---------------------------------------------------------------- 1. interpreter
Write-Say "checking interpreter"

$candidates = @()
if ($Python) { $candidates += ,@($Python) }
$candidates += ,@("py", "-3")   # Python Launcher: honours the newest 3.x install
$candidates += ,@("python")
$candidates += ,@("python3")

$PyExe = $null
$PyPre = @()
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
    install one (per-user install needs no admin):
        winget install --scope user Python.Python.3.12
        or  py -3  after installing from https://www.python.org/downloads/windows/
    then re-run:  .\setup-windows.ps1 -Python C:\Path\To\python.exe
"@
}

$ver = & $PyExe @PyPre -c "import sys; print('%d.%d.%d' % sys.version_info[:3])"
& $PyExe @PyPre -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    Die "Python $ver is too old (need >= 3.11). Install 3.12: winget install --scope user Python.Python.3.12"
}
Write-Ok "Python $ver"
Write-Host ("    exe    : " + $PyExe + $(if ($PyPre) { " " + ($PyPre -join " ") } else { "" })) -ForegroundColor DarkGray

$arch = & $PyExe @PyPre -c "import platform; print(platform.machine())"
Write-Host ("    arch   : " + $arch + " / " + [System.Environment]::OSVersion.VersionString) -ForegroundColor DarkGray
if ($arch -match "ARM64") {
    Write-Ok "Windows on ARM: arm64 wheels for numpy/Pillow; the prebuilt glfw3.dll is x64 (runs emulated)"
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
Write-Host ("    note   : GLFW's win32 backend needs no extra runtime") -ForegroundColor DarkGray

if ($env:SESSIONNAME -and $env:SESSIONNAME.ToUpper().StartsWith("RDP")) {
    Write-Warn "Remote Desktop session: GPU contexts are usually limited to OpenGL 1.1"
    Write-Warn "run this locally for the probe in step 5"
}

# ------------------------------------------------ 2b. native GLFW (glfw3.dll)
# The glfw wheel bundles glfw3.dll (+ msvcr120.dll), so a system GLFW is OPTIONAL.
# But PyGLFW checks the default DLL search path first on Windows, so a PATH copy
# always wins. -SystemGlfw makes that explicit; -BundledGlfw pins the wheel copy.
Write-Say "checking native GLFW library (optional: the wheel bundles 3.4.0)"
$SysGlfw = Find-SystemGlfw
$PinGlfw = $null

if ($SysGlfw) {
    Write-Ok "system glfw3.dll  [$SysGlfw]  (takes precedence over the wheel on Windows)"
} else {
    Write-Warn "no glfw3.dll on PATH  (the wheel's bundled copy is used - that is fine)"
}

if ($BundledGlfw) {
    # resolved further down, once .venv exists
}
elseif ($SystemGlfw) {
    if (-not $SysGlfw) {
        Write-Host "  ! -SystemGlfw requested but no glfw3.dll found" -ForegroundColor Yellow
        Write-Host "    downloading glfw $GlfwVer prebuilt (lib-static-ucrt, no VC++ runtime needed)"
        Write-Host "    $GlfwZip" -ForegroundColor DarkGray
        New-Item -ItemType Directory -Force -Path $VendorDir | Out-Null
        $zip = Join-Path ([System.IO.Path]::GetTempPath()) "glfw-$GlfwVer.bin.WIN64.zip"
        try {
            Invoke-WebRequest -Uri $GlfwZip -OutFile $zip -UseBasicParsing
            $extract = Join-Path ([System.IO.Path]::GetTempPath()) "glfw-$GlfwVer-extract"
            Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
            Expand-Archive -Path $zip -DestinationPath $extract -Force
            $dll = $null
            foreach ($variant in @("lib-static-ucrt", "lib-vc2022", "lib-vc2019", "lib-mingw-w64")) {
                $cand = Join-Path $extract "glfw-$GlfwVer.bin.WIN64\$variant\glfw3.dll"
                if (Test-Path $cand) { $dll = $cand; break }
            }
            if (-not $dll) { throw "glfw3.dll not found in the archive" }
            Copy-Item $dll (Join-Path $VendorDir "glfw3.dll") -Force
            Copy-Item (Join-Path (Split-Path $dll -Parent) "glfw3dll.lib") $VendorDir -Force -ErrorAction SilentlyContinue
            Write-Ok "extracted: $(Join-Path $VendorDir 'glfw3.dll')"
        } catch {
            Die "download/extract failed: $($_.Exception.Message)
       install GLFW manually (vcpkg install glfw3:x64-windows) or drop -SystemGlfw"
        } finally {
            Remove-Item -Force $zip -ErrorAction SilentlyContinue
            Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
        }
        $SysGlfw = (Resolve-Path (Join-Path $VendorDir "glfw3.dll")).Path
    }
    $PinGlfw = $SysGlfw
    Write-Ok "using system GLFW: $PinGlfw"
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
    Write-Say "installing requirements.txt (includes the glfw wheel)"
    & uv pip install --quiet --python $VenvPy -r "requirements.txt"
    if ($LASTEXITCODE -ne 0) { Die "uv pip install -r requirements.txt failed" }
    if ($Legacy) {
        Write-Say "installing pyrr for the legacy scripts"
        & uv pip install --quiet --python $VenvPy -r "requirements-legacy.txt"
        if ($LASTEXITCODE -ne 0) { Die "uv pip install -r requirements-legacy.txt failed" }
    }
} else {
    Write-Say "installing requirements.txt (includes the glfw wheel)"
    & $VenvPy -m pip install --quiet --upgrade pip
    foreach ($r in $reqs) {
        if ($r -eq "requirements-legacy.txt") { Write-Say "installing pyrr for the legacy scripts" }
        & $VenvPy -m pip install --quiet -r $r
        if ($LASTEXITCODE -ne 0) { Die "pip install -r $r failed" }
    }
}
Write-Ok "packages installed"

# ------------------------------------------ 4b. pin the GLFW the venv will use
if ($BundledGlfw) {
    $SitePkg = & $VenvPy -c "import os, glfw; print(os.path.dirname(glfw.__file__))"
    if ($LASTEXITCODE -ne 0 -or -not $SitePkg) { Die "cannot locate the glfw package inside .venv" }
    $cand = Join-Path $SitePkg.Trim() "glfw3.dll"
    if (-not (Test-Path $cand)) {
        Die "the glfw wheel contains no bundled glfw3.dll -- reinstall it: uv pip install --force-reinstall glfw"
    }
    $PinGlfw = $cand
    Write-Ok "using the wheel's bundled GLFW: $PinGlfw"
}

if ($PinGlfw) { $env:PYGLFW_LIBRARY = $PinGlfw }
elseif ($env:PYGLFW_LIBRARY) {
    Write-Ok "inheriting PYGLFW_LIBRARY from the environment: $env:PYGLFW_LIBRARY"
}

Write-Say "verifying the GLFW native library loaded by the venv"
$probe = @'
import os, sys
try:
    from glfw import library
    import glfw
except ImportError as exc:
    print(f"    GLFW native library: NOT LOADED ({exc})")
    raise SystemExit(1)
lib = getattr(library.glfw, "_name", None)
if lib is None:
    print("    GLFW native library: NOT LOADED")
    raise SystemExit(1)
try:
    major, minor, rev = glfw.get_version()
except Exception:
    major = minor = rev = 0
where = "bundled (wheel)" if lib.startswith(sys.prefix) else "system"
print(f"    GLFW {major}.{minor}.{rev}  [{where}]  {lib}")
pin = os.environ.get("PYGLFW_LIBRARY")
if pin:
    print(f"    pinned by PYGLFW_LIBRARY={pin}")
elif where == "system":
    print("    note: a glfw3.dll from PATH shadows the wheel copy; use -BundledGlfw to pin it")
'@
$probe | & $VenvPy -
if ($LASTEXITCODE -ne 0) {
    Die "GLFW native library not loaded -- fix: uv pip install --force-reinstall glfw (or .\setup-windows.ps1 -SystemGlfw)"
}
Write-Ok "GLFW ok"

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

if ($PinGlfw) {
    Write-Host ""
    Write-Host "This run used PYGLFW_LIBRARY=$PinGlfw" -ForegroundColor DarkGray
    Write-Host "To keep it for new shells:" -ForegroundColor DarkGray
    Write-Host "  setx PYGLFW_LIBRARY `"$PinGlfw`"" -ForegroundColor DarkGray
}