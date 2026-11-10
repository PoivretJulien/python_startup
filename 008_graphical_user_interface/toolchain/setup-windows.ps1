# Windows bootstrap of the whole project: GLFW + wgpu (DX12) + OpenCV.
#
#   powershell -ExecutionPolicy Bypass -File .\toolchain\setup-windows.ps1
#   powershell -ExecutionPolicy Bypass -File .\toolchain\setup-windows.ps1 -NoOpenCV
#   powershell -ExecutionPolicy Bypass -File .\toolchain\setup-windows.ps1 -Python "C:\Python312\python.exe"
#
# wgpu picks DX12 automatically on Windows (then Vulkan, then GL). DX12 ships
# with Windows 10/11, so nothing system-wide is installed: every Python
# package (glfw, wgpu, numpy, Pillow, OpenCV) is a prebuilt wheel that carries
# its own native DLL. No administrator rights are needed.
#
# It can be started from any folder: it always works in the project root (the
# parent of toolchain\), which is where .venv is created.

[CmdletBinding()]
param(
    [string]$Python = "",
    [switch]$NoOpenCV
)

$ErrorActionPreference = "Stop"
Set-Location -Path (Split-Path -Parent $PSScriptRoot)

$Venv = ".venv"
$Req = "toolchain\requirements.txt"
$ReqCv = "toolchain\requirements-opencv.txt"
$VenvPython = Join-Path $Venv "Scripts\python.exe"

function Say($text)  { Write-Host "> $text" -ForegroundColor DarkYellow }
function Ok($text)   { Write-Host "  [ok] $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  [!]  $text" -ForegroundColor Yellow }
function Die($text)  { Write-Host "  [x]  $text" -ForegroundColor Red; exit 1 }

# ------------------------------------------------------------ 1. interpreter
Say "checking Windows + interpreter"

$os = [System.Environment]::OSVersion.Version
if ($os.Major -lt 10) {
    Die "Windows 10 or newer is required for DX12 (found $os)"
}
Ok "Windows $os ($env:PROCESSOR_ARCHITECTURE)"

# Pick an interpreter: -Python, then the py launcher (newest 3.x), then python.
$PyCmd = @()
if ($Python) {
    $PyCmd = @($Python)
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
    $PyCmd = @("py", "-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PyCmd = @("python")
} else {
    Die "no Python found. Install Python 3.11+ from https://www.python.org/downloads/
       (tick 'Add python.exe to PATH') or: winget install Python.Python.3.12"
}

# A simple (non-advanced) function on purpose: "-c" / "-m" land in $args
# untouched instead of being parsed as PowerShell parameters.
function Invoke-Py {
    $exe = $PyCmd[0]
    $pre = @()
    if ($PyCmd.Count -gt 1) { $pre = $PyCmd[1..($PyCmd.Count - 1)] }
    & $exe @pre @args
}

# The Microsoft Store "python.exe" stub opens the Store instead of running.
$version = Invoke-Py -c "import sys; print('%d.%d.%d' % sys.version_info[:3])" 2>$null
if (-not $version) {
    Die "'$($PyCmd -join ' ')' did not run. If it is the Microsoft Store stub, install Python
       from python.org or winget, or pass -Python C:\path\to\python.exe"
}
Invoke-Py -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    Die "Python >= 3.11 required, found $version. Install a newer one: winget install Python.Python.3.12"
}
$pyBits = Invoke-Py -c "import struct; print(struct.calcsize('P') * 8)"
if ($pyBits -ne "64") {
    Die "a 64-bit Python is required (wgpu ships 64-bit DLLs only), found $pyBits-bit"
}
Ok "Python $version (64-bit) via '$($PyCmd -join ' ')'"

# ------------------------------------------------------- 2. native GPU layer
Say "checking the native GPU layer"
$sys32 = Join-Path $env:WINDIR "System32"
if (Test-Path (Join-Path $sys32 "d3d12.dll")) {
    Ok "Direct3D 12 runtime (d3d12.dll)"
} else {
    Warn "d3d12.dll not found: wgpu will try Vulkan, then GL"
}
if (Test-Path (Join-Path $sys32 "vulkan-1.dll")) {
    Ok "Vulkan loader (vulkan-1.dll): --backend vulkan is available"
} else {
    Warn "no Vulkan loader: fine, DX12 is the default (GPU drivers install it if you want Vulkan)"
}
# wgpu-native and the glfw DLL link against the MSVC runtime.
if (Test-Path (Join-Path $sys32 "vcruntime140.dll")) {
    Ok "Visual C++ runtime (vcruntime140.dll)"
} else {
    Warn "Visual C++ runtime missing: install it if wgpu fails to load:
       winget install Microsoft.VCRedist.2015+.x64"
}
try {
    $gpus = Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }
    foreach ($gpu in $gpus) { Ok "GPU: $gpu" }
} catch {
    Warn "could not list GPUs (Get-CimInstance unavailable)"
}

# ---------------------------------------------------------------- 3. venv
$venvWorks = $false
if (Test-Path $VenvPython) {
    & $VenvPython -c "import sys" 2>$null
    $venvWorks = ($LASTEXITCODE -eq 0)
}
if ($venvWorks) {
    Say "reusing existing virtualenv ($Venv)"
} else {
    if (Test-Path $Venv) {
        Warn "$Venv is broken (or was made on another OS), recreating it"
        Remove-Item -Recurse -Force $Venv
    }
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        Say "creating virtualenv with uv"
        $interp = (Invoke-Py -c "import sys; print(sys.executable)")
        uv venv --python $interp $Venv | Out-Null
    } else {
        Say "creating virtualenv with stdlib venv (uv not found)"
        Invoke-Py -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { Die "could not create $Venv" }
}
Ok "venv -> $VenvPython"

Say "installing $Req (glfw, wgpu, rendercanvas, numpy, Pillow)"
$UseUv = [bool](Get-Command uv -ErrorAction SilentlyContinue)
if ($UseUv) {
    uv pip install --quiet --python $VenvPython -r $Req
} else {
    & $VenvPython -m pip install --quiet --upgrade pip
    & $VenvPython -m pip install --quiet -r $Req
}
if ($LASTEXITCODE -ne 0) { Die "package installation failed (see the pip output above)" }
Ok "packages installed"

# OpenCV is optional, so a missing wheel (very new Python) is a warning only.
if (-not $NoOpenCV) {
    Say "installing $ReqCv (OpenCV, optional: camera + video_link)"
    if ($UseUv) {
        uv pip install --quiet --python $VenvPython -r $ReqCv
    } else {
        & $VenvPython -m pip install --quiet -r $ReqCv
    }
    if ($LASTEXITCODE -eq 0) {
        Ok "OpenCV installed"
    } else {
        Warn "OpenCV could not be installed (no wheel for this Python yet?): only network\video_link needs it"
    }
}

# -------------------------------------------------------------- 4. verify
Say "probing wgpu (DX12 adapter)"
# Lets check_env.py print its ANSI colours in Windows Terminal and conhost.
$env:PYTHONIOENCODING = "utf-8"
& $VenvPython toolchain\check_env.py
if ($LASTEXITCODE -ne 0) {
    Warn "check_env.py reported problems (see above)"
}

Write-Host @"

Ready. From the project root:
  .venv\Scripts\python toolchain\check_env.py --window     # surface + present test
  .venv\Scripts\python examples\01_hover_shapes\main.py    # hover two shapes
  .venv\Scripts\python examples\02_udp_drag\main.py        # two windows over UDP
  .venv\Scripts\python examples\03_udp_video\main.py       # video between two windows
  .venv\Scripts\python examples\04_linked_circles\main.py  # two circles and a dashed line
  .venv\Scripts\python examples\05_keyboard\main.py        # the keyboard, on screen
  .venv\Scripts\python examples\06_slider\main.py          # sliders that set values
  .venv\Scripts\python graphics\main.py                    # the full 3D scene (--backend vulkan forces Vulkan)
  start guide.html                                         # how to use the API

Windows notes:
  * Trackpad pinch / two-finger pan are macOS-only: use the wheel to zoom and
    middle-drag (or Shift+wheel) to pan.
  * The first run of example 2 may show a Windows Firewall prompt for Python:
    allow it on private networks, it is the UDP link.
  * If scripts are blocked, run this file with:
      powershell -ExecutionPolicy Bypass -File .\toolchain\setup-windows.ps1
"@
