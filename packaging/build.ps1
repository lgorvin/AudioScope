# Build the Windows distribution into dist\AudioScope (and a zip next to it).
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File packaging\build.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    python -m venv .venv
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt pyinstaller
}

& $py packaging\make_icon.py
& $py -m PyInstaller packaging\audioscope.spec --noconfirm --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

# Smoke-test the frozen build (in-process and worker-process analysis)
$fixture = Join-Path $root "tests\fixtures\keys_Am_100_4bars.wav"
if (Test-Path $fixture) {
    $out = Join-Path $env:TEMP "audioscope_selftest.json"
    $p = Start-Process -FilePath "dist\AudioScope\AudioScope.exe" -ArgumentList "--selftest", "`"$fixture`"", "`"$out`"" -Wait -PassThru
    if ($p.ExitCode -ne 0) { throw "Self-test failed, see $out" }
    Write-Host "Self-test passed"
}

$zip = "dist\AudioScope-windows-x64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path "dist\AudioScope" -DestinationPath $zip
Write-Host "Built dist\AudioScope\AudioScope.exe and $zip"
