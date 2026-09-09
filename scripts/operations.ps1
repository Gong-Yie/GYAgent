[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [ValidateSet("install", "test", "evaluate", "serve", "backup", "restore", "migrate")]
    [string]$Command,
    [string]$DataDir = "data",
    [string]$Archive,
    [string]$Target,
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $projectRoot
try {
    switch ($Command) {
        "install" {
            uv sync --locked --extra dev
        }
        "test" {
            uv run python -m pytest -q
        }
        "evaluate" {
            uv run python scripts/evaluate_stage29.py
        }
        "serve" {
            $env:SC_OPERATIONS_DATA_DIR = $DataDir
            $env:SC_OPERATIONS_PORT = [string]$Port
            uv run python -c "import os; from pathlib import Path; from self_cognition.bootstrap import build_container; from self_cognition.interfaces.http import create_server; app = build_container(Path(os.environ['SC_OPERATIONS_DATA_DIR'])); server = create_server(app, port=int(os.environ['SC_OPERATIONS_PORT']), static_directory=Path('webui')); print(f'http://127.0.0.1:{server.server_port}'); server.serve_forever()"
        }
        "backup" {
            if (-not $Archive) { throw "-Archive is required for backup" }
            uv run python scripts/maintain_data.py backup $DataDir $Archive --config .env.example
        }
        "restore" {
            if (-not $Archive -or -not $Target) {
                throw "-Archive and -Target are required for restore"
            }
            uv run python scripts/maintain_data.py restore $Archive $Target
        }
        "migrate" {
            if (-not $Target) { throw "-Target is required for migrate" }
            uv run python scripts/maintain_data.py migrate $DataDir $Target --config .env.example
        }
    }
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
finally {
    Pop-Location
}
