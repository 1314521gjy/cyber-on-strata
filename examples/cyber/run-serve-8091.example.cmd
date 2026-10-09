@echo off
rem ============================================================================
rem  Example Windows launcher for the Strata serve layer.
rem  Replace every <PLACEHOLDER> before use:
rem    <REPO>      this repository checkout (cd target; holds serve\server.py)
rem    <PYTHON>    the python.exe that has the serve dependencies (requirements.txt)
rem    <RUNCONFIG> the run config json, e.g. examples\cyber\strata-cyber.example.json
rem    <LOGDIR>    a writable directory for the server's own output
rem    <CUDA>      the CUDA bin directory holding the runtime DLLs
rem  The server's own stdout/stderr must go to a file: its errors (for example
rem  "the engine exited before it was ready") are NOT written to the engine log.
rem ============================================================================
title Strata Cyber (8091)
cd /d "<REPO>"
set "PATH=<CUDA>;%PATH%"
set "STRATA_HC_SPLIT=0"
"<PYTHON>" "<REPO>\serve\server.py" --engine strata --config "<RUNCONFIG>" --port 8091 > "<LOGDIR>\serve-stdout.log" 2>&1
