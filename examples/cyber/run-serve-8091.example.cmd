@echo off
title Strata Cyber (8091)
cd /d "E:\strata-v0139"
set "PATH=E:\cuda-13.3\bin\x64;%PATH%"
set "STRATA_HC_SPLIT=0"
"E:\strata\.venv\Scripts\python.exe" "E:\strata-v0139\serve\server.py" --engine strata --config "E:\strata-models\Cyber\strata-cyber.json" --port 8091 > "E:\strata-models\Cyber\logs\serve-stdout.log" 2>&1