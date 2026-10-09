<#
  Cyber on Strata -- beta launcher (ASCII only; keep it that way).
  Text in, text out: tokenizes your prompt, runs the engine, detokenizes the result.

  Usage (from the package root):
    run.cmd -Prompt "write a python quicksort"
    run.cmd -PromptFile myprompt.txt -MaxNew 512
    run.cmd -Prompt "hi" -Reserve 1200 -Context 8192

  Options:
    -Prompt <text>      prompt text
    -PromptFile <path>  read the prompt from a UTF-8 text file
    -Model <path>       the .gguf; default: the first *.gguf under .\models\
    -Reserve <MiB>      VRAM held back from the expert cache (default 1500; see the README)
    -Context <N>        --max-context (default 4096)
    -MaxNew <N>         tokens to generate (default 256)
    -Prefill <N>        --prefill chunk (default 2048)
    -Workers <N>        --pool-workers (default 0 = auto)
    -Affinity <mode>    all | auto | p-cores (default auto)
    -Kv <mode>          fp16 | int8 | k8v4 | q4_0  (default int8 = k8v8)
    -KvResident <N>     keep N KV cells per layer in VRAM, park the rest in pinned RAM
                        (default 20480; 0 = off). Frees VRAM at long context.
                        Works with int8 / q4_0 / fp16; k8v4 is refused by the engine.
    -Chat               wrap the prompt in the model's chat markers (EXPERIMENTAL)
    -ShowLog            also print the whole engine log
#>
param(
  [string]$Prompt = "",
  [string]$PromptFile = "",
  [string]$Model = "",
  [int]$Reserve = 1500,
  [int]$Context = 4096,
  [int]$MaxNew = 256,
  [int]$Prefill = 2048,
  [int]$Workers = 0,
  [string]$Affinity = "auto",
  [string]$Kv = "int8",
  [int]$KvResident = 20480,
  [switch]$Chat,
  [switch]$ShowLog
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
function Die([string]$msg) { Write-Host ""; Write-Host "ERROR: $msg" -ForegroundColor Red; Write-Host ""; exit 2 }

$exe   = Join-Path $root 'engine\strata.exe'
$tok   = Join-Path $root 'engine\tokenize\llama-tokenize.exe'
$tab   = Join-Path $root 'engine\vocab_bytes.tsv'
$pack  = Join-Path $root 'pack\iq-native'
$profD = Join-Path $root 'profile'

foreach ($p in @($exe, $tok, $tab, $pack)) {
  if (-not (Test-Path -LiteralPath $p)) { Die "package file missing: $p" }
}

# ---- model -------------------------------------------------------------
if (-not $Model) {
  $dir = Join-Path $root 'models'
  $cand = @()
  if (Test-Path -LiteralPath $dir) {
    $cand = @(Get-ChildItem -LiteralPath $dir -Recurse -File -Filter *.gguf | Sort-Object Length -Descending)
  }
  if ($cand.Count -gt 0) { $Model = $cand[0].FullName }
  else { Die "no model found. Put the .gguf into <package>\models\ (or pass -Model <path>)" }
}
if (-not (Test-Path -LiteralPath $Model)) { Die "model not found: $Model" }

$prof = @(Get-ChildItem -LiteralPath $profD -File -Filter *.bin | Select-Object -First 1)
if ($prof.Count -eq 0) { Die "no expert profile (*.bin) in $profD" }
$profPath = $prof[0].FullName

# ---- prompt ------------------------------------------------------------
if ($PromptFile) {
  if (-not (Test-Path -LiteralPath $PromptFile)) { Die "prompt file not found: $PromptFile" }
  $text = [IO.File]::ReadAllText($PromptFile, [Text.Encoding]::UTF8)
} else {
  if (-not $Prompt) { Die 'no prompt. Pass -Prompt "text" or -PromptFile <path>' }
  $text = $Prompt
}
if ($Chat) {
  $text = "<|im_start|>user`n" + $text + "<|im_end|>`n<|im_start|>assistant`n"
}

$work = Join-Path $env:TEMP ("cyber-run-" + $PID)
New-Item -ItemType Directory -Force -Path $work | Out-Null
$pfile  = Join-Path $work 'prompt.txt'
$tfile  = Join-Path $work 'tokens.txt'
$tokOut = Join-Path $work 'tok.out'
$tokErr = Join-Path $work 'tok.err'
$engOut = Join-Path $work 'eng.out'
$engErr = Join-Path $work 'eng.err'
[IO.File]::WriteAllText($pfile, $text, (New-Object Text.UTF8Encoding($false)))

# ---- tokenize ----------------------------------------------------------
Write-Host "[1/3] tokenizing ..." -ForegroundColor DarkGray
$ErrorActionPreference = 'Continue'
& $tok '-m' $Model '-f' $pfile '--ids' 1> $tokOut 2> $tokErr
$ErrorActionPreference = 'Stop'
$raw = @(Get-Content -LiteralPath $tokOut -ErrorAction SilentlyContinue)
$m = [regex]::Match(($raw -join ' '), '\[[^\[\]]*\]')
if (-not $m.Success) {
  Write-Host "tokenizer output was empty/unparsable." -ForegroundColor Red
  Write-Host ("  stdout: " + (($raw | Select-Object -First 3) -join ' '))
  Write-Host ("  stderr: " + ((@(Get-Content -LiteralPath $tokErr -ErrorAction SilentlyContinue) | Select-Object -First 3) -join ' '))
  Die "cannot tokenize the prompt (is the .gguf the right file?)"
}
$idsCsv = (($m.Value -replace '[\[\]]', '') -replace ',', ' ')
$idsCsv = ($idsCsv -split '\s+' | Where-Object { $_ -ne '' }) -join ' '
$nIn = ($idsCsv -split ' ').Count
[IO.File]::WriteAllText($tfile, $idsCsv, [Text.Encoding]::ASCII)
Write-Host "      prompt tokens: $nIn"

if ($KvResident -gt 0 -and $Kv -eq 'k8v4') {
  Write-Host ("      note: --kv-resident cannot be combined with --kv " + $Kv + " -> using fp16 for the KV.") -ForegroundColor Yellow
  $Kv = 'fp16'
}

# ---- run ---------------------------------------------------------------
$argv = @(
  '--pack',            $pack,
  '--native',          $Model,
  '--no-ple',
  '--tokens-file',     $tfile,
  '--expert-profile',  $profPath,
  '--expert-cache',    'auto',
  '--vram-reserve-mib', "$Reserve",
  '--prefill',         "$Prefill",
  '--max-context',     "$Context",
  '--max-new',         "$MaxNew",
  '--kv',              $Kv,
  '--spec', '2', '--suffix-draft', '0', '--mtp-max-t', '1',
  '--pool-workers',    "$Workers",
  '--pool-affinity',   $Affinity,
  '--stats'
)
if ($KvResident -gt 0) { $argv += @('--kv-resident', "$KvResident") }
Write-Host "[2/3] running engine (reserve ${Reserve} MiB, context $Context, max-new $MaxNew) ..." -ForegroundColor DarkGray
$ErrorActionPreference = 'Continue'
& $exe @argv 1> $engOut 2> $engErr
$ErrorActionPreference = 'Stop'
$log = @(Get-Content -LiteralPath $engOut -ErrorAction SilentlyContinue) + @(Get-Content -LiteralPath $engErr -ErrorAction SilentlyContinue)
$logPath = Join-Path $root 'last-run.log'
$log | Set-Content -LiteralPath $logPath -Encoding UTF8

$outLine = @($log | Where-Object { $_ -match '^\s*output\s*:' } | Select-Object -Last 1)
if ($outLine.Count -eq 0) {
  Write-Host ""
  Write-Host "engine produced no 'output' line -- last 15 log lines:" -ForegroundColor Red
  $log | Select-Object -Last 15 | ForEach-Object { Write-Host "  $_" }
  Write-Host "full log: $logPath"
  exit 3
}
$outIds = @((($outLine[0] -replace '^\s*output\s*:', '').Trim() -split '\s+') | Where-Object { $_ -match '^\d+$' })

# ---- detokenize --------------------------------------------------------
Write-Host "[3/3] decoding ..." -ForegroundColor DarkGray
$map = @{}
foreach ($line in [IO.File]::ReadLines($tab)) {
  if ($line.Length -eq 0 -or $line[0] -eq '#') { continue }
  $i = $line.IndexOf("`t")
  if ($i -lt 1) { continue }
  $map[[int]$line.Substring(0, $i)] = $line.Substring($i + 1)
}
$bytes = New-Object 'System.Collections.Generic.List[byte]'
foreach ($id in $outIds) {
  $hex = $map[[int]$id]
  if (-not $hex) { continue }
  for ($k = 0; $k -lt $hex.Length; $k += 2) { $bytes.Add([Convert]::ToByte($hex.Substring($k, 2), 16)) }
}
$answer = [Text.Encoding]::UTF8.GetString($bytes.ToArray())

# ---- report ------------------------------------------------------------
$p = @{}
foreach ($line in $log) {
  if ($line -match 'decode\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s')  { $p['decode']  = $matches[1] }
  if ($line -match 'prefill\s+\d+ tokens in [\d.]+ ms\s+->\s+([\d.]+) tok/s') { $p['prefill'] = $matches[1] }
  if ($line -match 'expert cache auto:\s*(.*)')                                { $p['cache']   = $matches[1] }
  if ($line -match '(R4 expert-cache hits .*)')                                { $p['hits']    = $matches[1] }
  if ($line -match '(the CPU expert pool .*)')                                 { $p['pool']    = $matches[1] }
}
Write-Host ""
Write-Host "=================== ANSWER ===================" -ForegroundColor Cyan
Write-Host $answer
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host ("decode  : " + $p['decode'] + " tok/s      prefill: " + $p['prefill'] + " tok/s   (on " + $nIn + " prompt tokens -- prefill tok/s is only meaningful for long prompts)")
Write-Host ("cache   : " + $p['cache'])
Write-Host ("hits    : " + $p['hits'])
Write-Host ("cpupool : " + $p['pool'])
Write-Host ("log     : " + $logPath)
if ($ShowLog) { Write-Host ""; $log | ForEach-Object { Write-Host "  $_" } }
Write-Host ""
Write-Host "Please report: GPU model + VRAM, system RAM, and the cache / hits / decode lines above."
