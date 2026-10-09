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
    -KvResident <N>     DO NOT USE. Keep it at 0 (the default). N > 0 parks all but the last N KV
                        cells of each layer in pinned RAM; the engine then MASKS the cells it did
                        not make resident, so attention sees only a window - and at 20480 it also
                        stalls the prompt READ (a 21k prompt read 18432 tokens and generated 0).
                        Measured 2026-10-10; the launcher's 20480 default was wrong before v2.
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
  [int]$KvResident = 0,
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

# ---- what card is this? (a SUGGESTION only; it never overrides -Reserve) -------------------
# The defaults in this package are the 8 GB tier, so a small card needs no changes.  A bigger card
# should LOWER -Reserve (the cache is sized `free - reserve`), so print the card's tier up front.
function SuggestedReserve([int]$mib) {
  if ($mib -le 0)     { return 1500 }   # unknown -> the shipped 8 GB default
  if ($mib -le 9000)  { return 1500 }   # 8 GB
  if ($mib -le 13000) { return 2500 }   # 10-12 GB
  if ($mib -le 18000) { return 3500 }   # 16 GB
  if ($mib -le 26000) { return 5000 }   # 24 GB
  return 1500                           # 32 GB and up: 1500 already lets the cache hold all 10240
                                        # experts (measured) -- a bigger reserve only takes VRAM
                                        # away from the KV and the desktop
}
$gpuName = ""; $gpuMib = 0
try {
  $q = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
  if ($q) {
    $f = (@($q)[0] -split ',')
    if ($f.Count -ge 2) { $gpuName = $f[0].Trim(); [void][int]::TryParse($f[1].Trim(), [ref]$gpuMib) }
  }
} catch { }
if (-not $gpuName) {
  try {
    $vc = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA|Radeon|Arc' })
    if ($vc.Count -gt 0) {
      $gpuName = $vc[0].Name
      if ($vc[0].AdapterRAM -gt 0 -and $vc[0].AdapterRAM -lt 4000000000) { $gpuMib = [int]($vc[0].AdapterRAM / 1MB) }
    }
  } catch { }
}
if ($gpuName) {
  $sug = SuggestedReserve $gpuMib
  $shown = if ($gpuMib -gt 0) { "$gpuName ($gpuMib MiB)" } else { $gpuName }
  Write-Host ("      GPU: " + $shown) -ForegroundColor DarkGray
  if ($sug -ne $Reserve) {
    Write-Host ("      note: for this card -Reserve " + $sug + " would give a bigger expert cache " +
                "(you asked for " + $Reserve + ").  It is the ONLY knob that must match the card -- " +
                "see section 4 of the shipping doc.") -ForegroundColor Yellow
  }
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
Write-Host ""
Write-Host "Tuning (the package default is the 8 GB tier -- change these ONLY on a bigger card):" -ForegroundColor DarkGray
Write-Host "  -Reserve   8 GB 1500 | 12 GB 2500 | 16 GB 3500 | 24 GB 5000 | 32 GB 1500 (= every expert resident)" -ForegroundColor DarkGray
Write-Host "  -Context   8 GB 4096 | 12 GB 8192 | 16 GB 16384 | 24 GB+ 32768" -ForegroundColor DarkGray
Write-Host "  -Prefill   8 GB 2048 | 12 GB 4096 | 16 GB+ 8192   (it borrows expert-cache slots while the prompt runs)" -ForegroundColor DarkGray
Write-Host "  watch the engine's own line: expert cache auto: X GiB free, Y MiB reserved -> N slots" -ForegroundColor DarkGray
Write-Host "  keep -KvResident at 0: a nonzero value makes the engine DROP context it did not keep resident." -ForegroundColor DarkGray
