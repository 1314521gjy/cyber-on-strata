# 03 · 跑起来：命令行与"文本进文本出"的启动器

> 引擎的输入是 **token id**，输出也是 **token id**。本文给出完整命令行模板、每个参数的作用，
> 以及包内启动器怎么把两边接成分词 / 反分词。
> 证据等级：`A` = 本机自实测 ｜ `B` = 同机对照实现 ｜ `C` = 外部实测 ｜ `D` = 推导或出处自报。

---

## §1 引擎为什么自己不带 tokenizer

`src/program/generate.cpp:9-12` 逐字：

> **There is no tokenizer here.** `pack/full/tokenizer/` 和 `tools/strata_tokenizer.py` 存在，
> 而 C++ BPE 是 Phase 1 的交付物、不是这个程序的；所以 prompt 以 **ides** 形式经 `--tokens` 到达。
> 这不是占位符：这正是 Gate C1 需要的 —— C1 在**同一串 id** 上把 logits 与 llama.cpp 对照，
> 而只有一侧带 tokenizer 会成为第二个变量。

缺 `--tokens` 时的报错：`strata generate: --tokens is required (this build has no tokenizer; see the …)`。

**⇒ 分词与反分词必须由外部提供。** 本包的补法见 §3。

---

## §2 完整命令行模板

```
strata.exe ^
  --pack        <workdir>/iq-native ^
  --native      <models>/…MTP-UD-Q4_K_XL.gguf ^
  --no-ple ^
  --tokens-file <prompt 的 token id 文件> ^
  --expert-profile <workdir>/expert-profile-40x256-trace-cur.bin ^
  --expert-cache auto ^
  --vram-reserve-mib 24568 ^
  --prefill 2048 ^
  --max-context 4096 ^
  --max-new 512 ^
  --kv k8v4 ^
  --spec 2 --suffix-draft 0 --mtp-max-t 1 ^
  --pool-workers 6 --pool-affinity all ^
  --stats
```

> 上面这一套是**本机验证过的基准 argv**（`A` 级）。发货默认值见 `07-shipping-params.md`。

---

## §3 参数逐个说明

### 3.1 输入与身份

| 参数 | 作用 | 本配置的取值与理由 |
|---|---|---|
| `--pack DIR` | native pack 目录 | 必给。见 `02-pack-and-profile.md` |
| `--native FILE` | 源 GGUF（**专家从这里读**） | 必给 |
| `--no-ple` | 显式关闭 PLE | **Cyber 没有 PLE 表** ⇒ 不关的话引擎会索要 `--ple-gguf`，而它会静默改变下游所有数字。**这是"无表可用"下的必然设置，不是消融** |
| `--tokens LIST` / `--tokens-file FILE` | prompt 的 token id | **必填**（§1） |
| `--expert-profile P` | 专家常驻排名的来源 | **必给**。不带则命中率只有 2.97%（引擎 help 自述） |

### 3.2 显存预算（**唯一必须按卡改的参数在这一节**）

| 参数 | 它到底决定什么 | 发货规则 | 本机值 |
|---|---|---|---|
| **`--vram-reserve-mib N`** | **从专家缓存里扣掉、留给 KV / 激活 / 桌面的显存**（`缓存 = free − reserve`） | 🔴 **这是唯一必须按卡改的参数**。判据线是引擎自己打印的那一行：`expert cache auto: X GiB free, Y MiB reserved -> N slots`。先把 reserve 设小，看它报多少槽，再往上加直到留够 KV | 24568（**故意的限流值，8 GB 卡绝不能用**） |
| `--expert-cache auto` | 按 `free − reserve` 自动定槽数 | 发货一律 `auto`（写死槽数只用于实验） | auto |
| `--max-context N` | KV 容量。4k 时 KV ≈ 100 MB 级（可忽略）；越长越吃显存 | 8 GB 卡建议 ≤8192 起步；要长上下文就配 `--kv` 压位宽 | 4096 |
| `--kv fp16\|int8\|q4_0\|k8v4` | KV 位宽 | 显存紧就 `int8`；极端紧就 `k8v4`（有精度代价，**未测** ⇒ `D`） | `k8v4` |
| `--kv-resident N` | KV 流式：每层只留 N 格在显存（下限 **20480**），整份 K/V 放**钉住的宿主内存**，腾出的显存给专家槽 | 长上下文时用；**与 `--kv k8v4` / `q4_0` 不能同用**（引擎拒绝） | 0（未用） |
| `--prefill CHUNK` | 预填分块；`auto` = 能借多少缓存借多少（上限 8192） | 显存富余用 `auto`；紧就用 2048 | 2048 |

### 3.3 计算与调度

| 参数 | 作用 | 说明 |
|---|---|---|
| `--max-new N` | 最多生成多少 token | 交付臂用 512；实验臂常压到 8–32 |
| `--pool-workers N` | CPU 专家池线程数（默认 0 = 物理核 − 1） | ⚠️ 本机 6 与 4 **同带（NO SIGNAL）**⇒ 别指望它；按物理核数走 |
| `--pool-affinity all\|auto\|p-cores` | 混合核（大小核）CPU 的绑核 | 12/13/14 代 Intel 用 `auto`；纯大核或 AMD 用 `all` |
| `--spec 2` | 验证窗口宽度 | native pack 强制 `spec ≥ 2`；窗口是**纯成本**（`routed/layer` 1.26 → 1.85 → 2.55 对应窗口 2/4/spec4，而 `tokens/round` 恒 1.00） |
| `--mtp-max-t 1` | 把 **MTP 自己的窗口**压到最小（`0` = 跟随 `--spec`） | ⚠️ 口径容易混：`--spec 2` 是**验证窗口**，`--mtp-max-t` 是 **MTP 的窗口**。没有草稿头时 `1` 是对的；**要真用 MTP 必须给 `0` 或 ≥2** |
| `--suffix-draft 0` | 后缀草稿长度 | 基准口径 |
| `--dump-routing FILE` | 落盘每层每位置的路由（`int32 layer, int32 k, k ids, k weights`） | 采 profile 用；**实测可与 `--spec` 共存** |
| `--stats` | 打印统计 | **发货必须带**；它是回报的唯一机器可读来源 |
| `--stop-eos` / `--no-stop-eos` | 是否在模型自己的 EOS 上停机 | 默认 **`true`**（`generate.cpp:492`）；`--no-stop-eos` 是负控（raw 续写） |

### 3.4 低内存档与多卡

| 参数 | 说明 |
|---|---|
| `--mmap-experts` / `--resident-experts` | 放弃钉住的宿主银行，退回 mmap（低内存档）。内存 < 32 GB 才用；**冷读约慢一倍**（源码注释自报 71.97 vs 34.78 ms/token，`D`） |
| `--peer-device N` / `--expert-cache-device1..3 N\|auto` | 第二张卡当专家层（P2P）。**本机完全未验** |
| `--vram-elastic` / `--batch N` / `--slots N` | 只在 `--serve` 下有意义 ⇒ **未验** |

---

## §4 "文本进文本出"的启动器

### 4.1 组成

| 文件 | 作用 |
|---|---|
| `run.cmd` | 4 行包装（`@echo off` → `powershell -File run.ps1 %*`） |
| `run.ps1` | 主逻辑：分词 → 引擎 → 反分词 → 回报 |
| `engine/tokenize/` | `llama-tokenize.exe` + 7 个 DLL，共 **14 MB**；**实测可用**（`9419 -> 'Hello'`） |
| `engine/vocab_bytes.tsv` | **5,567,833 B**，248,320 条 `id<TAB>hexbytes` 的反分词表；**实测可用** |

### 4.2 三步

```
run.cmd -Prompt "用 python 写一个快速排序"
```

| 步 | 做什么 | 具体做法 |
|---|---|---|
| `[1/3]` 分词 | 文本 → token id | `llama-tokenize.exe -m <GGUF> -f <prompt.txt> --ids`；从 stdout 里取出第一对 `[...]`，去掉括号与逗号得一串空格分隔的 id，写进临时 `tokens.txt` |
| `[2/3]` 引擎 | 跑 `strata.exe`，argv 由启动器参数拼装 | 完整日志落到包根目录 `last-run.log` |
| `[3/3]` 反分词 | token id → 文本 | 读 `vocab_bytes.tsv`，把每个 id 映射成 **hex 字节串**，拼成一个字节数组，再整体 `UTF8` 解码 |

**为什么反分词要用 `id → 字节` 表、而不是逐 token 查词表拼字符串**：
BPE 的一个 token 可能只是某个 UTF-8 字符的一个字节片段。**必须先把所有 token 的字节拼好、再整体解码**，
逐 token 解码会在多字节字符处产生乱码。

### 4.3 启动器参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `-Prompt "文本"` | — | 直接给提示词 |
| `-PromptFile 文件` | — | 从 UTF-8 文本文件读提示词 |
| `-Model 路径` | `models\` 里第一个 `*.gguf` | 模型路径 |
| `-Reserve 数字` | **1500** | ← **按显存分档表填，这是唯一必须按卡改的** |
| `-Context N` | 4096 | `--max-context` |
| `-MaxNew N` | 256 | 最多生成多少 token |
| `-Prefill N` | 2048 | `--prefill` |
| `-Workers N` | 0（自动） | `--pool-workers` |
| `-Affinity all\|auto\|p-cores` | auto | 大小核绑核 |
| `-Kv fp16\|int8\|k8v4\|q4_0` | **int8** | KV 精度（引擎自带默认是 fp16） |
| `-KvResident N` | **20480** | 每层留 N 格 KV 在显存、其余放钉住的内存（0 = 关）。长上下文时腾显存 |
| `-Chat` | 关 | 用 chat 标记包一层（**实验性**） |
| `-ShowLog` | 关 | 额外打印引擎原始日志 |

**启动器的两条内置保护**：

1. **`--kv-resident` 与 `--kv k8v4` 不能同用** ⇒ 检测到这种组合时**自动改成 fp16 并提示**
   （引擎原话：`--kv k8v4 does not support --kv-resident streaming (yet)`）。
2. 启动器会检查包内 `engine/strata.exe`、`engine/tokenize/llama-tokenize.exe`、`engine/vocab_bytes.tsv`、
   `pack/iq-native` 四个物件是否在场，缺一个就 `ERROR: package file missing:` 退出。

### 4.4 跑完打印的三行回报

```
decode  : <tok/s>   tok/s      prefill: <tok/s> tok/s   (on <N> prompt tokens …)
cache   : <expert cache auto: X GiB free, Y MiB reserved -> N slots>
hits    : <R4 expert-cache hits …>
cpupool : <the CPU expert pool … ms/token>
log     : <last-run.log 路径>
```

`last-run.log` 是回报的载体 —— 完整引擎日志都在里面。

---

## §5 起跑前的环境断言（**缺一条就会得到假绿**）

| # | 断言 | 为什么 |
|---|---|---|
| 1 | **CUDA 13 的 DLL 目录在 `PATH`** | 否则进程静默死亡、**日志 0 字节**、批处理仍报成功 |
| 2 | 无残留引擎进程 | 残留会让新臂的显存与带宽读数不可信 |
| 3 | 显存接近空闲（本机取 `≤ 2500 MiB` 才放行） | 见 `07-shipping-params.md` §（可用显存比标称小） |
| 4 | **同一时刻只跑一个引擎** | 每个实例钉住约 19.7 GB 宿主内存 + 数 GiB 显存 |
| 5 | 跑前记录 `strata.exe` 的 sha256 | 事后必须复核未变（否则读数归属不明） |

---

## §6 出处

| 结论 | 素材 |
|---|---|
| 无 tokenizer 原文、`--tokens is required` | `<repo>/src/program/generate.cpp:9-12`、`:1775` |
| `--no-ple` 是必然设置、PLE 表不存在 | `rel/跳步记录.md` #42 |
| 基准 argv 与各参数的发货规则 | `rel/发货参数与显卡分档调优.md` §3.1、§3.2 |
| `--kv-resident` 与 `k8v4` 互斥、下限 20480 | `<repo>/src/program/generate.cpp:588-589`、`:1811`；`rel/发货参数与显卡分档调优.md` §4.3 |
| `--spec` / `--mtp-max-t` 口径、窗口是纯成本 | `rel/跳步记录.md` #101；`rel/状态收拢-v2.md` §2（`--mtp-max-t 1` 行） |
| 启动器三步、`--ids`、`id→bytes` 表、内置保护 | `run.ps1`（包内，实测件）；`rel/发货参数与显卡分档调优.md` §1.3、§3.3 |
| `llama-tokenize` + 7 DLL = 14 MB、`vocab_bytes.tsv` 5,567,833 B / 248,320 条 | `rel/发货参数与显卡分档调优.md` §2（物件清单） |
| tokenizer 抽查读数（`9419 -> 'Hello'`、`760 -> 'The'`、roundtrip 全等） | `rel/RUN-S7-S10-S11-产物就位与试驾准备.md` §2 |
| CUDA 13 DLL 在 PATH 的断言 | `rel/_s150-baseline-arms.md` §0.1；`rel/RUN-S23-性能诊断.md` §7 |
| 起跑断言四条 | `rel/_s150-baseline-arms.md` §3.1、§6 |
