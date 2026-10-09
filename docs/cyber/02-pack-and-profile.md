# 02 · 两个派生物：native pack 与 expert profile

> 引擎除 GGUF 之外还要两个派生物，都由 `tools/` 下的脚本生成，**各自带自证判据**。
> 证据等级：`A` 本机自实测 ｜ `B` 同机对照实现 ｜ `C` 外部实测 ｜ `D` 推导或出处自报。

---

## §1 为什么必须走 native pack

`include/strata/kernels/cpu/expert.hpp` 的说明注释逐字指出：native pack（Nano / IQ2_XS / IQ3_XXS 那类 blob 布局）
**不使用** QK/BB/QKA/QROW 的 Q2_0 canvas 布局；`expert_layout().native` 为真时每个 blob 的大小全部来自
`native_experts.txt`。Cyber 的专家是 **Q4_K / Q5_K / Q6_K** ⇒ **只有 native pack 一条路**。

⚠️ **不要用 `tools/strata_pack.py`**：它有一条**静默错件路径**（产出尺寸正确、内容错）。
本项目用 `tools/iq_pack.py`。

---

## §2 打 pack：`tools/iq_pack.py`

```
python tools/iq_pack.py --gguf <models>/…-MTP-UD-Q4_K_XL.gguf ^
                        --out  <workdir>/iq-native ^
                        --compat-bf16
```

| 参数 | 作用 |
|---|---|
| `--gguf` / `--out` | 源 GGUF（单文件）/ 输出目录 |
| `--compat-bf16` | 把小投影中引擎按 BF16 读的张量反量化/舍入到 BF16（nearest-even）。不加会报 `… but the engine requires BF16; use --compat-bf16` 并拒绝 |
| `--experts-bin`（**本配置不给**） | 多写一份 `experts.bin`，不改形态。**不给 ⇒ 专家不从 pack 读，引擎直接读 GGUF 原生张量** |

### 2.1 输出

| 物件 | 大小 / 读数 |
|---|---|
| `pack/iq-native/`（整目录） | **0.08 GiB** |
| `dense.bin` | **80,835,072 B** |
| `index.txt` | **653 行**（613 canonical + 40 合成零 indexer）；抬头 `# strata pack index v3`，记录源文件名 |
| `native_experts.txt` | **41 行** = 1 抬头 + **40 主干层**（最后一行数据以 `39 ` 开头） |
| `conversions.json` | 140 条，每条是**张量原始字节**的 sha256（条条不同是正常的） |
| `compat-bf16.json` | `--compat-bf16` 的舍入记录（142 条转换，142 exact，0 rounded，max \|err\| = 0） |
| `SOURCE.sha256` | **1,651 B**（见 §2.3） |
| `tokenizer/` | 词表派生件（`vocab.json` / `tokenizer.json` / `merges.txt` / `chat_template`） |

### 2.2 对本 fork 做的四处适配（`tools/iq_pack.py`）

| # | 改动 | 为什么 | 怎么验证 |
|---|---|---|---|
| 1 | 新增 `nextn_layers()`，读 `<arch>.nextn_predict_layers`（Cyber = 1），`n_layers = total - nextn` | 原实现从**张量**推层数：最高 `blk.40` ⇒ **算出 41**，而引擎 `n_layers` 是 **40** 层主干 | 修前报 `a malformed line: 40 …`，修后正常加载 |
| 2 | MTP 块（`blk.40`）排除出专家表与索引表 | 同上，第 41 行 `l=40 >= 40` 被判 malformed | `native_experts.txt` 最后一行数据以 `39 ` 开头 |
| 3 | 为 **10 个 QSA 层各合成 4 个零张量**（形状 `[2048,512]` bf16 / `[2048,128]` bf16 / `[128]` f32 / `[128]` f32） | 引擎 decode 路径要过 indexer 选择器，而 Cyber **没有 indexer**。合成零张量（选满）+ 引擎侧恒等斜坡 ⇒ 两条路都给出"选满 = dense" | pack 打印 `synthetic indexer tensors: 40`；`index.txt` 653 行 = 613 + 40。总增量 ≈ **26 MB** |
| 4 | `index.txt` 抬头记录源文件名 | `iq_pack.py`（以及 `strata_pack.py`）**都不写 `manifest.json`** ⇒ 溯源必须另找载体 | 抬头逐字含源 GGUF 文件名 |

### 2.3 `SOURCE.sha256`：来源与为什么它绑死源权重

**为什么需要它**：`iq_pack.py` 不写 manifest ⇒ "pack 与源 GGUF 的哈希回比"这条红线**原判据不存在**。
`SOURCE.sha256` 就是补上的东西 —— **pack 与"逐字节确定的源权重"之间的唯一绑定**。
（红线由来：曾发生 **21.9 GB 的文件、大小正确、哈希全错**的静默事故，未被发现。）

**格式**：`<键> <值>` 的溯源文档（**不是** `<hash> <path>` 清单 —— 按清单解析会 21 行全报 `UNPARSED`）。

```text
sha256  0BBCF3CC9BE4C976BAD20E641BAF629DAD9C178D39EBDC9CD72129179943C06A
bytes   22749880640
path    <models>/…/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
```

**怎么验证**：`Get-FileHash -Algorithm SHA256 <源 GGUF>` 与 `sha256` 键逐字相同 ⇒ 确认来源。
**不一致 = 另一份制品 ⇒ 建在它上面的所有读数作废。**

**⚠️ 已登记、不扩散的瑕疵**：`verify` 段是**上一代 pack** 的读数（写 `dense.bin 55,706,112 B` / `630 tensors`，
实际 `80,835,072 B` / `653 行`），且文件时间戳比 pack **早 4 分钟**。
⇒ **只有 `sha256` / `bytes` / `path` 三键有效**（已回比一致）；**原件不就地改**，以当前实测指纹为准。

### 2.4 pack 的三条判据

| # | 判据 | 怎么变红 |
|---|---|---|
| 1 | pack 落盘且**尺寸 / 条数可复算** | 对不上 ⇒ 停 |
| 2 | `SOURCE.sha256` 的 `sha256` 键 == 源 GGUF 实际 sha256 | 不等 ⇒ 另一份制品，全部读数作废 |
| 3 | **行级字节指纹比对** | 行序错 ⇒ 静默错件（**最危险的假绿 = 只验大小不验哈希**） |
| 附 | 引擎能加载并通过结构校验 | 加载器硬校验：三平面之和必须等于张量，否则拒绝 |

---

## §3 采 expert profile：`tools/make_profile.py`

`--expert-cache auto` 按 profile 的**顺序**把能负担的槽填满；排名决定哪些专家一开始就常驻。
**不带 profile 的默认策略实测只有 2.97% 命中**（引擎 `--help` 自述）⇒ **必须带**。

### 3.1 两步

**① 跑一次带 `--dump-routing` 的引擎臂**（prompt 要有代表性）：

```
strata.exe --pack <workdir>/iq-native --tokens-file <prompt ids> --no-ple ^
           --native <GGUF> --spec 2 --prefill 2048 ^
           --expert-profile <旧 profile>.bin --expert-cache auto --expert-cache-per-layer ^
           --vram-reserve-mib 24568 --max-context 4096 --max-new 256 --mtp-max-t 1 ^
           --dump-routing <workdir>/routing-trace-<stem>.bin
```

trace 格式：每条记录 `int32 layer, int32 k, k 个 expert id, k 个权重`。实测一次 8 份输入 ⇒ 8 个 trace 文件，
各 **737,280–789,120 B**；单次记录数示例 `1920 records`（138,240 B）。
**负控**：`--dump-routing` 若不能与 `--spec` 共存 ⇒ 这条路整条作废。**实测能共存**（日志 `routing dumped …`）。

**② 排序出 profile**（一条命令）：

```
python tools/make_profile.py <trace1>.bin … <trace8>.bin ^
       --no-base --n-layer 40 --n-expert 256 ^
       --out <workdir>/expert-profile-40x256-trace-cur.bin
```

工具输出（逐字）：`10240 ranked pairs (0 from the base, 10129 from the traces, 111 filled in)`

| 参数 | 说明 |
|---|---|
| `--n-layer 40` | **必须**等于引擎的 `n_layers`（见 §3.3） |
| `--n-expert 256` | 每层专家数 |
| `--no-base` | 不保留随包基准排名，完全按 trace 排 |
| `--reorder` | 把 trace 排在 base 前（base 已排满全部对时，`--base` 单独用**改不动顺序**） |

### 3.2 产物与自证

| 项 | 读数 |
|---|---|
| 文件 | `profile/expert-profile-40x256-trace-cur.bin`，**81,944 B** |
| sha256 | `FB4D307E2E547F2824AD41E8A0FC1A2B5A6F78921EB960477385FF2E6B87F79E` |
| 头部自证 | `magic=STRP  version=1  n_layer=40  n_expert=256  slots=10240  n_ranked=10240` |
| 尺寸判据（**真式**） | `24 + 4·len(ranked) + 4·N_LAYER·n_expert = 24 + 4·10240 + 4·40·256 = 81,944` ✅ **逐字节命中** |
| 尺寸判据（塌缩式） | `24 + 8·N_LAYER·n_expert` —— 只在"满排名"时成立；加"部分排名"模式后立刻变假绿。**要写真式** |

### 3.3 对本 fork 做的一处适配：新增 `--n-layer`

| 项 | 内容 |
|---|---|
| 改动 | 新增 `--n-layer`（默认 `N_LAYER` = 48）；`read_profile` / `read_trace` / `write_profile` / `rank_profile` **四个函数**都加 `n_layer` 参数；`main()` 改成关键字传参 |
| 为什么 | 原来**只有 `--n-expert`**。`--n-expert 256` 单独用会写出 **48×256** 的表，而**工具自己的 round-trip 断言抓不到**：`write_profile` 与 `read_profile` 比对的是**同一个模块常量**，错文件照样通过 |
| 代价 | 引擎端**会拦**（`src/core/expert_cache.cpp` 报 `read_expert_profile: %s is %ux%u but this model is %lldx%lld`，并逐对查越界）⇒ **大声失败，但代价是"一整个 pack 之后"才发现**，且工具自检本身依然是瞎的 |
| 怎么验证 | 用 `--n-layer 40` 出件，尺寸必须逐字节命中 §3.2 真式；故意只给 `--n-expert 256` ⇒ 引擎在 LOAD 期报 `is 48x256 but this model is 40x256` |

### 3.4 profile 的效果与判据

| 判据 | 旧 `trace8` | 新 `trace-cur` |
|---|---|---|
| 命中率 `R4 expert-cache hits` | 0.6372（4 次逐字相同） | **0.7837–0.7881** |
| CPU 专家 distinct/routed 每层 | 2.35 / 2.41 | **1.61–1.77** |
| `pool` ms/轮 | 8.20 | **6.26** |
| `-> N slots` | **1963** | **1963**（逐字相同 ⇒ 排除"偷槽"） |
| 同批交错 decode | 50.86（48.38–53.01, n=4） | **63.85（62.29–67.08, n=4）** |

配对下界 = `62.29 − 53.01 = +9.28 tok/s` > 噪声底 5.75 ⇒ **有信号**。

**负控（必须有牙）**：**故意 shuffle profile ⇒ 命中率必须塌回容量占比**。
若 shuffle 后命中率不动 ⇒ 判据本身失效，不是"profile 不重要"。

---

## §4 出处

| 结论 | 证据 |
|---|---|
| pack 命令与参数语义、`--experts-bin` 的行为 | `tools/iq_pack.py` 的参数定义与帮助文本 |
| native pack 的必要性、Q2_0 canvas 与 native blob 布局 | `include/strata/kernels/cpu/expert.hpp` 的说明注释 |
| `strata_pack.py` 的静默错件风险 | `tools/strata_pack.py` 与 `tools/iq_pack.py` 的读码对比 + 本机实跑 |
| 四处适配 | 本机实跑：修前报 `a malformed line: 40 …`，修后正常加载（本文档 `09-fork-changelog.md` §4.1） |
| pack 物件尺寸（`dense.bin` 80,835,072 B、653 行、41 行专家表） | 本机文件字节数与行数实测 |
| `SOURCE.sha256` 三键有效、`verify` 段过期、原件不就地改 | 本机 sha256 回比 + 文件时间戳比对（早 4 分钟） |
| 21.9 GB 假绿事故 | `SOURCE.sha256` 抬头注释里的历史登记 |
| profile 两步命令与工具逐字输出 | 本机实跑 `--dump-routing` + `tools/make_profile.py` |
| profile 尺寸真式 | `tools/make_profile.py` 的写入格式 + 本机产物 81,944 B 逐字节命中 |
| `--n-layer` 改动与"工具自检是瞎的" | `tools/make_profile.py` 的 round-trip 断言 + `src/core/expert_cache.cpp` 的加载期校验 |
| 新旧 profile 对照读数与"slots 逐字相同"判据 | 本机同批交错 8 臂（拉丁方，首臂丢弃） |
| 不带 profile 只有 2.97% 命中 | 引擎 `--help` 自述 |
| profile 格式头部（`MAGIC` = `STRP` + `<5I>` + 每对 `<HH>` + 每层 `int32` 表） | `tools/make_profile.py` 的读写实现 |
