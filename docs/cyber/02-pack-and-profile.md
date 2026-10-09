# 02 · 两个派生物：native pack 与 expert profile

> 引擎除 GGUF 之外还要两个派生物。两份都由 `tools/` 下的脚本生成，**各自带自证判据**。
> 证据等级：`A` = 本机自实测 ｜ `B` = 同机对照实现 ｜ `C` = 外部实测 ｜ `D` = 推导或出处自报。

---

## §1 为什么要 pack：Cyber 只能走 `iq_pack.py`

`src/core/expert.hpp:48-51` 逐字说明：native pack（Nano / IQ2_XS / IQ3_XXS 那类 blob 布局）
**不使用** QK/BB/QKA/QROW 的 Q2_0 canvas 布局；`expert_layout().native` 为真时每个 blob 的大小
全部来自 `native_experts.txt`。Cyber 的专家是 **Q4_K / Q5_K / Q6_K** ⇒ **只有 native pack 一条路**。

⚠️ **不要用 `tools/strata_pack.py`**：它有一条**明文记载的静默错件路径**（产出尺寸正确、内容错的 pack）。
`tools/iq_pack.py` 是本项目用的那一个。

---

## §2 打 pack：`tools/iq_pack.py`

### 2.1 命令

```
python tools/iq_pack.py --gguf <models>/Cyber-Tiel-Coder-35B-A3B/MTP-UD-Q4_K_XL.gguf ^
                        --out  <workdir>/iq-native ^
                        --compat-bf16
```

| 参数 | 作用 |
|---|---|
| `--gguf` | 源 GGUF（单文件） |
| `--out` | pack 输出目录 |
| `--compat-bf16` | 把小投影中引擎按 BF16 读的张量反量化/舍入到 BF16（nearest-even）。不加会在遇到这类张量时报 `… but the engine requires BF16; use --compat-bf16` 并拒绝 |
| `--experts-bin`（**本配置不给**） | 只多写一份 `experts.bin`，不改形态。**不给 ⇒ 专家不从 pack 读，引擎直接读 GGUF 原生张量**（`tools/iq_pack.py:502-503` 逐字："the engine otherwise reads the experts from the GGUF itself"） |

### 2.2 输出与读数

| 物件 | 大小 / 读数 |
|---|---|
| `pack/iq-native/`（整目录） | **0.08 GiB** |
| `dense.bin` | **80,835,072 B** |
| `index.txt` | **653 行**（613 canonical + 40 合成零 indexer）—— `# strata pack index v3` 抬头，记录源文件名 |
| `native_experts.txt` | **41 行** = 1 抬头 + **40 主干层**（最后一行数据以 `39 ` 开头） |
| `conversions.json` | 源张量字节哈希逐条（140 条，`iq_pack.py:337` 是 `sha256(raw.tobytes())` —— 是**张量原始字节**的哈希，条条不同是正常的） |
| `compat-bf16.json` | `--compat-bf16` 的舍入记录 |
| `SOURCE.sha256` | **1,651 B**（见 §2.4） |
| `tokenizer/` 子目录 | 词表派生件（`vocab.json` / `tokenizer.json` / `merges.txt` / `chat_template`） |

### 2.3 对本 fork 做的三处适配（`tools/iq_pack.py`）

| # | 改动 | 为什么 | 怎么验证 |
|---|---|---|---|
| 1 | 新增 `nextn_layers()`，读 GGUF 元数据 `<arch>.nextn_predict_layers`（Cyber = 1） | 原实现从**张量**推层数：`n_layers = 1 + max(int(n.split(".")[1]) for n in exps)` ⇒ 最高 `blk.40` ⇒ **算出 41**，而引擎的 `n_layers` 是 **40** 层主干 | 修前：引擎报 `native dense: cannot open`，随后 `a malformed line: 40 …`（`expert_layout.cpp:279` 的 `l >= n_layers` 检查）。修后 pack 正常加载 |
| 2 | MTP 块（`blk.40`）排除出专家表与索引表 | 同上，第 41 行 `l=40 >= 40` 被判 malformed | 报错消失；`native_experts.txt` 最后一行数据以 `39 ` 开头 |
| 3 | 为 **10 个 QSA 层各合成 4 个零张量**（选择器垫片） | 引擎的 decode 路径要过一个 indexer 选择器，而 Cyber **没有 indexer**。合成零张量 + 引擎侧的恒等斜坡 ⇒ 两条路都给出"选满 = dense" | pack 打印 `synthetic indexer tensors: 40`；`index.txt` 653 行 = 613 + 40；引擎日志 `77 MiB of weights loaded … (259 canonical tensors skipped)` |
| 4 | 默认输出目录中的 `index.txt:1` 抬头记录**源文件名** | `iq_pack.py`（以及 `strata_pack.py`）**都不写 `manifest.json`** ⇒ 溯源必须另想办法（见 §2.4） | `index.txt:1` 逐字含源 GGUF 文件名 |

> 4 个合成张量的形状（取自 `layout.cpp:78-79,105-106`，引擎形态取自 `iq_pack.py` 的 FORM 表）：
> `[2048,512] bf16` / `[2048,128] bf16` / `[128] f32` / `[128] f32`。总增量 ≈ **26 MB**。

### 2.4 `SOURCE.sha256`：来源与为什么它绑死源权重

**为什么需要它**：`iq_pack.py` **不写 manifest** ⇒ 红线"pack 与源 GGUF 的哈希回比"在这个工具链上**原判据不存在**。
`SOURCE.sha256` 就是补上的那件东西 —— **它是 pack 与"逐字节确定的源权重"之间的唯一绑定**。

> 这条红线的由来（必须写清楚）：曾发生过一次 **21.9 GB 的文件，大小逐字节正确、哈希全错**的静默事故，
> 未被发现，赔进去整一轮移植工作。

**格式**：`<键> <值>` 的溯源文档（**不是** `<hash> <path>` 清单 —— 按清单解析会 21 行全报 `UNPARSED`）。

**有效键（三个）**：

```
sha256  0BBCF3CC9BE4C976BAD20E641BAF629DAD9C178D39EBDC9CD72129179943C06A
bytes   22749880640
path    <models>/Cyber-Tiel-Coder-35B-A3B/Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf
```

**怎么验证（回比）**：

```
Get-FileHash -Algorithm SHA256 <源 GGUF>
```

与 `sha256` 三键逐字相同 ⇒ 这份 pack 确认来自那一个源。**不一致 = 另一份制品 ⇒ 建在它上面的所有读数作废。**

**⚠️ 一个已登记、不扩散的瑕疵**：`SOURCE.sha256` 的 `verify` 段是**上一代 pack** 的读数
（写 `dense.bin 55,706,112 B` / `630 tensors` / `142 转换`，而当前实际是 `80,835,072 B` / `653 行` / `140 转换`）；
文件时间戳（22:58:58）比 pack（23:02:27）**早 4 分钟**。

| 项 | 判定 |
|---|---|
| `sha256` / `bytes` / `path` 三键 | ✅ **有效**，已回比一致 |
| `verify` 段 | ⚠️ **审计过期**（文档问题，不是 pack 问题） |
| 处置 | **不就地改冻结产物** ⇒ 记录在案，以当前实测指纹为准；原件一字不动 |

### 2.5 pack 的三条判据

| # | 判据 | 怎么变红 |
|---|---|---|
| 1 | pack 落盘且**尺寸 / 条数可复算** | 对不上 ⇒ 停 |
| 2 | `SOURCE.sha256` 的 `sha256` 键 == 源 GGUF 实际 sha256 | 不等 ⇒ 另一份制品，全部读数作废 |
| 3 | **行级字节指纹比对**（`check_row_order.py`） | 行序错 ⇒ 静默错件（**最危险的假绿 = 只验大小不验哈希**） |
| 附 | 引擎能加载并通过结构校验 | 加载器的硬校验：三平面之和必须等于张量，否则拒绝加载 |

---

## §3 采 expert profile：`tools/make_profile.py`

### 3.1 为什么它是命中率的主要来源

`--expert-cache auto` 按 profile 的**顺序**把能负担的槽填满；排名决定哪些专家一开始就常驻。
**不带 profile 的默认策略实测只有 2.97% 命中**（引擎 help 自述）⇒ **必须带**。

### 3.2 两步：先落路由 trace，再排序

**第一步：跑一次带 `--dump-routing` 的引擎臂**（用一段有代表性的 prompt）：

```
strata.exe --pack <workdir>/iq-native --tokens-file <prompt token id 文件> --no-ple ^
           --native <GGUF> --spec 2 --prefill 2048 ^
           --expert-profile <旧 profile>.bin --expert-cache auto --expert-cache-per-layer ^
           --vram-reserve-mib 24568 --max-context 4096 --max-new 256 --mtp-max-t 1 ^
           --dump-routing <workdir>/routing-trace-cur-<stem>.bin
```

trace 格式：每条记录 `int32 layer, int32 k, k 个 expert id, k 个权重`。
实测一次 8 份输入 ⇒ 8 个 trace 文件，**737,280–789,120 B** 各一份；单次记录数示例 `1920 records`（138,240 B）。

负控：`--dump-routing` 若不能与 `--spec` 共存 ⇒ 这条路整条作废。**实测能共存**（`routing dumped …`）。

**第二步：排序出 profile**（一次命令即可）：

```
python tools/make_profile.py <trace1>.bin <trace2>.bin … ^
       --no-base --n-layer 40 --n-expert 256 ^
       --out <workdir>/expert-profile-40x256-trace-cur.bin
```

工具输出（逐字）：

```
wrote <workdir>/expert-profile-40x256-trace-cur.bin: 10240 ranked pairs (0 from the base, 10129 from the traces, 111 filled in)
```

| 参数 | 说明 |
|---|---|
| `--n-layer 40` | **必须**等于引擎的 `n_layers`。见 §3.4 |
| `--n-expert 256` | 每层专家数 |
| `--no-base` | 不保留随包基准排名，完全按 trace 排 |
| `--reorder` | 把 trace 排在 base 前（当 base 已排满全部对时，`--base` 单独用**改不动顺序**） |
| `--out` | 输出 |

### 3.3 产物与自证

| 项 | 读数 |
|---|---|
| 文件 | `profile/expert-profile-40x256-trace-cur.bin`，**81,944 B** |
| sha256 | `FB4D307E2E547F2824AD41E8A0FC1A2B5A6F78921EB960477385FF2E6B87F79E` |
| 头部自证 | `magic=STRP  version=1  n_layer=40  n_expert=256  slots=10240  n_ranked=10240` |
| 尺寸判据（**真式**） | `24 + 4·len(ranked) + 4·N_LAYER·n_expert = 24 + 4·10240 + 4·40·256 = 81,944` ✅ **逐字节命中** |
| 尺寸判据（塌缩式） | `24 + 8·N_LAYER·n_expert = 81,944` —— 只在"满排名"时成立；加"部分排名"模式后此式立刻变假绿。**写成真式** |

### 3.4 对本 fork 做的一处适配：新增 `--n-layer`

| 项 | 内容 |
|---|---|
| 改动 | 新增 `--n-layer`（默认 `N_LAYER` = 48）；`read_profile` / `read_trace` / `write_profile` / `rank_profile` 四个函数都加 `n_layer` 参数；`main()` 改成关键字传参 |
| 为什么 | 原来**只有 `--n-expert`**。`--n-expert 256` 单独用会写出 **48×256** 的表，而**工具自己的 round-trip 断言抓不到**：`write_profile` 与 `read_profile` 比对的是**同一个模块常量**，错文件照样通过 |
| 代价 | 引擎端**会拦**（`src/core/expert_cache.cpp:42-48` 逐字 `read_expert_profile: %s is %ux%u but this model is %lldx%lld`，`:69` 逐对查越界）⇒ **大声失败，但代价是"一整个 pack 之后"才发现**，且工具自检本身依然是瞎的 |
| 怎么验证 | 用 `--n-layer 40` 出件，尺寸必须逐字节命中 §3.3 真式；故意只给 `--n-expert 256` ⇒ 引擎在 LOAD 期报 `is 48x256 but this model is 40x256` |

### 3.5 profile 的效果与判据

| 判据 | 旧 `trace8` | 新 `trace-cur` |
|---|---|---|
| 命中率 `R4 expert-cache hits` | 0.6372（4 次逐字相同） | **0.7837–0.7881** |
| CPU 专家 distinct/routed 每层 | 2.35 / 2.41 | **1.61–1.77** |
| `pool` ms/轮 | 8.20 | **6.26** |
| `-> N slots` | **1963** | **1963**（逐字相同 ⇒ 排除"偷槽"） |
| 同批交错 decode | 50.86（48.38–53.01，n=4） | **63.85（62.29–67.08，n=4）** |

配对下界 = `62.29 − 53.01 = +9.28 tok/s` > 噪声底 5.75 ⇒ **有信号**。

**负控（必须有牙）**：**故意 shuffle profile ⇒ 命中率必须塌回容量占比**。
若 shuffle 后命中率不动 ⇒ 判据本身失效，不是"profile 不重要"。

---

## §4 出处

| 结论 | 素材 |
|---|---|
| pack 命令与三个参数、`--experts-bin` 语义 | `rel/RUN-S6-打pack与红线R2判据替换.md` §0、§（`--experts-bin` 注）；`<repo>/tools/iq_pack.py:594-601` |
| native pack 必要性、Q2_0 canvas 原文 | `rel/RUN-S2-实勘第三段-三条硬耦合链.md` §（引用 `expert.hpp:48-51`） |
| `strata_pack.py` 静默错件、工具链现成度 | `rel/RUN-阶段3-开工前必查.md` §4.2、§4.3 |
| 4 处适配（`nextn_layers` / MTP 排除 / 零 indexer 垫片 / 抬头） | `rel/RUN-S11-试驾实录-阻断链.md` §8.3、§3 表；`rel/跳步记录.md` #4；`rel/RUN-S2-实勘第二段-indexer硬墙.md` §（甲方案） |
| pack 物件尺寸（`dense.bin` 80,835,072 B、653 行） | `rel/发货参数与显卡分档调优.md` §2 |
| `SOURCE.sha256` 三键有效、`verify` 段过期、原件不就地改 | `rel/RUN-S11-试驾实录-阻断链.md` §17.3、§17.4；`rel/状态收拢-v1.md` §3.2、§7；`rel/状态收拢-v2.md` §（R2 行） |
| 21.9 GB 假绿事故 | `SOURCE.sha256` 抬头注释（第 2–3 行）；`rel/RUN-S6-打pack与红线R2判据替换.md` §（红线 R2 由来） |
| profile 两步命令与逐字输出 | `rel/RUN-S7-S10-S11-产物就位与试驾准备.md` §1；`rel/_s144-a2-report.md` §（采集链路） |
| profile 尺寸真式 C8、槽数与表内容无关 | `rel/总纲与分支协议.md` §2.3 C8；`rel/_s144-a2-report.md` §（判据①） |
| `--n-layer` 改动与 C9 | `rel/RUN-S7-S10-S11-产物就位与试驾准备.md` §1.1、§1.2；`rel/总纲与分支协议.md` C9 |
| 新旧 profile 对照读数与负控 | `rel/_s144-a2-report.md`（对照读数表、判据①②③） |
| 不带 profile 只有 2.97% 命中 | `rel/发货参数与显卡分档调优.md` §3.2（`--expert-profile P` 行） |
