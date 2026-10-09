# 09 · 本 fork 相对上游的改动清单

> 按文件 / 模块分组。每条一句话说清"**改了什么 · 为什么 · 怎么验证的**"。
> 证据等级：`A` = 本机自实测 ｜ `B` = 同机对照实现 ｜ `C` = 外部实测 ｜ `D` = 推导或出处自报。
> 行号为改动当时的记录值；行内容按现行源码逐字复核。
> ⚠️ 标 **`清单外`** 的条目在项目自述的改动清单里**没有**，是本次核对源码时发现的（原清单不完整）。

---

## §1 数值正确性

### 1.1 GDN z 闸门：sigmoid → SiLU（5 处 kernel + 1 处注释）

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/kernels/cuda/verify_kernels.cu` | `:227` | z 闸门用 `silu` 而非 `sigmoid` | 参考实现（`qwen35moe.cpp:273 ggml_silu`）用 SiLU；用 sigmoid 是**读错** | 32 token 文本由 `整是inatom…` 变为与参考**逐字符相同（88/88）** |
| `src/kernels/cuda/fused_gdn.cu` | `:69` | 同上 | 同上 | 日志注释逐字："Was sigmoid - that was the misreading." |
| `src/kernels/cuda/gdn.cu` | `:148-149` | 同上（`py[i] = po[i]*inv*ssm_norm[i]*(pz[i]*sigmoid_f(pz[i]))`） | 同上；`sigmoid_f` 本身**与 beta 闸门共享，不许改** | 见 `04-correctness.md` §4.1 七条判据 |
| `src/kernels/cuda/native_gdn_preprocess.cu` | `:111` | 同上 | 同上；该文件里的 `sigmoid` 与 beta 共享，注释明写"do not change it" | 同上 |
| `src/prefill/kernels.cu` | `:642-643` | 同上 | 同上；`sigm` 与 beta 共享，注释明写"do not change it" | 同上 |
| `src/core/layer.cpp` | `:317-318` | **注释**订正 | 旧注释写 "SIGMOID, not SiLU"，而它转述自一个**全树不存在**的参考文件 ⇒ 错注释会继续误导后来者 | 新注释记录："The old comment here said 'SIGMOID, not SiLU' and that misreading cost …" |

**判据汇总（改前 → 改后，`A` 级）**：

| 判据 | 改前 | 改后 |
|---|---|---|
| 首位 token | `107461`（"整"） | **`514`** |
| 32 token 文本 | `整是inatom.manNEnenenene…` | **与参考逐字符相同（88/88）** |
| `785` 死循环 | 位置 5–8 四连 | **0 次**，最长连续重复 = 1 |
| 与参考 logits 的 cos | −0.5124 / −0.0931 | **+0.999658522 / +0.999143362** |
| Spearman（248,320 维） | −0.0736 | **+0.9987** |
| **回滚负控** | — | 换回旧 exe ⇒ **逐字退回 `107461 161431 76 20378`** |

⚠️ **一条已知的不同步**：`src/kernels/gdn_parity.cpp:357,387-389` 的期望值仍按 **sigmoid 硬写** ⇒
该 parity 测试**预期会红**。本次**未跑该测试、未同步改它**（`未验`）。

### 1.2 EOS 默认停机

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/program/generate.cpp` | `:492` | `bool stop_eos = true`（原为 `false`） | 引擎原来**无视模型自己的收尾信号**（`785` 死循环、` *{content}*` 复读） | 默认臂 **496 token（< 600）**、末 token = **`248046` = EOS**、复读 **0 次** |
| `src/program/generate.cpp` | `:1445` | 新增 `--no-stop-eos`（`o.stop_eos = false`） | 保留 raw 续写能力，同时充当**负控** | 负控臂 **600 跑满**、` *{content}*` 复现 ×14 |

**附带登记的两条事实**（不改码，但必须在文档里说清）：

1. 引擎**从不读模型的 `eos_token_id`**（`src` 里 `tokenizer.ggml.` **零命中**），只用硬编码集合 `{248044, 248046}`。
   本模型 `eos_token_id = 248046` ⇒ 当前硬编码**恰好覆盖**。
2. 引擎**不打印** finish/stop 行 ⇒ "停机"只能由 **token 数 < `--max-new`** ＋ **末 token = 248046** ＋
   **结尾无后续生成** 三条合起来判定。**要变成日志自证需再加一次引擎改动（未做）。**

---

## §2 MTP（多 token 预测）

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/program/generate.cpp` | `:8231` | 草稿放置 `drafts[i-1]` → **`drafts[i]`**（`use_mtp` 分支） | 验证窗口消费 `drafts[1..T-1]`（`drafts[0]` 预测的是已经采样出来的那个 token，会被丢掉）⇒ 原来**错位一格** | 接受率 **0.023 → 0.312（窗4）/ 0.632（窗2）**；16-id prompt 上 `GOLD-identical = True`（32/32） |
| `src/core/mtp.cpp` | `:1163-1165` | `max_steps = std::min(max_t_, max_drafts_)`（原为 `max_t_ - 1`） | 窗要消费 `drafts[1..T-1]` ⇒ 草稿器必须产 `max_t_` 跳而不是 `max_t_-1` 跳 | 与上一条**同一次编译**落地；窗 2 / 窗 4 两档的分母都 > 0 |

**结果（`A` 级）**：修改**有效**（接受率站上 0.5，第一次），但**整体仍输**：

| 配置 | decode | `-> N slots` | CPU 专家/层 |
|---|---|---|---|
| 无 MTP（同通道基线） | **55.14** | 1963 | 1.57 |
| MTP 窗 2 | 50.02（44.91–55.13, n=2） | **1839** | 2.01 / 2.17 |
| MTP 窗 4 | 47.55（44.62–50.48, n=2） | **1838** | 4.50 / 7.35 |

⇒ **判"冻结（非判死）"**。机制：① 窗内多行验证开销 ② **草稿头吃 124–125 个槽**。
⚠️ **本条的两条口径提醒**：`--mtp-max-t 1` 是"把 **MTP 自己的窗口**压到最小"（`0` = 跟随 `--spec`），
**不是**"把验证窗口压到地板" —— 后者是 `--spec 2`。要真用 MTP 必须给 `--mtp-max-t 0` 或 ≥2。

---

## §3 预填 / MMQ

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/prefill/prefill.cpp` + 构建开关 | KQUANTS 决策路径；`-DSTRATA_MMQ_KQUANTS=1` | 打开 CUDA 侧 Q4_K / Q5_K / Q5_1 的 MMQ 预填核 | Cyber 的专家是 Q4_K/Q5_K/Q6_K 家族（不是 Q2_0 canvas）⇒ 不开就走不到这些核。上游默认 **OFF**（`CMakeLists.txt:76`） | 探针行 `mmq plan: any=1 fallback=1 layers=40 gu_type=12 d_type=13 supported(gu)=1 supported(d)=1`（`A` 级，`logs/cyber-sys-s146-probe-r1.log:26,39`） |
| `src/prefill/moe_mmq.cu` | `:115`、`:120`、`:208` | **`清单外`**：`#ifdef STRATA_MMQ_KQUANTS` 守着的 Q5_0 / Q4_K / Q5_1 路径与 `GGML_TYPE_Q5_0` 分支 | 开关的**实际落点**，与上一条是同一个改动 | 同上（`gu_type=12 / d_type=13` 即为其证据） |
| `src/prefill/prefill.cpp` | `:701-718` | **诊断探针**（打印上面那一行） | 为了**证明开关真的生效**（此前只有"落了码"、无自证） | ⚠️ **验证完应删**；删后哈希须回到基线 `715E2258…6288`。**当前仍未删**（`未验`） |

⚠️ **速度收益一个数都没有**：A/B 四臂 **0/4 未跑**。现有只有 parity 单线程微基准 ——
Q5_K down **1.58–1.87×**、Q4_K gate **1.17–1.41×**（4 token、单线程、CPU）。**引擎级收益未验。**

---

## §4 工具链（`tools/`）

### 4.1 `tools/iq_pack.py`

| # | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|
| 1 | 新增 `nextn_layers(src)`：读 GGUF 元数据 `<arch>.nextn_predict_layers`（Cyber = 1），`n_layers = total_layers - nextn` | 原实现从**张量**推层数 ⇒ 最高 `blk.40` ⇒ **算出 41**，而引擎的 `n_layers` 是 **40** 层主干 | 修前引擎报 `native dense: cannot open` / `a malformed line: 40 …`（`expert_layout.cpp:279` 的 `l >= n_layers`）；修后正常加载 |
| 2 | 把 MTP 块（`blk.40`）排除出**专家表**与**索引表** | 第 41 行 `l=40 >= 40` 被判 malformed | `native_experts.txt` **41 行** = 1 抬头 + **40 主干层**，最后一行数据以 `39 ` 开头 |
| 3 | 为 **10 个 QSA 层各合成 4 个零张量**（选择器垫片；形状 `[2048,512] bf16` / `[2048,128] bf16` / `[128] f32` / `[128] f32`，取自 `layout.cpp:78-79,105-106`） | 引擎 decode 路径要过一个 indexer 选择器，而 Cyber **没有 indexer**。合成零张量（选满）+ 引擎侧恒等斜坡 ⇒ 两条路都给出"选满 = dense" | pack 打印 `synthetic indexer tensors: 40`；`index.txt` **653 行**（613 + 40）；引擎日志 `77 MiB of weights loaded … (259 canonical tensors skipped)` |
| 4 | `index.txt:1` 抬头记录**源文件名** | `iq_pack.py`（以及 `strata_pack.py`）**都不写 `manifest.json`** ⇒ 溯源必须另找载体 | `index.txt:1` 逐字含源 GGUF 文件名 |

> **回滚路径**：`iq_pack.py` 改动前的版本是未修改的上游文件 ⇒ **git 本身就是备份**
> （`git checkout -- tools/iq_pack.py` 可还原）；改动 diff 已存档（**2,902 B**）。

### 4.2 `tools/make_profile.py`

| # | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|
| 1 | 新增 `--n-layer`（默认 `N_LAYER` = 48） | 原来**只有 `--n-expert`**。`--n-expert 256` 单独用会写出 **48×256** 的表 | 用 `--n-layer 40` 出件的尺寸必须逐字节命中真式（见下） |
| 2 | `read_profile` / `read_trace` / `write_profile` / `rank_profile` **四个函数**都加 `n_layer` 参数 | 常量 `N_LAYER` 原来在**四处**被直接引用；只改一处会漏 | 四处都传 `n_layer` |
| 3 | `main()` 改成**关键字传参** | 给 `read_profile` 插了 `n_layer` 参数后，若调用点仍用位置传参，`ne` 会绑到 `n_layer` 上 ⇒ **静默错** | 代码复核 |
| 4 | 用法文档串同步加 `--n-layer 40` | 一个事实只写一处，但**引用者必须全改** | 文档串在场 |

**为什么必须加（「纯假绿」的机制）**：工具自己的 round-trip 断言**抓不到**这个错 ——
`write_profile` 与 `read_profile` 比对的是**同一个模块常量**，错文件照样通过。
**部分修正（只加新结论，旧结论不动）**：假绿只对**工具自己的自检**成立；引擎端**会拦**
（`src/core/expert_cache.cpp:42-48` 报 `read_expert_profile: %s is %ux%u but this model is %lldx%lld`，
`:69` 逐对查越界）⇒ **大声失败，但代价是"一整个 pack 之后"才发现**。

**判据（尺寸真式）**：`bytes = 24 + 4·len(ranked) + 4·N_LAYER·n_expert`。
实测 `81,944 = 24 + 4·10240 + 4·40·256` ✅ **逐字节命中**。
（塌缩式 `24 + 8·N_LAYER·n_expert` 只在"满排名"时成立 —— 加"部分排名"模式后立刻变假绿 ⇒ **写要写真式**。）

---

## §5 数据产物（不是源码，但必须一起说）

| 产物 | 生成方式 | 与上游的关系 |
|---|---|---|
| `pack/iq-native/` | `tools/iq_pack.py --gguf <GGUF> --out <dir> --compat-bf16` | 上游工具 + §4.1 的四处改动 |
| `pack/iq-native/SOURCE.sha256` | **手工写入**的溯源文档（`<键> <值>` 格式，三个有效键 `sha256` / `bytes` / `path`） | 上游**没有**这件东西；补它是因为 `iq_pack.py` 不写 manifest ⇒ 红线"pack 与源哈希回比"原判据不存在 |
| `profile/expert-profile-40x256-trace-cur.bin` | `--dump-routing` 采 8 份 trace → `tools/make_profile.py --no-base --n-layer 40 --n-expert 256` | 上游工具 + §4.2 的四处改动 |
| `engine/tokenize/` + `engine/vocab_bytes.tsv` | 由 `tools/strata_tokenizer.py` 的 `extract()` 取出 | 引擎**自己不带 tokenizer**（`generate.cpp:9-12` 逐字 "There is no tokenizer here."）⇒ 包内补齐 |
| `run.cmd` + `run.ps1` | 新写的一键启动器（纯 ASCII、零 Python 依赖） | 上游**没有**；补它是因为引擎只吃 token id |

⚠️ **`SOURCE.sha256` 的一个已知瑕疵**：它的 `verify` 段是**上一代 pack** 的读数
（`dense.bin 55,706,112 B` / `630 tensors` / `142 转换`，而当前实际是 `80,835,072 B` / `653 行` / `140 转换`），
且文件时间戳比 pack **早 4 分钟**。
**⇒ 只有 `sha256` / `bytes` / `path` 三键有效**（已按这三键回比一致）；
**原件不就地改**（冻结产物），以当前实测指纹为准。

---

## §6 改动纪律（本项目自己踩过才写下的）

1. **一次编译原则**：能合并的改动搭同一趟车，**一次编完**。
2. **改前备份（含二进制）**：备份落在 `<workdir>/_backups/<日期>-<主题>/`，含 `.before` / `.after` 二进制与源文件。
3. **不许就地改冻结产物**：结论被推翻时**只加新结论，不动旧结论**，并在旧结论处留指针。
4. **一个事实只能有一个出处**：别处只许指向、不许复制。
5. **产物必须自证**：改动后要给出"它确实按我以为的路跑了"的自证字段（计数器 / 探针行）。
6. **负面结论也是资产**：判死的路要连证据一起写下来（见 `08-known-issues.md` §1）。

---

## §7 出处

| 结论 | 素材 |
|---|---|
| GDN z 闸门 6 处改动、七条判据、回滚负控 | `rel/z闸门修复收口.md` §2、§3；`rel/状态收拢-v3.md` §1b |
| `gdn_parity` 未同步 | `rel/z闸门修复收口.md` §5.3 |
| EOS 默认 true、`--no-stop-eos` | `rel/z闸门修复收口.md` §4.2；`rel/状态收拢-v3.md` F3；`rel/_append-166.md` |
| 引擎不读 `eos_token_id`、硬编码 `{248044,248046}` | `rel/z闸门修复收口.md` §4.2 |
| MTP 草稿放置 `drafts[i]` | `rel/状态收拢-v3.md` §1b（`generate.cpp:8231`）、F4 |
| `mtp.cpp` `max_steps` | `rel/_append-147.md` #134；`rel/状态收拢-v3.md` §1b |
| MTP 接受率与三档读数、124–125 槽 | `rel/_s144-a2-report.md`；`rel/_s145-a3-report.md`；`rel/_s149-optlist-diff.md`（窗口 6 行） |
| `--mtp-max-t` 的口径订正 | `rel/跳步记录.md` #101；`rel/HANDOFF-MTP与本轮交接.md` |
| KQUANTS 落点与自证行 | `rel/状态收拢-v3.md` F7、§1b；`rel/_s146-devbatch-report.md` §2.1、§7 |
| KQUANTS 的速度收益 0/4、parity 微基准 | `rel/状态收拢-v3.md` §3.1 |
| `iq_pack.py` 四处改动与回滚路径 | `rel/RUN-S11-试驾实录-阻断链.md` §3、§8.3；`rel/RUN-S7-S10-S11-产物就位与试驾准备.md` §（`iq_pack.py` 改动与 diff） |
| `index.txt` 653 行、`native_experts.txt` 41 行 | `rel/发货参数与显卡分档调优.md` §2；`rel/跳步记录.md` #4 |
| `make_profile.py` 四处改动与 C9 | `rel/RUN-S7-S10-S11-产物就位与试驾准备.md` §1.1、§1.2；`rel/总纲与分支协议.md` C9 |
| 尺寸真式 C8 | `rel/总纲与分支协议.md` C8 |
| `SOURCE.sha256` 三键有效、`verify` 段过期、原件不就地改 | `rel/RUN-S11-试驾实录-阻断链.md` §17.3、§17.4 |
| 无 tokenizer / 大页拒绝 / 一次编译 / 备份纪律 | `<repo>/src/program/generate.cpp:9-12`；`rel/_append-147.md` #134；`rel/状态收拢-v3.md` §7 |
