# 00 · 总览：这是什么、解决什么问题、达成什么

> 对象：本 fork（Strata 引擎，MIT，上游 `Niko1221/Strata`）+ **Cyber-Tiel-Coder-35B-A3B**（GGUF，`qwen35moe`）。
> 证据等级：`A` = 本机自实测 ｜ `B` = 同机对照实现 ｜ `C` = 外部实测 ｜ `D` = 推导或出处自报。
> 本机参照机（下文所有"本机"）：RTX 4080 SUPER（32760 MiB）+ i7-13790F（8P+8E）+ 96 GiB DDR4-2667。

---

## §1 这是什么

把一个 **35B 总参 / 约 3B 激活**的 MoE 模型，跑在**专家权重流式**的配置下：

| 机制 | 做法 |
|---|---|
| 专家权重常驻**内存** | 18.33 GiB 专家竞技场钉在宿主内存（`cudaHostRegister PORTABLE`） |
| 专家按需**上卡** | 显存里放一个专家缓存（槽的多少由 `--vram-reserve-mib` 决定） |
| 未命中的专家 | 交给 **CPU 线程池**就地计算（或多 token 时进 staging 走 PCIe） |
| 模型来源 | 22.75 GB 的 GGUF，专家不从 pack 读，引擎直接读 GGUF 原生张量 |

结果是：**一张 8 GB 级显存的卡 + ≥32 GB 内存**在额度上足够跑起来（`A` 级读数与 8 GB 卡同档，但 8 GB 卡本身**未实测**，见 §4）。

---

## §2 模型与权重的确定身份

| 项 | 值 |
|---|---|
| 架构 | `qwen35moe`：40 层主干 / 256 路由专家 / top-8 / hidden 2048 / FFN 512 |
| 注意力 | **10 层全注意力 + 30 层 GDN 线性注意力** |
| 第 41 块 | MTP 头（`nextn_predict_layers = 1`；引擎的 `n_layers` = 40 层主干） |
| 权重文件 | `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`，**22,749,880,640 B** |
| sha256 | `0BBCF3CC9BE4C976BAD20E641BAF629DAD9C178D39EBDC9CD72129179943C06A`（MTP 头是 Q3_K） |
| 词汇表 | 248,320 |

⚠️ 同仓库还有 IQ3_XXS / IQ4_XS / Q2_K_XL 等其它量化 —— **那些不是本文所有读数所对应的制品**。下错文件会得到另一套数字。

---

## §3 实测效果（`A` 级）

### 3.1 验收档：流式（`--vram-reserve-mib 24568`，专家缓存 4.20 GiB）

| 项 | 读数 | 出处 |
|---|---|---|
| decode | **57.68 / 58.60 / 61.56 / 61.83 / 63.48 tok/s**（同批同配置） | `rel/logs/cyber-s150-{f1,f2,b2,kqoff1,kqoff2}.log` |
| prefill | **1736–2203 tok/s**；TTFT ≈1.8–1.9 s @2002 tokens | 同上 |
| 专家缓存 | 规划 1960 slots / 生效 **2348 slots / 4.20 GiB 显存** | `rel/logs/cyber-s150-u1.log.err` |
| 命中率 | **0.7794**（6948 / 8915） | `rel/logs/cyber-s150-f2.log` |
| CPU 专家池 | **1.54 个专家/层**、**5.42–5.98 ms/token**、**20.5–22.5 GB/s** | 同上 |
| 宿主专家银行 | 钉住 **19,675,480,064 B** | `rel/logs/cyber-s150-f2.log.err` |
| per token | **11.2–11.4 ms/token** | `rel/_append-149.md` §145 |

### 3.2 专家全常驻档（`--vram-reserve-mib 1500`，32 GB 卡）

| 项 | 读数 |
|---|---|
| decode | **134.18 / 136.02 tok/s**（同批交错，配对下界 +70.70） |
| 生效缓存 | **10240 slots / 18.32 GiB 显存** |
| CPU 专家池 | **完全不工作**：`0.000 ms/token`、`0 dispatches`、`0.00 专家/层` |
| 命中率 | **0.0000**（槽够多，不存在未命中） |

⇒ 该档**不是本项目的验收形态**（验收基准是流式档），但它标定了"CPU 池成本 = 多少"。

### 3.3 正确性

| 项 | 读数 |
|---|---|
| 与外部 llama.cpp 参考 | 流式验收档：**逐位相同到生成位 6**（index 0–6），index 7 是近并列翻转 |
| 近并列判据 | 我们的选择落在参考 **top-10 第 2 位**，`Δlogit = 0.325 nat` |
| 常驻档 | 首分岔推到 **index 15**（同 profile、只差 reserve） |
| 输出可复现性 | 同一配置下 token id **完全可复现**（多次逐字节相同） |
| 问答抽测 | 10/10 全对，且参考（同 id、贪心）给出同样答案 |

---

## §4 边界与不支持的东西（**发货时必须一起说**）

| 项 | 状态 |
|---|---|
| **没有 `serve` 子命令** | ⚠️ `strata.exe serve` ⇒ `unknown argument: serve`。**有 `--serve` 旗标**（参数解析可过），但**未使用过 ⇒ 未验** ⇒ **本包只有命令行形态，不要承诺 API / 服务端** |
| **引擎自己不带 tokenizer** | ⚠️ 仍然如此（源码 `src/program/generate.cpp:9` 逐字 "There is no tokenizer here."；缺失时报 `--tokens is required`）。分词必须外部提供，见 `03-run.md` |
| **只在 Ada 实机验证过** | ⚠️ 手上只有 RTX 4080 SUPER（SM 8.9）。构建产物含 `sm_75 / sm_86 / sm_89 / sm_120` 四代机器码（`cuobjdump` 实测），但 **30 系 / 20 系 / 50 系没有实机跑过**。判据是"文件里确实装了那一代的机器码"，不是"跑过" |
| GTX 16 / 10 系及更老 | ❌ 不支持（引擎要求 `sm_75` 起，构建期 CMake 直接 `FATAL_ERROR`） |
| AMD / 集显 / Apple | ❌ 不支持（本 fork 只用 CUDA 路径） |
| **8 GB 卡上的实跑** | ❌ **未实测**。额度账与 8 GB 同档（都是 ≈4.2 GiB 可用），机制上可外推，但**没有在 8 GB 卡上跑过** |
| PLE（per-layer embedding） | ❌ 不适用：Cyber **没有 PLE 表**，`--no-ple` 是"无表可用"下的必然设置。试打开会 `ENGINE_FAIL`：`per_layer_token_embd.weight is not in …gguf` |
| MTP 投机 | ⚠️ **冻结**（非判死）：接受率已被新 profile 抬到 0.550（窗2）/0.583（窗4），但 decode 仍输（同通道基线 55.14 vs 50.02 / 47.55）。**发货先按 `--spec 2` 走，别开更大窗** |
| 多卡（`--peer-device` / `--expert-cache-device1..3`） | ❌ **完全未验** |
| `--vram-elastic` / `--batch` | ❌ 未验（只在 `--serve` 下有意义） |
| 大页（large pages） | ⚠️ 收益**未分离**：记 `+0 到 +6%`。授予 `SeLockMemoryPrivilege` 的通道下引擎报 `large pages (2097152 B)`；普通直启报 `large pages refused … VirtualAlloc error 1314` |

---

## §5 三条最容易踩的读数纪律

1. **速度结论必须同批交错**：同配置**同批**散布 ≥ **5.75 tok/s**，**跨批**漂移 **15–20%**。跨批数字不可直比。
2. **切换配置后的第一条臂会被污染**：实测 43.15 vs 63.48 tok/s（池带宽 11.4 vs 22.5 GB/s）⇒ **报数请跑第二条**。
3. **历史 88–89 tok/s 不可引用**：那是"输出乱码时代"的读数（路由单调 ⇒ 命中率虚高）。已作废。

---

## §6 出处

| 结论 | 素材 |
|---|---|
| 流式档全部读数 / 命中率 / CPU 池 / 宿主银行 | `rel/发货参数与显卡分档调优.md` §1.1、§3.1；`rel/_s150-baseline-arms.md` §0、§2.3 |
| 常驻档 134.18 / 136.02、slots 10240、pool 0.000 | `rel/_s150-baseline-arms.md` §2.3–§2.6 |
| 参考串与逐位对照、近并列判据 | `rel/_s141-ref-2003-report.md` §0、§2；`rel/_s150-baseline-arms.md` §4 |
| index 7 / 15 分岔、profile 对照 | `rel/_s144-a2-report.md`（对照读数表）；`rel/_s145-a3-report.md`（读数表） |
| MTP 冻结、接受率 0.550 / 0.583 | `rel/_s145-a3-report.md`（结论） |
| 边界（无 serve / 无 tokenizer / 只在 Ada 验过） | `rel/发货参数与显卡分档调优.md` §0.1、§1.3、§4.1；`src/program/generate.cpp:9`、`:1575`（错误文案） |
| PLE 无表 | `rel/跳步记录.md` #42 |
| 噪声底、污染臂、88–89 作废 | `rel/_s149-optlist-diff.md`（`--pool-workers 6` 行、`STRATA_PREFILL_CPU_SHARE` 行）；`rel/发货参数与显卡分档调优.md` §6 R7、R8；`rel/状态收拢-v3.md` §0、F11 |
| 逐项拆分的完整出处 | 见 `05-performance.md` §5 |
