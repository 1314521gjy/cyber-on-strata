# 00 · 总览：这是什么、解决什么问题、达成什么

> 对象：本 fork（Strata 引擎，MIT，上游 `Niko1221/Strata`）+ **Cyber-Tiel-Coder-35B-A3B**（GGUF，`qwen35moe`）。
> 证据等级：`A` 本机自实测 ｜ `B` 同机对照实现 ｜ `C` 外部实测 ｜ `D` 推导或出处自报。
> 测量环境（下称"本机"）：RTX 4080 SUPER（32760 MiB）+ i7-13790F（8P+8E）+ 96 GiB DDR4-2667。

---

## §1 这是什么

把 **35B 总参 / 约 3B 激活**的 MoE 跑在**专家流式**配置下：

| 机制 | 做法 |
|---|---|
| 专家权重常驻**内存** | 18.33 GiB 专家竞技场钉在宿主内存（`cudaHostRegister PORTABLE`） |
| 专家按需**上卡** | 显存里放专家缓存，槽数由 `--vram-reserve-mib` 决定 |
| 未命中的专家 | 交给 **CPU 线程池**就地算；多 token 时部分进 staging 走 PCIe |
| 权重来源 | 22.75 GB GGUF；**专家不从 pack 读**，引擎直接读 GGUF 原生张量 |

⇒ **一张 8 GB 级显存的卡 + ≥32 GB 内存**在额度上足够（本机读数与 8 GB 同档；但 8 GB 卡本身**未实测**，见 §4）。

---

## §2 模型与权重的确定身份

| 项 | 值 |
|---|---|
| 架构 | `qwen35moe`：40 层主干 / 256 路由专家 / top-8 / hidden 2048 / FFN 512 |
| 注意力 | **10 层全注意力 + 30 层 GDN 线性注意力** |
| 第 41 块 | MTP 头（`nextn_predict_layers = 1`；引擎 `n_layers` = 40 层主干） |
| 权重 | `Cyber-Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf`，**22,749,880,640 B**（MTP 头是 Q3_K） |
| sha256 | `0BBCF3CC9BE4C976BAD20E641BAF629DAD9C178D39EBDC9CD72129179943C06A` |
| 词汇表 | 248,320 |

⚠️ 同一发布页还有 IQ3_XXS / IQ4_XS / Q2_K_XL 等其它量化 —— **那些不是本文读数所对应的制品**。

---

## §3 实测效果（`A` 级）

### 3.1 验收档：流式（`--vram-reserve-mib 24568`）

| 项 | 读数 |
|---|---|
| decode | **57.68 / 58.60 / 61.56 / 61.83 / 63.48 tok/s**（同批同配置） |
| prefill | **1736–2203 tok/s**；TTFT ≈1.8–1.9 s @2002 tokens |
| 专家缓存 | 规划 1960 slots / 生效 **2348 slots / 4.20 GiB 显存** |
| 命中率 | **0.7794**（6948 / 8915） |
| CPU 专家池 | **1.54 个专家/层**、**5.42–5.98 ms/token**、**20.5–22.5 GB/s** |
| 宿主专家银行 | 钉住 **19,675,480,064 B** |
| per token | **11.2–11.4 ms/token** |

### 3.2 专家全常驻档（`--vram-reserve-mib 1500`，32 GB 卡）

| 项 | 读数 |
|---|---|
| decode | **134.18 / 136.02 tok/s**（同批交错，配对下界 **+70.70**） |
| 生效缓存 | **10240 slots / 18.32 GiB 显存** |
| CPU 专家池 | **完全不工作**：`0.000 ms/token`、`0 dispatches`、`0.00 专家/层` |
| 命中率 | **0.0000**（槽够多，不存在未命中） |

⇒ 该档**不是验收形态**（验收基准是流式档），但它标定了"CPU 池成本 = 多少"。

### 3.3 正确性

| 项 | 读数 |
|---|---|
| 与外部 llama.cpp 参考 | 流式验收档：**逐位相同到生成位 6**（index 0–6）；index 7 是**近并列翻转**（我们的选择落在参考 top-10 第 2 位，Δlogit = 0.325 nat） |
| 常驻档 | 同 profile、只差 reserve ⇒ 首分岔推到 **index 15** |
| 输出可复现性 | 同一配置下 token id **完全可复现**（多次逐字节相同） |
| 问答抽测 | **10/10 全对**，参考（同 id、贪心）给出同样答案 |

---

## §4 边界与不支持的东西（**发货时必须一起说**）

| 项 | 状态 |
|---|---|
| **没有 `serve` 子命令** | ⚠️ `strata.exe serve` ⇒ `unknown argument: serve`。**有 `--serve` 旗标**（参数解析可过），但**未使用过 ⇒ 未验** ⇒ **只有命令行形态，不要承诺 API / 服务端** |
| **引擎自己不带 tokenizer** | ⚠️ `src/program/generate.cpp` 抬头逐字 "There is no tokenizer here."；缺 `--tokens` 时报 `--tokens is required`。分词必须外部提供（见 `03-run.md`） |
| **只在 Ada 实机验证过** | ⚠️ 手上只有 RTX 4080 SUPER（SM 8.9）。产物含 `sm_75 / sm_86 / sm_89 / sm_120` 四代机器码（`cuobjdump` 实测），但 **30 / 20 / 50 系没有实机跑过**。判据是"文件里确实装了那一代的机器码"，不是"跑过" |
| GTX 16 / 10 系及更老 | ❌ 不支持（引擎要求 `sm_75` 起，构建期 `FATAL_ERROR`） |
| AMD / 集显 / Apple | ❌ 不支持（本 fork 只用 CUDA 路径） |
| **8 GB 卡上的实跑** | ❌ **未实测**。额度账与 8 GB 同档（都是 ≈4.2 GiB 可用），机制上可外推，但没跑过 |
| PLE（per-layer embedding） | ❌ 不适用：Cyber **没有 PLE 表**，`--no-ple` 是"无表可用"下的必然设置。试打开会失败（`per_layer_token_embd.weight is not in …gguf`） |
| MTP 投机 | ⚠️ **冻结**（非判死）：接受率已抬到 0.550（窗2）/0.583（窗4），但 decode 仍输（同通道基线 55.14 vs 50.02 / 47.55）。**发货先按 `--spec 2` 走，别开更大窗** |
| 多卡（`--peer-device` / `--expert-cache-device1..3`） | ❌ **完全未验** |
| `--vram-elastic` / `--batch` | ❌ 未验（只在 `--serve` 下有意义） |
| 大页（large pages） | ⚠️ 收益**未分离**，记 `+0 到 +6%`。提权通道下报 `large pages (2097152 B)`；普通直启报 `large pages refused … VirtualAlloc error 1314` |

---

## §5 三条最容易踩的读数纪律

1. **速度结论必须同批交错**：同配置**同批**散布 ≥ **5.75 tok/s**；**跨批**漂移 **15–20%**。跨批数字不可直比。
2. **切换配置后的第一条臂会被污染**：实测 43.15 vs 63.48 tok/s（池带宽 11.4 vs 22.5 GB/s）⇒ **报数请跑第二条**。
3. **历史 88–89 tok/s 不可引用**：出自"输出乱码时代"（路由单调 ⇒ 命中虚高）。已作废。

---

## §6 出处

**测量环境**：本机 = RTX 4080 SUPER + i7-13790F + 96 GiB DDR4-2667；日期 2026-10-08 ~ 2026-10-09。

| 结论 | 证据 |
|---|---|
| 流式档 decode / prefill / 命中率 / CPU 池 / 宿主银行 / 文件缓存 | 本机引擎 `--stats` 自报行 |
| 常驻档 134.18 / 136.02、slots 10240、`pool 0.000` | 本机 ABBA 同批交错臂 |
| 参考串、逐位对照、近并列判据 | 外部 llama.cpp 服务端的 `/props`、`return_tokens`、`n_probs` 自报，与本机 `output :` 逐位比（同机对照实现；判据规则见本文档 `04-correctness.md` §3） |
| MTP 冻结、接受率 0.550 / 0.583 | 本机引擎 `speculation` 自报行（本文档 `04-correctness.md` §4.2） |
| 边界（无 `serve` 子命令、无 tokenizer、只在 Ada 验过） | `src/program/generate.cpp` 的抬头注释与参数校验分支；`cuobjdump` 产物架构检查 |
| PLE 无表 | 引擎运行报错 `per_layer_token_embd.weight is not in …gguf`（本文档 `04-correctness.md` §4.1） |
| 噪声底、污染臂、历史 88–89 作废 | 本机同批交错读数的散布统计与作废判据（本文档 `05-performance.md` §4） |
| 逐项拆分的完整出处 | 本文档 `05-performance.md` §5 |
