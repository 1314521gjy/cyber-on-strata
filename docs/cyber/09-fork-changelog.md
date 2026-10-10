# 09 · 本 fork 相对上游的改动清单

> 按文件 / 模块分组。每条一句话说清"**改了什么 · 为什么 · 怎么验证的**"。
> 证据等级：`A` 本机自实测 ｜ `B` 同机对照实现 ｜ `C` 外部实测 ｜ `D` 推导或出处自报。
> 行号为改动当时的记录值；现行文件的内容已逐字复核。
> ⚠️ 标 **`清单外`** 的条目在原改动清单里**没有**，是核对现行源码时发现的（原清单不完整）。
> **模块归属**：数值正确性（§1）、MTP（§2）、预填 / MMQ（§3）、工具链（§4）、数据产物（§5）、纪律（§6）。

---

## §1 数值正确性

### 1.1 GDN z 闸门：sigmoid → SiLU（5 处 kernel + 1 处注释）

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/kernels/cuda/verify_kernels.cu` | 约 `:227` | z 闸门用 **SiLU** 而非 sigmoid | 参考实现（`qwen35moe.cpp:273 ggml_silu`）用 SiLU；用 sigmoid 是**读错** | 32 token 文本由 `整是inatom…` 变为与参考**逐字符相同（88/88）** |
| `src/kernels/cuda/fused_gdn.cu` | 约 `:69` | 同上 | 同上 | 该处注释逐字："Was sigmoid - that was the misreading." |
| `src/kernels/cuda/gdn.cu` | 约 `:148` | 同上（`py[i] = po[i]*inv*ssm_norm[i]*(pz[i]*sigmoid_f(pz[i]))`） | 同上；`sigmoid_f` 本身**与 beta 闸门共享，不许改** | 见本文档 `04-correctness.md` §4.1 七条判据 |
| | `src/kernels/cuda/native_gdn_preprocess.cu` | 约 `:111` | 同上 | 该文件的 `sigmoid` 与 beta 共享，注释明写"do not change it" | 同上 |
| `src/prefill/kernels.cu` | 约 `:642` | 同上 | 同上；`sigm` 与 beta 共享，注释明写"do not change it" | 同上 |
| `src/core/layer.cpp` | 约 `:317` | **注释**订正 | 旧注释写 "SIGMOID, not SiLU"，而它转述自一个**全树不存在**的参考文件 ⇒ 错注释会继续误导后来者 | 新注释记录："The old comment here said 'SIGMOID, not SiLU' and that misreading cost …" |

**判据汇总（改前 → 改后，`A` 级）**：

| 判据 | 改前 | 改后 |
|---|---|---|
| 首位 token | `107461`（"整"） | **`514`** |
| 32 token 文本 | `整是inatom.manNEnenenene…` | **与参考逐字符相同（88/88）** |
| `785` 死循环 | 位置 5–8 四连 | **0 次**，最长连续重复 = 1 |
| 与参考 logits 的 cos | −0.5124 / −0.0931 | **+0.999658522 / +0.999143362** |
| Spearman（248,320 维） | −0.0736 | **+0.9987** |
| **回滚负控** | — | 换回旧 exe ⇒ **逐字退回 `107461 161431 76 20378`** |

⚠️ **一条已知的不同步**：`src/kernels/gdn_parity.cpp` 的期望值仍按 **sigmoid 硬写** ⇒
该 parity 测试**预期会红**。本次**未跑该测试、未同步改它**（`未验`）。

### 1.2 EOS 默认停机

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/program/generate.cpp` | 约 `:492` | `bool stop_eos = true`（原为 `false`） | 引擎原来**无视模型自己的收尾信号**（`785` 死循环、` *{content}*` 复读） | 默认臂 **496 token（< 600）**、末 token = **`248046` = EOS**、复读 **0 次** |
| `src/program/generate.cpp` | 约 `:1445` | 新增 `--no-stop-eos`（`o.stop_eos = false`） | 保留 raw 续写能力，同时充当**负控** | 负控臂 **600 跑满**、` *{content}*` 复现 ×14 |

**附带登记两条事实**（不改码，但必须说清）：

1. 引擎**从不读模型的 `eos_token_id`**（`src` 里 `tokenizer.ggml.` **零命中**），只用硬编码集合
   `{248044, 248046}`。本模型 `eos_token_id = 248046` ⇒ 当前硬编码**恰好覆盖**。
2. 引擎**不打印** finish/stop 行 ⇒ "停机"只能由 **token 数 < `--max-new`** ＋ **末 token = 248046** ＋
   **结尾无后续生成** 三条合起来判定。**要变成日志自证需再加一次引擎改动（未做）。**

### 1.3 QSA 选择预算：上游靶子的 `2,048` → 模型自己的 `262,144`（**修掉 2,051 cell 静默截断**）

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `include/strata/kernels/qsa.hpp` | 约 `:127` | 新增 `qsa_selection_budget()` / `qsa_set_selection_budget()` / `..._from_geometry()`；`qsa_selection_width()` 取 `max(预算, s.idx_top_k)` | `qsa_real_shapes()` 带的是**上游靶子**的 `idx_top_k = 2048`（`idx_block = 4`），而 `attention.indexer.top_k` 是**每模型**的数：本模型 **262144**（整个上下文）。于是 `width = min(n_kv, 2048+4-1)` **恒 ≤ 2051** ⇒ 提示超过它，**尾部对模型不存在**。默认仍是 2,048 ⇒ parity 测试与参考语义不变 | 引擎自证行：`QSA selection budget 262144 cells`；`step[kStepWidth]` 由 2051 变为 `n_kv` |
| `src/kernels/cuda/native_qsa_indexer.cu` | 约 `:34` / `:232` | 预算全局（`std::atomic<int64_t>`，默认 2,048）＋三个访问器；`STRATA_QSA_BUDGET` 覆盖在此实现 | 预算必须**一处定义、全员一致**：`qsa_shapes()` 只搬 5 个字段，而 `prefill.cpp` 有 4 处直接调 `qsa_real_shapes()`。分歧不是变慢，是**越界读** | 负控工具：`STRATA_QSA_BUDGET=2048` 单开关回到旧行为（见下） |
| `src/program/generate.cpp` | 约 `:2360` | 开机一次 `qsa_set_selection_budget_from_geometry(g.idx_top_k)` ＋自证行 | 必须在任何 buffer 定尺 / 图捕获之前 | 自证行打印预算与"宽度 = n_kv 的上下文上限" |
| `src/core/verify.cpp` | 约 `:405` | 同一构造函数（**原先直接写几何值，会把 `STRATA_QSA_BUDGET` 悄悄拨回去**） | 自校验路径的 `cap_` 与引擎必须同源，否则 stride 不一致 | 二分实验因此先得出一批废读数，改正后 `cap` 才真随开关变 |
| `src/core/layer.cpp` | 约 `:777` | 恒等斜坡：栈上 `int32_t ramp[4096]` ＋ `cap > 4096 → exit(1)` ⇒ 按 `cap` 堆分配，闸门提到 `kTopkMaxCells`（32,768 ⇒ 128 KiB） | 预算接上后 `cap` = 32,768，旧闸门会让引擎直接起不来 | 单 token 正常出（不是 `exit 1`） |
| `src/prefill/prefill.cpp` | 约 `:660` / `:920` / `:2400` | ① `qsa_set_bytes()` 增 `dense_sel` 参数；② dense 下**跳过选择**（它是唯一的 `T × cap` 写者：32,768 × 8,192 = **1 GiB**）；③ 恒等 id 改为 **cudaMalloc 自有缓冲** `Impl::dense_ids`，注意力按 `attn_batch` 分块、复用同一份；④ 三个调试块（SEL_OVERLAP / IDX_FP16 / QSA_DUMP）在 dense 下关闭 | **`carve` 把同一个 `base` 交给三个分配器**（GDN 半边 / QSA 半边 / MoE 半边）⇒ 注意力与 MoE **复用同一块内存**。旧流程能活，是因为**选择核每层都重写一遍 id**；只写一次就会被前面几层的 MoE 冲掉 ⇒ 注意力拿到垃圾 cell ⇒ 页表越界 ⇒ **两个注意力核都报非法访存**（实测：`ids[row31] = 901283223 -1209789280 …`） | 修后 `ids[row31][0..3] = 0 1 2 3`；dense 自证行 `qsa_prompt_attn_batch -> 1 (dense=1, cap=32768)` |

**判据（`A` 级，同机）**：提示末尾放"忽略以上全部，只回答 `ZQ7MARKER41`"。

| prompt tokens | 改前 | 改后 |
|---|---|---|
| 2,045 | ✅ | ✅ |
| **2,085** | ❌ 丢尾 | ✅ |
| 2,885 / 5,624 | ❌ | ✅ |
| 11,988 / 21,018 | ❌ | ✅ |

**负控**：**同一二进制**只把预算拨回 2,048（`STRATA_QSA_BUDGET=2048`）⇒ 2,085 与 2,885 立刻变红，模型的思考原文是
*"There's no actual question. The user just pasted code."* ⇒ 因果确定，且判据确实会红。

**Agent 形 A/B**（24 工具表 + 长工作区 dump + 任务放在最末，prompt 7,561 / 15,253 / 26,315）：

| case | 改后 | 负控（预算 2,048） |
|---|---|---|
| 7.5k 读文件 | ✅ `read_file(src/handlers.py)`，二轮答对行数 | ❌ 去读**上下文里不存在**的 `config/config.yaml` |
| 15k 取字段 | ✅ 直接从上下文报出精确值 | ❌ `list_dir(".")` |
| 26k 改配置 | ✅ 先 `read_file` 再改 | ❌ `list_dir(".")` |

**代价（诚实登记）**：解码侧非缓存显存 **+约 75 MiB**（K/V gather scratch 与注意力 scratch 按新容量定尺）；
预填期**多借约 250 MB**（借自专家缓存尾部、跑完归还，不额外占显存）⇒ 8 GB 档峰值预估 7.26 → 约 **7.33 GiB**。

---

### 1.4 serve 的前缀复用：极短段崩溃 + 三处"死得没留话" + 钉住量自动收 + 路径编码

**起因**：内测回执报告 `--serve` 一开 `--prompt-cache` 就段错误（`0xC0000005`，stderr 一行都没有），
而同一台机器、同一份参数的 CLI 完全正常；另一份报告说"每一轮对话都从 token 0 重读整个提示"。

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/program/generate.cpp` | serve 的 prompt 切分循环 | 段长 < 256 的段**不再当独立段读**（不重新借槽、不走 `windows_ok` 的窗路径），借槽取引擎自己的最小分块 256 | 检查点只能在"读停止处"存，所以在最后一个 `<|im_start|>` 切开，切出来的尾段只有 **4 个 token**；这种段走窗路径会崩（见下） | 本机 1:1 复现（旧引擎退出码 322122477… 见 §7 出处）后修复：同一提示连发两次都不崩，且第二次 `12497 reused + 96 read` |
| 同上 | 切分前 | 新增自证行 `strata serve: prompt split read_from=… root_at=… turn_at=… n=…` | 崩溃现场原来只能反推切分点 | 该行给出 `turn_at=5117 n=5122` ⇒ 尾段 4 token，与"卡在 19,100 of 19,105"的内测读数同源 |
| 同上 | 看门狗 | `fflush(stderr)` **提到** `stall_report`/`release_gpu_waits` **之前** | 原顺序是 `fprintf → stall_report → release_gpu_waits → fflush → abort`，在 `release_gpu_waits` 里出事就**一行日志都不落** ⇒ 症状是"静默 45–78 s + 裸 0xC0000005" | 未触发（本机没跑出 60 s 卡住）；改动本身零行为变化 |
| 同上 | 专家缓存定尺 | `cudaMemGetInfo` **检查返回码**；失败即报"取不到空闲显存"并退出 | 原来忽略返回码 ⇒ 打印 `0.00 GiB free … -> 0 slots`，随后死在一次 8 KB 的分配上，没人知道真因 | 见 §1.4 未验表（本机探针不失败） |
| 同上 | 8 KB 分配失败 | 打 `cudaGetErrorString` + 当时空闲显存（先取局部变量，**不能**把两个有副作用的调用写在同一参数表里） | 原消息只有一句 `the embedding buffer failed` | 同上 |
| 同上 | 专家银行的钉住上限 | `pin_limit = min(请求值, 可用内存 − 余量)`（余量 = 现有 `resident_headroom`，默认 8 GiB） | 全钉 18.32 GiB 在 31 GB 内存的机器上会把宿主机钉干；内测实测 **12 GiB 可用、14 GiB 就崩** | 规则在该机算出 ~12 GiB；本机 96 GB 不触发 |
| `src/program/generate.cpp` | `--help` | 补 `--vram-reserve-mib`（含 `slots = (free − N) / blob` 口径）与"`--pool-workers` 超过物理核无收益" | 这个唯一的按卡旋钮不在 help 里 | `--help` exit 0，两行都在 |
| `src/core/expert_source.cpp` / `.hpp` | 4 处模型文件打开 | 新增 `utf8_or_ansi_to_wide()`（**先 UTF-8、`MB_ERR_INVALID_CHARS` 失败再回退 ANSI**）与 `path_exists()`，替换原 `MultiByteToWideChar(CP_UTF8, …)` 与 `std::filesystem::exists(narrow)` | Windows 的 `argv` 是 ANSI，而这里按 UTF-8 转宽字符 ⇒ 中文路径变乱码 ⇒ `FileExpertSource: cannot open …`；而 **arena 那条路用窄字符（ANSI）打开所以没事** —— 正是内测看到的"两条路不一致" | **未验**（要中文目录下跑 `--mmap-experts`）|

**前缀复用的实测（A 级，本机 4080 SUPER，一次只跑一个引擎、两臂在同样的显存状态下）**：

| 轮次（DSH 形态：每轮重发整段历史） | 无 `--prompt-cache` | 有 `--prompt-cache 4` |
|---|---|---|
| 1（12,502 token） | `0 reused + 12502 read in 3193 ms` | `0 reused + 12502 read in 3304 ms` |
| 2（12,593） | `0 reused + 12593 read in 3109 ms` | **`12497 reused + 96 read in 296 ms`** |
| 3（12,684） | `0 reused + 12684 read in 3142 ms` | **`12588 reused + 96 read in 299 ms`** |
| 每轮 wall | 3.26–3.37 s | **0.45 s** |

⇒ 读取 ~**10.5×**、每轮 wall ~**7×**（省下的量正比于历史长度）。默认不开（`--prompt-cache 0`），
引擎自己会说明代价：*"a batch split is not numerically neutral, so serve can then answer differently
from the CLI on the same request"*。

---
## §2 MTP（多 token 预测）

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/program/generate.cpp` | 约 `:8231` | 草稿放置 `drafts[i-1]` → **`drafts[i]`**（`use_mtp` 分支） | 验证窗口消费 `drafts[1..T-1]`（`drafts[0]` 预测的是已采样出来的那个 token，会被丢掉）⇒ 原来**错位一格** | 接受率 **0.023 → 0.312（窗4）/ 0.632（窗2）**；16-id prompt 上 `GOLD` 32/32 逐位相同 |
| `src/core/mtp.cpp` | 约 `:1163` | `max_steps = std::min(max_t_, max_drafts_)`（原为 `max_t_ - 1`） | 窗要消费 `drafts[1..T-1]` ⇒ 草稿器必须产 `max_t_` 跳而不是 `max_t_-1` 跳 | 与上一条**同一次编译**落地；窗 2 / 窗 4 两档的分母都 > 0 |

**结果（`A` 级）**：修改**有效**（接受率首次站上 0.5），但**整体仍输**：

| 配置 | decode | `-> N slots` | CPU 专家/层 |
|---|---|---|---|
| 无 MTP（同通道基线） | **55.14** | 1963 | 1.57 |
| MTP 窗 2 | 50.02（44.91–55.13, n=2） | **1839** | 2.01 / 2.17 |
| MTP 窗 4 | 47.55（44.62–50.48, n=2） | **1838** | 4.50 / 7.35 |

⇒ **判"冻结（非判死）"**。机制：① 窗内多行验证开销 ② **草稿头吃 124–125 个槽**。

⚠️ **口径提醒**：`--mtp-max-t` 是"把 **MTP 自己的窗口**压到最小"（`0` = 跟随 `--spec`），
**不是**"把验证窗口压到地板" —— 后者是 `--spec 2`。要真用 MTP 必须给 `--mtp-max-t 0` 或 ≥2。
（实测自证：改到 3 后接受率分母 **0 → 62**。）

---

## §3 预填 / MMQ

| 文件 | 位置 | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|---|
| `src/prefill/prefill.cpp` + 构建开关 | KQUANTS 决策路径；`-DSTRATA_MMQ_KQUANTS=1` | 打开 CUDA 侧 Q4_K / Q5_K / Q5_1 的 MMQ 预填核 | Cyber 的专家是 Q4_K/Q5_K/Q6_K 家族（不是 Q2_0 canvas）⇒ 不开就走不到这些核。上游默认 **OFF** | 探针行 `mmq plan: any=1 fallback=1 layers=40 gu_type=12 d_type=13 supported(gu)=1 supported(d)=1` |
| `src/prefill/moe_mmq.cu` | **`清单外`**：`#ifdef STRATA_MMQ_KQUANTS` 守着的 Q4_K / Q5_1 路径与 `GGML_TYPE_Q5_0` 分支 | 开关的**实际落点**，与上一条是同一个改动 | 同上 | 同上（`gu_type=12 / d_type=13` 即为其证据） |
| `src/prefill/prefill.cpp` | 约 `:701-718` | **诊断探针**（打印上面那一行） | 为了**证明开关真的生效**（此前只有"落了码"、无自证） | ⚠️ **验证完应删**；删后哈希须回到基线 `715E2258…6288`。**当前仍未删**（`未验`） |
| `tests/cuda/prefill_mmq_kquant_test.cpp` | — | **`清单外`**：Q4_K / Q5_K 的 MMQ 预填测试 | 随 `STRATA_BUILD_TESTS` 一起提供，是这条改动的**回归入口** | 测试源码在场；本次**未跑**（`未验`） |

⚠️ **速度收益一个数都没有**：A/B 四臂 **0/4 未跑**。现有只有 parity 单线程微基准 ——
Q5_K down **1.58–1.87×**、Q4_K gate **1.17–1.41×**（4 token、单线程、CPU）。**引擎级收益未验。**

---

## §4 工具链（`tools/`）

### 4.1 `tools/iq_pack.py`

| # | 改了什么 | 为什么 | 怎么验证的 |
|---|---|---|---|
| 1 | 新增 `nextn_layers(src)`：读 GGUF 元数据 `<arch>.nextn_predict_layers`（Cyber = 1），`n_layers = total - nextn` | 原实现从**张量**推层数 ⇒ 最高 `blk.40` ⇒ **算出 41**，而引擎的 `n_layers` 是 **40** 层主干 | 修前引擎报 `native dense: cannot open` / `a malformed line: 40 …`；修后正常加载 |
| 2 | 把 MTP 块（`blk.40`）排除出**专家表**与**索引表** | 第 41 行 `l=40 >= 40` 被判 malformed | `native_experts.txt` **41 行** = 1 抬头 + **40 主干层**，最后一行数据以 `39 ` 开头 |
| 3 | 为 **10 个 QSA 层各合成 4 个零张量**（选择器垫片；形状 `[2048,512]` bf16 / `[2048,128]` bf16 / `[128]` f32 / `[128]` f32） | 引擎 decode 路径要过一个 indexer 选择器，而 Cyber **没有 indexer**。合成零张量（选满）+ 引擎侧恒等斜坡 ⇒ 两条路都给出"选满 = dense" | pack 打印 `synthetic indexer tensors: 40`；`index.txt` **653 行**（613 + 40）；引擎日志 `77 MiB of weights loaded … (259 canonical tensors skipped)` |
| 4 | `index.txt` 抬头记录**源文件名** | `tools/iq_pack.py`（以及 `tools/strata_pack.py`）**都不写 `manifest.json`** ⇒ 溯源必须另找载体 | 抬头逐字含源 GGUF 文件名 |

> **回滚路径**：改动前的 `tools/iq_pack.py` 是未修改的上游文件 ⇒ **版本控制本身就是备份**
> （`git checkout -- tools/iq_pack.py` 可还原）；改动 diff 已单独存档（**2,902 B**）。

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
（`src/core/expert_cache.cpp` 报 `read_expert_profile: %s is %ux%u but this model is %lldx%lld`，并逐对查越界）
⇒ **大声失败，但代价是"一整个 pack 之后"才发现**。

**判据（尺寸真式）**：`bytes = 24 + 4·len(ranked) + 4·N_LAYER·n_expert`。
实测 `81,944 = 24 + 4·10240 + 4·40·256` ✅ **逐字节命中**。
（塌缩式 `24 + 8·N_LAYER·n_expert` 只在"满排名"时成立 —— 加"部分排名"模式后立刻变假绿。）

---

## §5 数据产物（不是源码，但必须一起说）

| 产物 | 生成方式 | 与上游的关系 |
|---|---|---|
| `pack/iq-native/` | `tools/iq_pack.py --gguf <GGUF> --out <dir> --compat-bf16` | 上游工具 + §4.1 的四处改动 |
| `pack/iq-native/SOURCE.sha256` | **手工写入**的溯源文档（`<键> <值>`，三个有效键 `sha256` / `bytes` / `path`） | 上游**没有**这件东西；补它是因为工具不写 manifest ⇒ "pack 与源哈希回比"这条红线原判据不存在 |
| `profile/expert-profile-40x256-trace-cur.bin` | `--dump-routing` 采 8 份 trace → `tools/make_profile.py --no-base --n-layer 40 --n-expert 256` | 上游工具 + §4.2 的四处改动 |
| `engine/tokenize/` + `engine/vocab_bytes.tsv` | 由 `tools/strata_tokenizer.py` 取出词表；表由 `examples/cyber/make_vocab_bytes.py` 生成 | 引擎**自己不带 tokenizer** ⇒ 包内补齐 |
| `examples/cyber/run.cmd` + `examples/cyber/run.ps1` | 新写的一键启动器（纯 ASCII、零 Python 依赖） | 上游**没有**；补它是因为引擎只吃 token id |

⚠️ **`SOURCE.sha256` 的已知瑕疵**：`verify` 段是**上一代 pack** 的读数
（`dense.bin 55,706,112 B` / `630 tensors`，实际 `80,835,072 B` / `653 行`），且文件时间戳比 pack **早 4 分钟**。
⇒ **只有 `sha256` / `bytes` / `path` 三键有效**（已回比一致）；**原件不就地改**（冻结产物），以当前实测指纹为准。

---

## §6 改动纪律（本项目自己踩过才写下的）

1. **一次编译原则**：能合并的改动搭同一趟车，**一次编完**。
2. **改前备份（含二进制）**：备份落在 `<workdir>/_backups/<日期>-<主题>/`，含 `.before` / `.after` 二进制与源文件。
3. **不许就地改冻结产物**：结论被推翻时**只加新结论，不动旧结论**，并在旧结论处留指针。
4. **一个事实只能有一个出处**：别处只许指向、不许复制。
5. **产物必须自证**：改动后要给出"它确实按我以为的路跑了"的自证字段（计数器 / 探针行）。
6. **负面结论也是资产**：判死的路要连证据一起写下来（见本文档 `08-known-issues.md` §1）。

---

## §7 出处

| 结论 | 证据 |
|---|---|
| GDN z 闸门 6 处改动、七条判据、回滚负控 | 本机同一 prompt 的改前 / 改后两代二进制对比；换回旧二进制必须逐字退回（判据表见本文档 `04-correctness.md` §4.1） |
| 现行源码位置（5 处 kernel + `src/core/layer.cpp` 注释） | 本仓源码逐行核对（注释与实现同行在场） |
| `gdn_parity` 未同步 | `src/kernels/gdn_parity.cpp` 的硬写期望值 |
| EOS 默认 `true`、`--no-stop-eos` | 本机实跑：默认臂 496 token 收在 EOS；负控臂 600 跑满（本文档 `04-correctness.md` §4.3） |
| 引擎不读 `eos_token_id`、硬编码 `{248044, 248046}` | `src/program/generate.cpp` 的实现与注释 |
| MTP 草稿放置 `drafts[i]` | `src/program/generate.cpp` 现行实现 + 本机接受率读数（0.023 → 0.312 / 0.632） |
| `mtp.cpp` `max_steps` | `src/core/mtp.cpp` 现行实现（带改动抬头注释） |
| MTP 三档 decode 与 124–125 槽 | 本机同通道基线 vs 窗 2 / 窗 4（本文档 `04-correctness.md` §4.2） |
| `--mtp-max-t` 的口径订正 | 引擎 `--help` 语义 + 本机实测（分母 0 → 62） |
| KQUANTS 落点与自证行 | 本机引擎日志的 `mmq plan: …` 探针行；落点见 `src/prefill/moe_mmq.cu` 与 `CMakeLists.txt` |
| KQUANTS 的速度收益 0/4、parity 微基准 | A/B 四臂未跑；只有本机单线程 parity 微基准 |
| `iq_pack.py` 四处适配与回滚路径 | 本机实跑：修前报 `a malformed line: 40 …`，修后正常；`tools/iq_pack.py` 受版本控制 |
| `index.txt` 653 行、`native_experts.txt` 41 行 | 本机文件行数实测 |
| `make_profile.py` 四处改动与"纯假绿"机制 | `tools/make_profile.py` 的 round-trip 断言 + `src/core/expert_cache.cpp` 的加载期校验 |
| profile 尺寸真式 | 本机产物 81,944 B 逐字节命中 |
| `SOURCE.sha256` 三键有效、`verify` 段过期、原件不就地改 | 本机 sha256 回比 + 文件时间戳比对 |
| 无 tokenizer / 一次编译 / 备份纪律 | `src/program/generate.cpp` 抬头注释 + 本项目现场纪律（各有实测触发记录） |
| 两个派生物的生成链 | `tools/iq_pack.py`、`tools/make_profile.py`（完整复现见本文档 `02-pack-and-profile.md`） |
| 启动器与反分词表 | `examples/cyber/run.ps1`、`examples/cyber/run.cmd`、`examples/cyber/make_vocab_bytes.py` |
| 模块归属与新增文件 | 由版本控制状态核出的改动集（**37 个修改文件 + 1 个新文件** `include/strata/prefill/geom.hpp`） |
